"""Celery tasks for asynchronous review processing.

Defines the background tasks that are dispatched by webhook handlers
and execute the full review pipeline outside the request lifecycle.
"""

import asyncio
import time
from typing import Any

from celery import shared_task
from celery.utils.log import get_task_logger

from code_review_agent.tasks.celery_app import celery_app

task_logger = get_task_logger(__name__)


def _run_async(coro: Any) -> Any:
    """Run an async coroutine from a sync Celery task.

    Creates a new event loop for each task execution to avoid
    conflicts with other async code.

    Args:
        coro: The coroutine to execute.

    Returns:
        The coroutine's return value.
    """
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@celery_app.task(
    bind=True,
    name="code_review_agent.tasks.review_tasks.review_pull_request",
    max_retries=3,
    default_retry_delay=30,
    acks_late=True,
    track_started=True,
)
def review_pull_request(
    self: Any,
    repo_full_name: str,
    pr_number: int,
    installation_id: int,
) -> dict[str, Any]:
    """Execute a full code review on a pull request.

    This is the main entry point for async review processing.
    Orchestrates the entire pipeline: fetch PR, analyze, review,
    generate patches, and post results to GitHub.

    Args:
        self: Celery task instance (bound).
        repo_full_name: Repository full name (owner/repo).
        pr_number: Pull request number.
        installation_id: GitHub App installation ID.

    Returns:
        Review result summary dict.
    """
    task_logger.info(
        "Starting review task: %s#%d", repo_full_name, pr_number
    )

    try:
        result = _run_async(
            _execute_review(repo_full_name, pr_number, installation_id)
        )
        return result

    except Exception as exc:
        task_logger.error(
            "Review task failed: %s#%d - %s",
            repo_full_name, pr_number, str(exc),
        )
        # Retry with exponential backoff
        raise self.retry(exc=exc, countdown=30 * (2 ** self.request.retries))


async def _execute_review(
    repo_full_name: str,
    pr_number: int,
    installation_id: int,
) -> dict[str, Any]:
    """Execute the full review pipeline asynchronously.

    Args:
        repo_full_name: Repository full name.
        pr_number: PR number.
        installation_id: GitHub App installation ID.

    Returns:
        Summary of review results.
    """
    from code_review_agent.config import get_settings
    from code_review_agent.github.auth import GitHubAppAuth
    from code_review_agent.github.client import GitHubClient
    from code_review_agent.github.pr_fetcher import PRFetcher
    from code_review_agent.core.engine import ReviewEngine
    from code_review_agent.core.patch_generator import PatchGenerator
    from code_review_agent.llm.factory import create_llm_provider
    from code_review_agent.rag.embeddings import EmbeddingService
    from code_review_agent.rag.vector_store import VectorStoreClient
    from code_review_agent.rag.retriever import ContextRetriever
    from code_review_agent.logging import get_logger, bind_context, clear_context

    logger = get_logger("review_task")
    settings = get_settings()
    start_time = time.time()

    # Bind context for logging
    bind_context(repo=repo_full_name, pr=pr_number)

    try:
        owner, repo = repo_full_name.split("/", 1)

        # --- Initialize services ---
        auth = GitHubAppAuth(
            app_id=settings.github_app_id,
            private_key_path=settings.github_private_key_path,
            api_url=settings.github_api_url,
        )
        github_client = GitHubClient(auth=auth)
        pr_fetcher = PRFetcher(client=github_client)

        llm = create_llm_provider()
        embedding_service = EmbeddingService()
        vector_store = VectorStoreClient()
        retriever = ContextRetriever(embedding_service, vector_store)

        # --- Fetch PR context ---
        logger.info("fetching PR context")
        pr_context = await pr_fetcher.fetch_pr_context(
            owner=owner,
            repo=repo,
            pr_number=pr_number,
            installation_id=installation_id,
        )

        # --- Skip if no reviewable files ---
        if not pr_context.reviewable_files:
            logger.info("no reviewable files, skipping")
            return {
                "status": "skipped",
                "reason": "no reviewable files",
                "pr_number": pr_number,
            }

        # --- Run review ---
        logger.info("running review engine")
        engine = ReviewEngine(llm=llm, retriever=retriever)
        review_output = await engine.review(pr_context)

        # --- Generate patches ---
        logger.info("generating patches")
        patch_generator = PatchGenerator(llm=llm)
        patch_set = await patch_generator.generate_patches(review_output, pr_context)

        # --- Post results to GitHub ---
        logger.info("posting review to GitHub")
        await _post_review_to_github(
            github_client=github_client,
            pr_context=pr_context,
            review_output=review_output,
            patch_set=patch_set,
        )

        duration = time.time() - start_time

        # --- Store in memory ---
        # Note: Database session management would be handled here
        # in a full implementation with proper session lifecycle

        result = {
            "status": "completed",
            "pr_number": pr_number,
            "repo": repo_full_name,
            "decision": review_output.decision,
            "total_comments": review_output.total_comments,
            "patches_generated": len(patch_set.valid_patches),
            "duration_seconds": round(duration, 2),
        }

        logger.info("review task completed", **result)
        return result

    except Exception as e:
        logger.error("review task failed", error=str(e))
        raise

    finally:
        clear_context()


async def _post_review_to_github(
    github_client: "GitHubClient",
    pr_context: "PRContext",
    review_output: "ReviewOutput",
    patch_set: "PatchSet",
) -> None:
    """Post review results back to GitHub as a PR review.

    Converts the review output into GitHub API format with inline
    comments positioned correctly on the diff.

    Args:
        github_client: Authenticated GitHub client.
        pr_context: PR context.
        review_output: Review results.
        patch_set: Generated patches.
    """
    from code_review_agent.core.schemas import ReviewOutput
    from code_review_agent.core.patch_generator import PatchSet
    from code_review_agent.logging import get_logger

    logger = get_logger("github_post")

    # Map decision to GitHub review event
    event_map = {
        "approve": "APPROVE",
        "request_changes": "REQUEST_CHANGES",
        "comment": "COMMENT",
    }
    event = event_map.get(review_output.decision, "COMMENT")

    # Build inline comments
    inline_comments: list[dict[str, Any]] = []

    for comment in review_output.all_comments:
        if not comment.line_number:
            continue

        # Find the diff position for this file/line
        file_diff = pr_context.diff.get_file(comment.file_path)
        if not file_diff:
            continue

        diff_position = _find_diff_position(file_diff, comment.line_number)
        if not diff_position:
            continue

        # Format comment body with severity badge
        severity_emoji = {
            "critical": "🚨",
            "high": "⚠️",
            "medium": "📝",
            "low": "💡",
            "info": "ℹ️",
        }
        emoji = severity_emoji.get(comment.severity, "📝")
        body = f"{emoji} **{comment.severity.upper()}** ({comment.category})\n\n{comment.message}"

        # Add suggested fix as GitHub suggestion block
        if comment.suggested_fix:
            body += f"\n\n```suggestion\n{comment.suggested_fix}\n```"

        inline_comments.append({
            "path": comment.file_path,
            "position": diff_position,
            "body": body,
        })

    # Build review summary
    summary = f"## 🤖 Automated Code Review\n\n{review_output.summary}"
    if patch_set.valid_patches:
        summary += f"\n\n📦 Generated {len(patch_set.valid_patches)} auto-fix patches."

    # Post review
    try:
        await github_client.create_review(
            owner=pr_context.owner,
            repo=pr_context.repo,
            pr_number=pr_context.pr_number,
            installation_id=pr_context.installation_id,
            body=summary,
            event=event,
            comments=inline_comments if inline_comments else None,
        )
        logger.info(
            "review posted",
            review_event=event,
            inline_comments=len(inline_comments),
        )
    except Exception as e:
        logger.error("failed to post review to GitHub", error=str(e))
        raise


def _find_diff_position(file_diff: Any, target_line: int) -> int | None:
    """Find the diff position for a given line number.

    Maps a new-file line number to the GitHub diff position needed
    for posting inline comments.

    Args:
        file_diff: The FileDiff object.
        target_line: Line number in the new file.

    Returns:
        Diff position number, or None if line is not in the diff.
    """
    for hunk in file_diff.hunks:
        for line in hunk.lines:
            if line.new_line_number == target_line:
                return line.diff_position
    return None

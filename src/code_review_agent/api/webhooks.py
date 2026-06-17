"""GitHub webhook handlers for pull request events.

Receives webhook payloads from GitHub, verifies their signatures,
and dispatches them to the appropriate review pipeline.
"""

from typing import Any

from fastapi import APIRouter, Header, HTTPException, Request, status

from code_review_agent.api.webhook_security import verify_webhook_signature
from code_review_agent.config import get_settings
from code_review_agent.logging import get_logger, bind_context, clear_context

logger = get_logger("webhooks")

router = APIRouter(prefix="/webhooks", tags=["webhooks"])


@router.post(
    "/github",
    summary="GitHub webhook receiver",
    description="Receives and processes GitHub webhook events for pull request reviews.",
    status_code=status.HTTP_202_ACCEPTED,
)
async def github_webhook(
    request: Request,
    x_hub_signature_256: str | None = Header(default=None),
    x_github_event: str | None = Header(default=None),
    x_github_delivery: str | None = Header(default=None),
) -> dict[str, str]:
    """Handle incoming GitHub webhook events.

    Verifies the webhook signature, parses the event type, and
    dispatches pull request events to the review pipeline.

    Args:
        request: The incoming HTTP request.
        x_hub_signature_256: GitHub's HMAC-SHA256 signature header.
        x_github_event: The type of GitHub event (e.g., 'pull_request').
        x_github_delivery: Unique delivery ID for idempotency tracking.

    Returns:
        Acknowledgment response with processing status.

    Raises:
        HTTPException: If signature verification fails or payload is invalid.
    """
    settings = get_settings()

    # Read raw body for signature verification
    body = await request.body()

    # Verify webhook signature
    if not verify_webhook_signature(
        payload=body,
        signature_header=x_hub_signature_256,
        secret=settings.github_webhook_secret,
    ):
        logger.warning("webhook signature verification failed")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid webhook signature",
        )

    # Parse payload
    try:
        payload: dict[str, Any] = await request.json()
    except Exception:
        logger.error("failed to parse webhook payload")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid JSON payload",
        )

    # Bind context for logging
    bind_context(
        delivery_id=x_github_delivery,
        event_type=x_github_event,
    )

    logger.info(
        "webhook received",
        event=x_github_event,
        delivery_id=x_github_delivery,
    )

    # Route based on event type
    try:
        if x_github_event == "pull_request":
            return await _handle_pull_request_event(payload)
        elif x_github_event == "pull_request_review":
            return await _handle_pull_request_review_event(payload)
        elif x_github_event == "ping":
            return _handle_ping_event(payload)
        else:
            logger.debug("ignoring unhandled event type", event=x_github_event)
            return {"status": "ignored", "reason": f"unhandled event: {x_github_event}"}
    finally:
        clear_context()


async def _handle_pull_request_event(payload: dict[str, Any]) -> dict[str, str]:
    """Handle pull_request webhook events.

    Triggers a review for opened, synchronize (new push), and
    reopened actions.

    Args:
        payload: The webhook event payload.

    Returns:
        Processing status response.
    """
    action = payload.get("action", "")
    pr_data = payload.get("pull_request", {})
    repo_data = payload.get("repository", {})

    pr_number = pr_data.get("number")
    repo_full_name = repo_data.get("full_name", "")

    logger.info(
        "pull request event",
        action=action,
        pr_number=pr_number,
        repo=repo_full_name,
    )

    # Only review on specific actions
    reviewable_actions = {"opened", "synchronize", "reopened"}
    if action not in reviewable_actions:
        return {
            "status": "ignored",
            "reason": f"action '{action}' does not trigger review",
        }

    # TODO: Dispatch to Celery task for async review processing
    # from code_review_agent.tasks import review_pull_request
    # review_pull_request.delay(
    #     repo_full_name=repo_full_name,
    #     pr_number=pr_number,
    #     installation_id=payload.get("installation", {}).get("id"),
    # )

    logger.info(
        "review queued",
        pr_number=pr_number,
        repo=repo_full_name,
    )

    return {
        "status": "accepted",
        "message": f"Review queued for {repo_full_name}#{pr_number}",
    }


async def _handle_pull_request_review_event(payload: dict[str, Any]) -> dict[str, str]:
    """Handle pull_request_review events.

    Tracks when reviews are submitted to learn from human reviewer patterns.

    Args:
        payload: The webhook event payload.

    Returns:
        Processing status response.
    """
    action = payload.get("action", "")
    review = payload.get("review", {})

    logger.info(
        "pull request review event",
        action=action,
        state=review.get("state"),
        reviewer=review.get("user", {}).get("login"),
    )

    # TODO: Store human review for learning/memory
    return {"status": "accepted", "message": "Review event recorded"}


def _handle_ping_event(payload: dict[str, Any]) -> dict[str, str]:
    """Handle GitHub ping event sent on webhook creation.

    Args:
        payload: The ping event payload.

    Returns:
        Pong response confirming webhook is active.
    """
    zen = payload.get("zen", "")
    hook_id = payload.get("hook_id")
    logger.info("ping received", zen=zen, hook_id=hook_id)
    return {"status": "pong", "zen": zen}

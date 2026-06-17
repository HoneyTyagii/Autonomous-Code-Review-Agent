"""Structured logging configuration using structlog.

Provides JSON logging in production and colored console output
in development, with contextual binding and correlation IDs.
"""

import logging
import sys
from typing import Any

import structlog
from structlog.types import EventDict, WrappedLogger


def add_app_context(
    logger: WrappedLogger, method_name: str, event_dict: EventDict
) -> EventDict:
    """Add application-level context to all log entries.

    Args:
        logger: The wrapped logger instance.
        method_name: The name of the log method called.
        event_dict: The current event dictionary.

    Returns:
        The enriched event dictionary.
    """
    event_dict.setdefault("app", "code-review-agent")
    return event_dict


def setup_logging(log_level: str = "INFO", json_output: bool = False) -> None:
    """Configure structured logging for the application.

    Sets up structlog with appropriate processors for the environment:
    - Development: colored console output with pretty printing
    - Production: JSON-formatted output for log aggregation

    Args:
        log_level: The minimum log level to output (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        json_output: If True, output JSON logs. If False, use colored console output.
    """
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        add_app_context,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        structlog.processors.TimeStamper(fmt="iso"),
    ]

    if json_output:
        shared_processors.append(structlog.processors.JSONRenderer())
    else:
        shared_processors.append(
            structlog.dev.ConsoleRenderer(
                colors=True,
                exception_formatter=structlog.dev.plain_traceback,
            )
        )

    structlog.configure(
        processors=shared_processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, log_level.upper(), logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )

    # Also configure stdlib logging to respect the same level
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper(), logging.INFO),
    )


def get_logger(name: str | None = None, **initial_context: Any) -> structlog.stdlib.BoundLogger:
    """Get a structured logger instance with optional initial context.

    Args:
        name: Optional component name to bind to all log entries from this logger.
        **initial_context: Additional key-value pairs to bind to the logger.

    Returns:
        A bound structlog logger ready for use.

    Example:
        >>> logger = get_logger("github_client", repo="owner/repo")
        >>> logger.info("fetching PR", pr_number=42)
    """
    logger = structlog.get_logger()
    if name:
        logger = logger.bind(component=name)
    if initial_context:
        logger = logger.bind(**initial_context)
    return logger


def bind_context(**context: Any) -> None:
    """Bind context variables that will appear in all subsequent log entries.

    Useful for adding request-scoped context like correlation IDs.

    Args:
        **context: Key-value pairs to bind to the context.

    Example:
        >>> bind_context(request_id="abc-123", pr_number=42)
    """
    structlog.contextvars.bind_contextvars(**context)


def clear_context() -> None:
    """Clear all bound context variables.

    Should be called at the end of request processing to avoid
    context leaking between requests.
    """
    structlog.contextvars.clear_contextvars()

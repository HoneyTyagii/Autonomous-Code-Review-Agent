"""Webhook signature verification for GitHub webhooks.

Implements HMAC-SHA256 signature verification as specified by GitHub:
https://docs.github.com/en/webhooks/using-webhooks/validating-webhook-deliveries
"""

import hashlib
import hmac

from code_review_agent.logging import get_logger

logger = get_logger("webhook_security")


def verify_webhook_signature(
    payload: bytes,
    signature_header: str | None,
    secret: str,
) -> bool:
    """Verify the GitHub webhook HMAC-SHA256 signature.

    Compares the signature sent by GitHub in the X-Hub-Signature-256 header
    against a locally computed HMAC using the configured webhook secret.
    Uses constant-time comparison to prevent timing attacks.

    Args:
        payload: The raw request body bytes.
        signature_header: The X-Hub-Signature-256 header value (format: "sha256=<hex>").
        secret: The webhook secret configured in the GitHub App settings.

    Returns:
        True if the signature is valid, False otherwise.
    """
    # If no secret configured, skip verification (development only)
    if not secret:
        logger.warning("webhook secret not configured, skipping verification")
        return True

    # Signature header must be present
    if not signature_header:
        logger.warning("missing X-Hub-Signature-256 header")
        return False

    # Parse the signature format "sha256=<hex_digest>"
    if not signature_header.startswith("sha256="):
        logger.warning("invalid signature format", header=signature_header[:20])
        return False

    received_signature = signature_header[7:]  # Strip "sha256=" prefix

    # Compute expected signature
    expected_signature = hmac.HMAC(
        key=secret.encode("utf-8"),
        msg=payload,
        digestmod=hashlib.sha256,
    ).hexdigest()

    # Constant-time comparison to prevent timing attacks
    is_valid = hmac.compare_digest(received_signature, expected_signature)

    if not is_valid:
        logger.warning("webhook signature mismatch")

    return is_valid

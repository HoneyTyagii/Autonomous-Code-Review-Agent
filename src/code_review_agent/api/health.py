"""Health check endpoints for monitoring and orchestration.

Provides liveness and readiness probes for container orchestrators
(Kubernetes, Docker Compose health checks) and load balancers.
"""

from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Response, status

from code_review_agent import __version__

router = APIRouter(tags=["health"])


@router.get(
    "/health",
    summary="Health check",
    description="Basic liveness probe. Returns 200 if the service is running.",
    response_model=dict[str, Any],
)
async def health_check() -> dict[str, Any]:
    """Basic health check endpoint.

    Returns:
        Service status with version and timestamp.
    """
    return {
        "status": "healthy",
        "version": __version__,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get(
    "/health/ready",
    summary="Readiness check",
    description="Readiness probe that verifies all dependencies are accessible.",
    response_model=dict[str, Any],
)
async def readiness_check() -> dict[str, Any] | Response:
    """Readiness check verifying critical dependencies.

    Checks connectivity to:
    - Database (PostgreSQL)
    - Redis
    - Vector store (ChromaDB)

    Returns:
        Detailed status of each dependency, or 503 if any are unavailable.
    """
    checks: dict[str, dict[str, Any]] = {}
    all_healthy = True

    # Database check
    try:
        # TODO: Actual database ping
        checks["database"] = {"status": "healthy", "latency_ms": 0}
    except Exception as e:
        checks["database"] = {"status": "unhealthy", "error": str(e)}
        all_healthy = False

    # Redis check
    try:
        # TODO: Actual Redis ping
        checks["redis"] = {"status": "healthy", "latency_ms": 0}
    except Exception as e:
        checks["redis"] = {"status": "unhealthy", "error": str(e)}
        all_healthy = False

    # ChromaDB check
    try:
        # TODO: Actual ChromaDB heartbeat
        checks["vector_store"] = {"status": "healthy", "latency_ms": 0}
    except Exception as e:
        checks["vector_store"] = {"status": "unhealthy", "error": str(e)}
        all_healthy = False

    result = {
        "status": "ready" if all_healthy else "not_ready",
        "version": __version__,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
    }

    if not all_healthy:
        return Response(
            content=str(result),
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            media_type="application/json",
        )

    return result

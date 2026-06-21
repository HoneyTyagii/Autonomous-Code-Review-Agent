"""Tests for health check endpoints."""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from code_review_agent.main import create_app


@pytest.fixture
def client() -> TestClient:
    """Create a FastAPI test client."""
    return TestClient(create_app())


@patch("code_review_agent.api.health.create_llm_provider")
def test_health_check_returns_healthy(mock_create_provider: AsyncMock, client: TestClient) -> None:
    """Basic liveness probe should always return healthy."""
    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "timestamp" in data
    mock_create_provider.assert_not_called()


@patch("code_review_agent.api.health.create_llm_provider")
def test_readiness_check_all_healthy(mock_create_provider: AsyncMock, client: TestClient) -> None:
    """Readiness probe should return ready when all dependencies are healthy."""
    mock_provider = AsyncMock()
    mock_provider.health_check = AsyncMock(
        return_value={
            "status": "healthy",
            "provider": "openai",
            "model": "gpt-4o",
            "latency_ms": 123.45,
        }
    )
    mock_create_provider.return_value = mock_provider

    response = client.get("/health/ready")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert data["checks"]["database"]["status"] == "healthy"
    assert data["checks"]["redis"]["status"] == "healthy"
    assert data["checks"]["vector_store"]["status"] == "healthy"
    assert data["checks"]["llm"]["status"] == "healthy"
    assert data["checks"]["llm"]["provider"] == "openai"
    mock_create_provider.assert_called_once()


@patch("code_review_agent.api.health.create_llm_provider")
def test_readiness_check_returns_503_when_llm_unhealthy(
    mock_create_provider: AsyncMock,
    client: TestClient,
) -> None:
    """Readiness probe should return 503 when LLM connectivity fails."""
    mock_create_provider.side_effect = ValueError("OPENAI_API_KEY is missing")

    response = client.get("/health/ready")

    assert response.status_code == 503
    data = response.json()
    assert data["status"] == "not_ready"
    assert data["checks"]["llm"]["status"] == "unhealthy"
    assert "OPENAI_API_KEY" in data["checks"]["llm"]["error"]

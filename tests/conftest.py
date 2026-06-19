"""Shared test fixtures and configuration."""

import pytest
from unittest.mock import AsyncMock, MagicMock


@pytest.fixture
def mock_github_client():
    """Create a mock GitHub client."""
    client = AsyncMock()
    client.get_pull_request = AsyncMock(return_value={
        "title": "Test PR",
        "body": "Test description",
        "number": 1,
        "user": {"login": "testuser"},
        "base": {"ref": "main"},
        "head": {"ref": "feature-branch", "sha": "abc123"},
        "labels": [],
    })
    client.get_pull_request_files = AsyncMock(return_value=[])
    client.create_review = AsyncMock(return_value={"id": 1})
    return client


@pytest.fixture
def mock_llm():
    """Create a mock LLM provider."""
    from code_review_agent.llm.base import LLMResponse

    llm = AsyncMock()
    llm.generate = AsyncMock(return_value=LLMResponse(
        content="Test response",
        model="test-model",
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150,
    ))
    llm.generate_structured = AsyncMock(return_value=LLMResponse(
        content='{"summary": "Looks good", "decision": "approve", "confidence": 0.9, "file_reviews": []}',
        model="test-model",
        prompt_tokens=200,
        completion_tokens=100,
        total_tokens=300,
    ))
    llm.provider_name = "test"
    llm.model_name = "test-model"
    return llm


@pytest.fixture
def sample_diff():
    """Create a sample unified diff for testing."""
    return """@@ -1,5 +1,7 @@
 import os
+import sys

 def hello():
-    print("hello")
+    name = os.environ.get("NAME", "world")
+    print(f"hello {name}")
     return True
"""


@pytest.fixture
def sample_pr_files():
    """Create sample GitHub PR file data."""
    return [
        {
            "filename": "src/main.py",
            "status": "modified",
            "additions": 3,
            "deletions": 1,
            "patch": "@@ -1,5 +1,7 @@\n import os\n+import sys\n \n def hello():\n-    print(\"hello\")\n+    name = os.environ.get(\"NAME\", \"world\")\n+    print(f\"hello {name}\")\n     return True\n",
        }
    ]

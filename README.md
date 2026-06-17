# Autonomous Code Review Agent

An AI-powered code review agent that automatically reviews pull requests, understands repository context, enforces coding standards, runs security analysis, and provides actionable feedback — all without human intervention.

## Architecture

```
PR Opened (GitHub Webhook)
        ↓
   Fetch Diff
        ↓
 Understand Repository (RAG + Tree-sitter AST)
        ↓
 Retrieve Coding Standards (Vector Store)
        ↓
 Run Static Analysis (Docker Sandbox)
        ↓
 Run Tests (Docker Sandbox)
        ↓
 Security Review (Bandit, Semgrep patterns)
        ↓
 AI Review (LLM with full context)
        ↓
 Suggest Fixes + Generate Patches
        ↓
 Post Comments / Approve / Request Changes
```

## Features

- **GitHub Integration**: Webhook-driven PR review with inline comments
- **Repository Understanding**: RAG-based codebase comprehension using embeddings
- **Coding Standards Enforcement**: Configurable rules loaded into vector store
- **Static Analysis**: Runs linters and analyzers in isolated Docker containers
- **Security Scanning**: Detects vulnerabilities, secrets, and unsafe patterns
- **Bug Prediction**: ML-based identification of likely-buggy code patterns
- **Auto-generated Patches**: Suggests concrete fixes, not just problems
- **Review Memory**: Learns from previous reviews and repository patterns
- **MCP Integration**: Model Context Protocol for extensible tool access

## Tech Stack

- **Language**: Python 3.11+
- **Framework**: FastAPI
- **LLM**: OpenAI GPT-4 / Claude (configurable)
- **Embeddings**: OpenAI / Sentence Transformers
- **Vector Store**: ChromaDB
- **AST Parsing**: Tree-sitter
- **Container Sandbox**: Docker SDK
- **Database**: PostgreSQL (review history) + Redis (caching)
- **Queue**: Celery + Redis
- **GitHub**: PyGithub + webhooks

## Quick Start

```bash
# Clone and setup
git clone <repo-url>
cd code-review-agent
cp .env.example .env  # Fill in your keys

# Run with Docker
docker-compose up -d

# Or run locally
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e ".[dev]"
python -m code_review_agent.main
```

## Configuration

See `.env.example` for all configuration options. Key settings:

- `GITHUB_APP_ID` / `GITHUB_PRIVATE_KEY`: GitHub App credentials
- `OPENAI_API_KEY`: LLM provider key
- `DATABASE_URL`: PostgreSQL connection string
- `REDIS_URL`: Redis connection string

## Project Structure

```
code-review-agent/
├── src/
│   └── code_review_agent/
│       ├── main.py              # FastAPI app entry point
│       ├── config.py            # Settings and configuration
│       ├── api/                 # Webhook handlers and REST API
│       ├── core/                # Core review engine
│       ├── github/              # GitHub API integration
│       ├── analysis/            # Static analysis & security
│       ├── llm/                 # LLM providers and prompting
│       ├── rag/                 # RAG pipeline for repo understanding
│       ├── sandbox/             # Docker sandbox execution
│       ├── memory/              # Review history and learning
│       ├── models/              # Database models
│       └── mcp/                 # Model Context Protocol server
├── tests/
├── docker/
├── docs/
└── docker-compose.yml
```

## License

MIT

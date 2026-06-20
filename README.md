<div align="center">

# 🤖 Autonomous Code Review Agent

**A production-grade AI agent that autonomously reviews GitHub pull requests — understanding repository context, enforcing coding standards, running security analysis, and posting actionable feedback with auto-generated fix patches.**

Open a PR. The agent reviews it. No human in the loop.

<p>
<img alt="Python" src="https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white">
<img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white">
<img alt="PostgreSQL" src="https://img.shields.io/badge/PostgreSQL-16-4169E1?logo=postgresql&logoColor=white">
<img alt="Redis" src="https://img.shields.io/badge/Redis-7-DC382D?logo=redis&logoColor=white">
<img alt="Docker" src="https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white">
<img alt="OpenAI" src="https://img.shields.io/badge/OpenAI-GPT--4o-412991?logo=openai&logoColor=white">
<img alt="License" src="https://img.shields.io/badge/license-MIT-green">
</p>

</div>

---

## ✨ Highlights

| | Capability | Description |
|---|---|---|
| 🔗 | **GitHub App Integration** | Webhook-driven — installs on any repo, posts reviews as inline comments with approve/reject |
| 🌳 | **Multi-language AST** | Tree-sitter parsing for Python, JavaScript, TypeScript — understands code structure, not just text |
| 🧠 | **RAG-powered Context** | ChromaDB vector store retrieves related code + coding standards for context-aware reviews |
| 🐳 | **Docker Sandbox** | Runs linters, type checkers, and tests in isolated containers with CPU/memory limits |
| 🔒 | **Security Scanning** | Bandit + Semgrep + regex secret detection — catches vulnerabilities before merge |
| 🐛 | **Bug Prediction** | Heuristic-based risk scoring (entropy, error handling removal, sensitive file changes) |
| 🩹 | **Auto-generated Patches** | Produces `git apply`-ready unified diffs and GitHub suggestion blocks |
| 📚 | **Review Memory** | Learns from past reviews — tracks author patterns, detects false positives, avoids repetition |
| 🔌 | **MCP Server** | Model Context Protocol interface for IDE integration and extensible tool access |
| ⚡ | **Async Processing** | Celery + Redis task queue — reviews run in background, webhook returns instantly |

---

## 🚀 Quickstart

### Option 1 — Full stack with Docker Compose (recommended)

```bash
git clone https://github.com/HoneyTyagii/Autonomous-Code-Review-Agent.git
cd Autonomous-Code-Review-Agent
cp .env.example .env   # Fill in your API keys and GitHub App credentials

docker compose up -d --build
docker compose run --rm migrations
```

### Option 2 — Local development

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/Mac

pip install -e ".[dev]"
python -m code_review_agent.main
```

### Verify it's running

| Endpoint | Purpose |
|---|---|
| http://localhost:8000/health | Liveness check → `{"status":"healthy"}` |
| http://localhost:8000/health/ready | Readiness check (DB, Redis, ChromaDB) |
| http://localhost:8000/docs | Interactive Swagger UI |

---

## 🗺️ How It Works

```
┌──────────────────┐
│  PR Opened       │  GitHub webhook fires
└────────┬─────────┘
         ▼
┌──────────────────┐
│  Fetch Diff      │  Parse unified diff → structured hunks with line mapping
└────────┬─────────┘
         ▼
┌──────────────────┐
│  Understand Repo │  Tree-sitter AST + RAG retrieval of related code
└────────┬─────────┘
         ▼
┌──────────────────┐
│  Load Standards  │  Vector search for applicable coding rules
└────────┬─────────┘
         ▼
┌──────────────────┐
│  Static Analysis │  ruff, mypy, eslint in Docker sandbox
└────────┬─────────┘
         ▼
┌──────────────────┐
│  Security Scan   │  Bandit + Semgrep + secret detection
└────────┬─────────┘
         ▼
┌──────────────────┐
│  Bug Prediction  │  Heuristic risk scoring on the diff
└────────┬─────────┘
         ▼
┌──────────────────┐
│  LLM Review      │  GPT-4o / Claude with structured JSON output
└────────┬─────────┘
         ▼
┌──────────────────┐
│  Generate Fixes  │  Auto-patches as unified diffs
└────────┬─────────┘
         ▼
┌──────────────────┐
│  Post to GitHub  │  Inline comments + approve / request changes
└──────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **API** | FastAPI, uvicorn, Pydantic |
| **LLM** | OpenAI GPT-4o, Anthropic Claude (pluggable via factory) |
| **Embeddings** | OpenAI `text-embedding-3-small` / Sentence Transformers |
| **Vector Store** | ChromaDB (cosine similarity, per-repo collections) |
| **AST** | Tree-sitter (Python, JavaScript, TypeScript grammars) |
| **Database** | PostgreSQL 16 + SQLAlchemy 2.0 (async) + Alembic migrations |
| **Queue** | Celery + Redis (separate broker/result queues) |
| **Sandbox** | Docker SDK (resource-limited, read-only, network-isolated) |
| **Security** | Bandit, Semgrep, regex-based secret detection |
| **GitHub** | App JWT auth, installation tokens, webhook HMAC verification |
| **Observability** | structlog (JSON in prod, colored console in dev) |
| **MCP** | Model Context Protocol server (stdio transport) |

---

## 📁 Project Structure

```
src/code_review_agent/
├── main.py                    FastAPI app + lifespan management
├── config.py                  Pydantic settings (all env vars)
├── logging.py                 Structured logging (structlog)
│
├── api/                       HTTP layer
│   ├── webhooks.py              GitHub webhook receiver
│   ├── webhook_security.py      HMAC-SHA256 signature verification
│   └── health.py                Liveness + readiness probes
│
├── github/                    GitHub integration
│   ├── auth.py                  JWT + installation token auth
│   ├── client.py                Async API client with retry + rate limits
│   ├── diff_parser.py           Unified diff → structured hunks
│   └── pr_fetcher.py            Assembles complete PR context
│
├── analysis/                  Code analysis
│   ├── ast_parser.py            Tree-sitter multi-language parsing
│   ├── security_scanner.py      Bandit + Semgrep + secrets
│   └── bug_predictor.py         Heuristic risk scoring
│
├── core/                      Review engine
│   ├── engine.py                Pipeline orchestrator (whole-PR / per-file)
│   ├── prompts.py               LLM prompt construction
│   ├── schemas.py               Structured output JSON schemas
│   └── patch_generator.py       Auto-fix generation (unified diff)
│
├── llm/                       LLM abstraction
│   ├── base.py                  Interface + Message/Response types
│   ├── openai_provider.py       OpenAI with retry + JSON mode
│   ├── anthropic_provider.py    Claude with system message handling
│   └── factory.py               Provider factory from config
│
├── rag/                       Retrieval-Augmented Generation
│   ├── embeddings.py            Multi-provider embedding service
│   ├── vector_store.py          ChromaDB client wrapper
│   ├── indexer.py               AST-aware code chunking + batch indexing
│   ├── retriever.py             Semantic context retrieval
│   └── standards_loader.py      Coding standards ingestion (Markdown → rules)
│
├── sandbox/                   Isolated execution
│   ├── docker_sandbox.py        Docker container management
│   └── analysis_runner.py       Linter/test orchestration
│
├── memory/                    Learning system
│   ├── review_memory.py         Persistence + semantic search over history
│   └── learning.py              Author profiling + false positive detection
│
├── tasks/                     Background processing
│   ├── celery_app.py            Celery configuration + routing
│   └── review_tasks.py          Async review pipeline + GitHub posting
│
├── models/                    Database ORM
│   ├── repository.py            Repository tracking model
│   └── review.py                Review + ReviewComment models
│
├── db/                        Database layer
│   └── session.py               Async engine + session factory
│
└── mcp/                       Model Context Protocol
    ├── server.py                JSON-RPC server
    ├── tools.py                 6 exposed tools
    └── transport.py             Stdio transport for IDEs
```

---

## 🧪 Development

```bash
make install          # Install in dev mode
make test             # Run pytest with coverage
make check            # Lint + typecheck
make format           # Auto-format with ruff
make dev              # Full dev stack (hot-reload + Flower monitoring)
```

| Command | Description |
|---------|-------------|
| `make dev` | Start full dev stack with hot-reload |
| `make test` | Run pytest with coverage |
| `make lint` | Run ruff linter |
| `make typecheck` | Run mypy |
| `make migrate` | Run database migrations |
| `make psql` | Connect to PostgreSQL |
| `make logs` | Tail all service logs |
| `make clean` | Remove build artifacts |

---

## 🔌 MCP Integration

The agent exposes a [Model Context Protocol](https://modelcontextprotocol.io) server for IDE integration:

```bash
python -m code_review_agent.mcp.transport
```

**Exposed tools:**

| Tool | Description |
|------|-------------|
| `review_diff` | Review a code diff with full LLM pipeline |
| `check_security` | Run security scans on code |
| `query_standards` | Find relevant coding standards |
| `search_past_reviews` | Semantic search over review history |
| `generate_patch` | Generate a fix as unified diff |
| `predict_bugs` | Heuristic bug risk analysis |

---

## ⚙️ Configuration

All settings via environment variables (see [`.env.example`](.env.example)):

| Variable | Description | Default |
|----------|-------------|---------|
| `GITHUB_APP_ID` | GitHub App numeric ID | — |
| `GITHUB_WEBHOOK_SECRET` | Webhook HMAC secret | — |
| `LLM_PROVIDER` | `openai` or `anthropic` | `openai` |
| `OPENAI_MODEL` | Model for reviews | `gpt-4o` |
| `ENABLE_SECURITY_SCAN` | Run Bandit + Semgrep | `true` |
| `SANDBOX_TIMEOUT` | Docker sandbox timeout (s) | `300` |
| `SANDBOX_MEMORY_LIMIT` | Container memory cap | `512m` |
| `DATABASE_URL` | PostgreSQL connection | `postgresql+asyncpg://...` |
| `REDIS_URL` | Redis connection | `redis://localhost:6379/0` |

---

## 🏗️ Demonstrates

| Domain | What's showcased |
|--------|-----------------|
| **AI/ML Engineering** | LLM orchestration, RAG pipelines, embeddings, structured output, prompt engineering |
| **Backend Development** | Async Python, FastAPI, SQLAlchemy, Celery, Redis, PostgreSQL |
| **DevOps** | Docker multi-stage builds, Compose orchestration, health checks, Makefile automation |
| **Systems Design** | Event-driven architecture, webhook processing, background tasks, graceful degradation |
| **Security Engineering** | HMAC verification, sandboxed execution, secret scanning, non-root containers |
| **Software Engineering** | Clean architecture, SOLID principles, type safety, structured logging, migrations |

---

## 📄 License

MIT

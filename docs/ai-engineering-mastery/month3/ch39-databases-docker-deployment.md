# Chapter 39 — Databases, Docker & Deployment

## Learning Objectives

- Design database schemas for AI applications
- Containerize AI apps with Docker
- Deploy to cloud platforms
- Set up CI/CD for AI projects

---

## Database Schema for AI Apps

```
┌──────────────────────────────────────────────────────────────┐
│  AI APP DATABASE SCHEMA (SQLite/PostgreSQL)                │
│                                                              │
│  users                                                       │
│  ┌─────────┬──────────┬───────────┐                         │
│  │ id (PK) │ email    │ created   │                         │
│  └─────────┴──────────┴───────────┘                         │
│                                                              │
│  documents                                                  │
│  ┌─────────┬──────────┬──────────┬──────────┐              │
│  │ id (PK) │ title    │ content  │ user_id  │              │
│  └─────────┴──────────┴──────────┴──────────┘              │
│                                                              │
│  chunks (for RAG)                                           │
│  ┌─────────┬──────────┬─────────┬────────┐                 │
│  │ id (PK) │ doc_id   │ text    │ vector  │                 │
│  └─────────┴──────────┴─────────┴────────┘                 │
│                                                              │
│  conversations                                              │
│  ┌─────────┬──────────┬────────┐                           │
│  │ id (PK) │ user_id  │ title  │                           │
│  └─────────┴──────────┴────────┘                           │
│                                                              │
│  messages                                                   │
│  ┌─────────┬─────────────┬────────┬──────────┐             │
│  │ id (PK) │ conv_id    │ role   │ content  │             │
│  └─────────┴─────────────┴────────┴──────────┘             │
│                                                              │
│  Plus: vector DB (ChromaDB) for semantic search             │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

## Docker for AI Apps

```dockerfile
# Dockerfile for an AI application
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 8000

# Run
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
version: '3.8'
services:
  api:
    build: .
    ports: ["8000:8000"]
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    depends_on: [db, chromadb]

  db:
    image: postgres:16-alpine
    environment:
      POSTGRES_DB: edify
      POSTGRES_PASSWORD: secret
    volumes: ["db_data:/var/lib/postgresql/data"]

  chromadb:
    image: chromadb/chroma:latest
    ports: ["8001:8000"]

volumes:
  db_data:
```

## Deployment Options

```
┌──────────────────────────────────────────────────────────────┐
│  DEPLOYMENT OPTIONS                                         │
│                                                              │
│  1. LOCAL (development)                                     │
│  → docker-compose up                                         │
│  → uvicorn main:app --reload                                 │
│                                                              │
│  2. CLOUD VM (simple production)                           │
│  → Rent VPS (DigitalOcean, AWS EC2)                        │
│  → Install Docker → docker-compose up                     │
│  → Set up nginx as reverse proxy                           │
│                                                              │
│  3. PLATFORM AS A SERVICE (easiest)                       │
│  → Railway, Render, Fly.io                                  │
│  → Push code → automatic deploy                            │
│  → Free tiers available                                    │
│                                                              │
│  4. SERVERLESS (scaling)                                    │
│  → AWS Lambda, Cloud Run                                    │
│  → Pay per request                                          │
│  → Good for spiky workloads                                 │
│                                                              │
│  5. KUBERNETES (enterprise scale)                          │
│  → For large teams and high traffic                        │
│  → Complex but highly scalable                             │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

## Git & GitHub Workflow

```bash
# Initialize
git init
git add .
git commit -m "Initial commit"

# .gitignore for AI projects
echo """
__pycache__/
*.pyc
.env
venv/
.venv/
*.db
chromadb/
node_modules/
dist/
""" > .gitignore

# Branch workflow
git checkout -b feature/rag-pipeline
# ... make changes ...
git add . && git commit -m "Add RAG pipeline"
git push origin feature/rag-pipeline
# Create PR → review → merge to main
```

## Chapter Summary

```
• Database: users, documents, chunks, conversations, messages
• Vector DB: ChromaDB for semantic search
• Docker: containerize app + database + vector DB
• Deploy: Railway/Render (easy), Cloud VM (control), Serverless (scale)
• Git: feature branches, PRs, .gitignore for secrets/data
```

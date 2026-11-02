# Telecom Support RAG Agent

A support assistant for a fictional mobile operator: answers questions about plans, roaming and billing
from a document catalog, calls tools to recommend and compare bundles, and is gated by an eval suite.

> Work in progress, built as a hands-on AI-engineering project. Results table and architecture diagram land here as phases complete.

- **Plan & progress:** [ROADMAP.md](ROADMAP.md)
- **Lessons:** [learn/](learn/)
- **Interview prep:** [interview-bank.md](interview-bank.md)
- **Original spec:** [AI_Engineer_Side_Project.md](AI_Engineer_Side_Project.md)

Data is synthetic or from public operator pages only.

```bash
uv sync && cp .env.example .env   # then fill .env
uv run pytest                     # unit tests (no API calls)
uv run pytest -m live             # live API tests (costs a fraction of a cent)
```

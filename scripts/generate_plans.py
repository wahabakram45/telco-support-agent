"""Build the plan catalog and render one markdown page per plan version.

Run: uv run python scripts/generate_plans.py
Writes data/catalog/plans.json and data/docs/plans/*.md. See learn/01-data-for-rag.md Task 2.
"""

from datetime import date  # noqa: F401 (you'll use this in build_plans)

from telco_support_agent.catalog import PLANS_PATH, Plan, render_plan_markdown, save_plans
from telco_support_agent.documents import DOCS_ROOT

PLAN_DOCS_DIR = DOCS_ROOT / "plans"


def build_plans() -> list[Plan]:
    """Every plan version: ≥30 current plans across both markets and all types, plus traps T1–T5."""
    # TODO(you) — lesson 01, Task 2. tests/test_catalog_data.py checks the requirements.
    raise NotImplementedError


def main() -> None:
    plans = build_plans()
    save_plans(plans)

    by_id = {p.plan_id: p for p in plans}
    PLAN_DOCS_DIR.mkdir(parents=True, exist_ok=True)
    for old in PLAN_DOCS_DIR.glob("*.md"):
        old.unlink()
    for p in plans:
        (PLAN_DOCS_DIR / f"{p.plan_id}.md").write_text(render_plan_markdown(p, by_id))

    current = sum(p.is_current for p in plans)
    print(f"{len(plans)} plan versions ({current} current) -> {PLANS_PATH} and {PLAN_DOCS_DIR}/")


if __name__ == "__main__":
    main()

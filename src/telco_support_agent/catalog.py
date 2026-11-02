"""Plan catalog: the structured source of truth for prices and allowances.

See learn/01-data-for-rag.md §1 and Task 1.
"""

import json
from datetime import date  # noqa: F401 (you'll use this in Plan)
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel

Market = Literal["KSA", "UAE"]
Currency = Literal["SAR", "AED"]
PlanType = Literal["prepaid", "postpaid", "data_only", "roaming_addon"]
CURRENCY_BY_MARKET: dict[str, str] = {"KSA": "SAR", "UAE": "AED"}

PLANS_PATH = Path("data/catalog/plans.json")


class Plan(BaseModel):
    """One version of a plan.

    A price change creates a new version: the old one gets `effective_to` and `replaced_by`.
    """

    # TODO(you) — lesson 01, Task 1: define the fields from the lesson's table (exact names),
    # with types, defaults and constraints, then add the validators. Forbid unknown fields.

    @property
    def is_current(self) -> bool:
        return self.effective_to is None and self.status == "active"


def load_plans(path: Path = PLANS_PATH) -> list[Plan]:
    return [Plan.model_validate(p) for p in json.loads(path.read_text())]


def save_plans(plans: list[Plan], path: Path = PLANS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [p.model_dump(mode="json") for p in plans]
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n")


_TYPE_LABEL = {
    "prepaid": "Prepaid plan",
    "postpaid": "Postpaid plan",
    "data_only": "Data-only plan",
    "roaming_addon": "Roaming add-on",
}


def _num(x: float) -> str:
    return f"{x:g}"


def render_plan_markdown(plan: Plan, plans_by_id: dict[str, Plan]) -> str:
    """Render one plan version as a markdown document with YAML frontmatter."""
    meta = {
        "doc_id": f"plan-{plan.plan_id}",
        "title": f"{plan.name} ({plan.market})",
        "doc_type": "plan",
        "market": plan.market,
        "plan_ids": [plan.plan_id],
        "effective_from": plan.effective_from,
        "effective_to": plan.effective_to,
        "superseded_by": f"plan-{plan.replaced_by}" if plan.replaced_by else None,
    }

    if plan.replaced_by:
        new = plans_by_id[plan.replaced_by]
        status = (
            f"**No longer available.** This version was offered from {plan.effective_from} "
            f"to {plan.effective_to} and was replaced by the current {new.name} plan "
            f"from {new.effective_from}."
        )
    elif plan.status == "discontinued":
        status = "**Discontinued.** This plan is no longer offered to new or existing customers."
    elif plan.effective_to:
        status = f"**No longer available** after {plan.effective_to}."
    else:
        status = f"**Available now** (since {plan.effective_from})."

    if plan.unlimited_data:
        data = f"Unlimited (full speed up to {_num(plan.fair_usage_gb)} GB, then reduced speed)"
    else:
        data = f"{_num(plan.data_gb)} GB"

    rows = [
        ("Local data", data),
        ("Local minutes", str(plan.local_minutes)),
        ("SMS", str(plan.sms)),
    ]
    if plan.roaming_countries:
        roaming = f"{_num(plan.roaming_data_gb or 0)} GB in {', '.join(plan.roaming_countries)}"
        rows.append(("Roaming data", roaming))

    lines = [
        f"# {plan.name} ({plan.market})",
        "",
        f"{_TYPE_LABEL[plan.plan_type]} from TelcoX {plan.market}.",
        "",
        f"**Price:** {plan.price:.2f} {plan.currency} every {plan.validity_days} days",
        "",
        status,
        "",
        "| Allowance | Amount |",
        "|---|---|",
        *[f"| {k} | {v} |" for k, v in rows],
    ]
    if plan.features:
        lines += ["", "**Included features:**", *[f"- {f}" for f in plan.features]]

    front = yaml.safe_dump(meta, sort_keys=False, allow_unicode=True)
    return f"---\n{front}---\n\n" + "\n".join(lines) + "\n"

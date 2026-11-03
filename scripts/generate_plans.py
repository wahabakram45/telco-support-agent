"""Build the plan catalog and render one markdown page per plan version.

Run: uv run python scripts/generate_plans.py
Writes data/catalog/plans.json and data/docs/plans/*.md. See learn/01-data-for-rag.md Task 2.
"""

from datetime import date

from telco_support_agent.catalog import (
    CURRENCY_BY_MARKET,
    PLANS_PATH,
    Plan,
    render_plan_markdown,
    save_plans,
)
from telco_support_agent.documents import DOCS_ROOT

PLAN_DOCS_DIR = DOCS_ROOT / "plans"

LAUNCH = date(2026, 1, 1)
REFRESH = date(2026, 7, 1)  # mid-year price refresh
OLD_FROM, OLD_TO = date(2025, 1, 1), date(2026, 6, 30)

HOME_COUNTRY = {"KSA": "Saudi Arabia", "UAE": "UAE"}
GCC = ["Saudi Arabia", "UAE", "Bahrain", "Kuwait", "Oman", "Qatar"]
EUROPE = ["UK", "France", "Germany", "Italy", "Spain", "Turkey"]
ASIA = ["India", "Pakistan", "Bangladesh", "Philippines", "Malaysia", "Indonesia"]
WORLD = ["USA", "Canada", "Australia", "Japan", *EUROPE, *ASIA]

# Monthly prices: SAR in KSA, AED in UAE.
PRICES = {
    "KSA": {
        "flex-start": 35,
        "flex-smart": 55,
        "flex-smart-plus": 69,
        "flex-pro": 99,
        "flex-unlimited": 199,
        "max-basic": 115,
        "max-advanced": 150,
        "max-premium": 250,
        "max-elite": 399,
        "datago-20": 45,
        "datago-60": 95,
        "datago-150": 175,
        "travel-gcc": 49,
        "travel-europe": 119,
        "travel-asia": 99,
        "travel-world": 199,
    },
    "UAE": {
        "flex-start": 39,
        "flex-smart": 59,
        "flex-smart-plus": 75,
        "flex-pro": 109,
        "flex-unlimited": 219,
        "max-basic": 125,
        "max-advanced": 165,
        "max-premium": 275,
        "max-elite": 425,
        "datago-20": 49,
        "datago-60": 99,
        "datago-150": 189,
        "travel-gcc": 45,
        "travel-europe": 109,
        "travel-asia": 95,
        "travel-world": 189,
    },
}

# Plans whose current version started at the mid-year refresh.
REFRESHED = {("KSA", "flex-smart"), ("KSA", "max-advanced"), ("UAE", "datago-60")}


def _plan(market: str, slug: str, name: str, plan_type: str, validity_days: int, **kw) -> Plan:
    """A plan in `market`, priced from PRICES unless `price` is given."""
    if "price" not in kw:
        kw["price"] = PRICES[market][slug]
    kw.setdefault("effective_from", REFRESH if (market, slug) in REFRESHED else LAUNCH)
    kw.setdefault("local_minutes", 0)
    kw.setdefault("sms", 0)
    return Plan(
        plan_id=f"{slug}-{market.lower()}",
        name=name,
        market=market,
        currency=CURRENCY_BY_MARKET[market],
        plan_type=plan_type,
        validity_days=validity_days,
        **kw,
    )


def _current_plans(m: str) -> list[Plan]:
    away_gcc = [c for c in GCC if c != HOME_COUNTRY[m]]
    unlimited = {"data_gb": None, "unlimited_data": True}
    return [
        # Prepaid, 28 days
        _plan(m, "flex-start", "Flex Start", "prepaid", 28, data_gb=5, local_minutes=100, sms=50),
        _plan(m, "flex-smart", "Flex Smart", "prepaid", 28, data_gb=12, local_minutes=300, sms=100),
        _plan(
            m,
            "flex-smart-plus",
            "Flex Smart Plus",
            "prepaid",
            28,
            data_gb=20,
            local_minutes=500,
            sms=100,
            features=["5G"],
        ),
        _plan(
            m,
            "flex-pro",
            "Flex Pro",
            "prepaid",
            28,
            data_gb=40,
            local_minutes=1000,
            sms=200,
            features=["5G"],
        ),
        _plan(
            m,
            "flex-unlimited",
            "Flex Unlimited",
            "prepaid",
            28,
            **unlimited,
            fair_usage_gb=150,
            local_minutes=2000,
            sms=500,
            features=["5G"],
        ),
        # Postpaid, 30 days
        _plan(
            m,
            "max-basic",
            "Max Basic",
            "postpaid",
            30,
            data_gb=30,
            local_minutes=500,
            sms=100,
            features=["5G"],
        ),
        _plan(
            m,
            "max-advanced",
            "Max Advanced",
            "postpaid",
            30,
            data_gb=60,
            local_minutes=1500,
            sms=300,
            features=["5G", "Data rollover"],
        ),
        _plan(
            m,
            "max-premium",
            "Max Premium",
            "postpaid",
            30,
            **unlimited,
            fair_usage_gb=200,
            local_minutes=3000,
            sms=1000,
            features=["5G", "Free calls to TelcoX numbers"],
        ),
        _plan(
            m,
            "max-elite",
            "Max Elite",
            "postpaid",
            30,
            **unlimited,
            fair_usage_gb=500,
            local_minutes=5000,
            sms=1000,
            roaming_countries=away_gcc,
            roaming_data_gb=10,
            features=["5G", "Free calls to TelcoX numbers", "Airport lounge access"],
        ),
        # Data-only, 30 days
        _plan(m, "datago-20", "DataGo 20", "data_only", 30, data_gb=20),
        _plan(m, "datago-60", "DataGo 60", "data_only", 30, data_gb=60),
        _plan(m, "datago-150", "DataGo 150", "data_only", 30, data_gb=150, features=["5G"]),
        # Roaming add-ons
        _plan(
            m,
            "travel-gcc",
            "Travel GCC",
            "roaming_addon",
            7,
            data_gb=0,
            roaming_countries=away_gcc,
            roaming_data_gb=3,
        ),
        _plan(
            m,
            "travel-europe",
            "Travel Europe",
            "roaming_addon",
            14,
            data_gb=0,
            roaming_countries=EUROPE,
            roaming_data_gb=8,
        ),
        _plan(
            m,
            "travel-asia",
            "Travel Asia",
            "roaming_addon",
            14,
            data_gb=0,
            roaming_countries=ASIA,
            roaming_data_gb=6,
        ),
        _plan(
            m,
            "travel-world",
            "Travel World",
            "roaming_addon",
            30,
            data_gb=0,
            roaming_countries=WORLD,
            roaming_data_gb=10,
        ),
    ]


def _past_plans() -> list[Plan]:
    """Versions that are no longer sold."""
    old = {"effective_from": OLD_FROM, "effective_to": OLD_TO}
    return [
        _plan(
            "KSA",
            "flex-smart-2025",
            "Flex Smart",
            "prepaid",
            28,
            price=59,
            data_gb=10,
            local_minutes=300,
            sms=100,
            replaced_by="flex-smart-ksa",
            **old,
        ),
        _plan(
            "KSA",
            "max-advanced-2025",
            "Max Advanced",
            "postpaid",
            30,
            price=175,
            data_gb=50,
            local_minutes=1500,
            sms=300,
            features=["5G", "Data rollover"],
            replaced_by="max-advanced-ksa",
            **old,
        ),
        _plan(
            "UAE",
            "datago-60-2025",
            "DataGo 60",
            "data_only",
            30,
            price=110,
            data_gb=60,
            replaced_by="datago-60-uae",
            **old,
        ),
        _plan(
            "KSA",
            "flex-classic",
            "Flex Classic",
            "prepaid",
            28,
            price=25,
            data_gb=1,
            local_minutes=150,
            sms=50,
            effective_from=date(2024, 1, 1),
            effective_to=date(2026, 3, 31),
            status="discontinued",
        ),
        _plan(
            "UAE",
            "social-pack",
            "Social Pack",
            "data_only",
            30,
            price=30,
            data_gb=5,
            features=["Social apps don't use your data"],
            effective_from=date(2024, 6, 1),
            effective_to=date(2025, 12, 31),
            status="discontinued",
        ),
    ]


def build_plans() -> list[Plan]:
    """Every plan version: ≥30 current plans across both markets and all types, plus traps T1–T5."""
    return [*_current_plans("KSA"), *_current_plans("UAE"), *_past_plans()]


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

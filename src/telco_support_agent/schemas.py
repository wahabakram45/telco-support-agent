"""Pydantic schemas shared across the app: structured outputs now, tool arguments later."""

from pydantic import BaseModel, Field


class UsageProfile(BaseModel):
    """A subscriber's monthly usage, extracted from free text.

    Fields are optional on purpose: if the user didn't say it, the model must return null
    rather than invent a number (our first guardrail).
    """

    # TODO(you) — lesson 00a: make tests/test_schemas.py pass.
    #   - forbid unknown fields (model_config = ConfigDict(...))
    #   - no negative values anywhere (Field(..., ge=0))
    #   - a sane upper bound on data_gb (le=...): what's the most data a plan could plausibly have?

    data_gb: float | None = Field(None, description="Monthly mobile data in GB; null if not stated")
    minutes: int | None = Field(None, description="Monthly voice minutes; null if not stated")
    sms: int | None = Field(None, description="Monthly SMS count; null if not stated")
    needs_roaming: bool | None = Field(None, description="True if the user mentions travel/roaming")
    budget_monthly: float | None = Field(None, description="Max monthly budget in local currency")

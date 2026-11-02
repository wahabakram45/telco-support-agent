import pytest

from telco_support_agent.checks import find_unsupported_numbers


@pytest.mark.parametrize(
    "text,source,expected",
    [
        ("It costs 49.99 SAR.", "price 49.99", set()),
        ("It costs 59 SAR.", "price 49 SAR", {"59"}),
        ("You get 1,000 minutes.", "1000 minutes", set()),
        ("You get 1000 minutes.", "1,000 minutes", set()),
        ("50 GB of data", "50.0 GB", set()),
        ("VAT is 15%.", "VAT of 15 percent", set()),
        ("Step 1. Open the app. Step 2. Tap roaming.", "", set()),
        ("Valid for 24 hours, costs 25 SAR", "25 SAR for 24 hours", set()),
        ("Late fee 30 SAR, due in 21 days", "late fee 25 SAR, 21 days", {"30"}),
        ("Speed drops to 1.5 Mbps", "speed drops to 1 Mbps", {"1.5"}),
        ("No numbers here.", "", set()),
    ],
)
def test_find_unsupported_numbers(text, source, expected):
    assert find_unsupported_numbers(text, source) == expected

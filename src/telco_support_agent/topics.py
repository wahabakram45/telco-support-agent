"""Document topics and their facts: the ground truth for generated policies and FAQs.

Every number a generated document states must come from its topic's facts. Evals (Phase 3) check
answers against these facts. TelcoX is fictional.
"""

from datetime import date

from pydantic import BaseModel


class Topic(BaseModel):
    doc_id: str
    title: str
    doc_type: str  # "policy" or "faq"
    market: str  # "KSA", "UAE" or "ALL"
    effective_from: date
    effective_to: date | None = None
    superseded_by: str | None = None
    target_words: int = 350
    outline: list[str] = []  # fixed section headings, for long documents
    facts: list[str]


TOPICS = [
    # --- Roaming: T6 (superseded policy) and T2/T7-style market differences ---
    Topic(
        doc_id="roaming-policy-ksa",
        title="Roaming Policy (KSA)",
        doc_type="policy",
        market="KSA",
        effective_from=date(2026, 7, 1),
        facts=[
            "Pay-as-you-go roaming data costs 30 SAR per 100 MB in GCC countries.",
            "Pay-as-you-go roaming data costs 60 SAR per 100 MB in all other countries.",
            "The GCC Day Pass costs 25 SAR for 1 GB of roaming data, valid for 24 hours.",
            "Incoming calls while roaming cost 3 SAR per minute.",
            "Data roaming is disabled by default on new lines.",
            "Customers activate roaming in the TelcoX app or by dialling *123#.",
            "Data roaming is suspended when roaming charges reach 500 SAR in a billing cycle, "
            "unless the customer opts out of this limit in the app.",
        ],
    ),
    Topic(
        doc_id="roaming-policy-ksa-2025",
        title="Roaming Policy (KSA) — 2025",
        doc_type="policy",
        market="KSA",
        effective_from=date(2025, 1, 1),
        effective_to=date(2026, 6, 30),
        superseded_by="roaming-policy-ksa",
        facts=[
            "Pay-as-you-go roaming data costs 40 SAR per 100 MB in GCC countries.",
            "Pay-as-you-go roaming data costs 75 SAR per 100 MB in all other countries.",
            "The GCC Day Pass costs 35 SAR for 500 MB of roaming data, valid for 24 hours.",
            "Incoming calls while roaming cost 3 SAR per minute.",
            "Customers activate roaming in the TelcoX app or by dialling *123#.",
            "Data roaming is suspended when roaming charges reach 300 SAR in a billing cycle.",
        ],
    ),
    Topic(
        doc_id="roaming-policy-uae",
        title="Roaming Policy (UAE)",
        doc_type="policy",
        market="UAE",
        effective_from=date(2026, 3, 1),
        facts=[
            "Pay-as-you-go roaming data costs 25 AED per 100 MB in GCC countries.",
            "Pay-as-you-go roaming data costs 50 AED per 100 MB in all other countries.",
            "The GCC Day Pass costs 20 AED for 1 GB of roaming data, valid for 24 hours.",
            "Incoming calls while roaming cost 2.5 AED per minute.",
            "Data roaming is disabled by default on new lines.",
            "Customers activate roaming in the TelcoX app or by dialling *123#.",
            "Data roaming is suspended when roaming charges reach 400 AED in a billing cycle, "
            "unless the customer opts out of this limit in the app.",
        ],
    ),
    # --- Fair usage (T4) ---
    Topic(
        doc_id="fair-usage-policy",
        title="Fair Usage Policy",
        doc_type="policy",
        market="ALL",
        effective_from=date(2026, 1, 1),
        facts=[
            "Unlimited data plans give full-speed data up to the plan's fair usage threshold "
            "in each billing cycle.",
            "After the threshold is reached, speed is reduced to 1 Mbps until the plan renews.",
            "The threshold resets when the plan renews.",
            "After the threshold, video streaming may be limited to 480p.",
            "Tethering (mobile hotspot) on unlimited plans is capped at 20 GB per billing cycle "
            "and counts toward the fair usage threshold.",
            "The fair usage threshold for each plan is listed on its plan page.",
        ],
    ),
    # --- Billing: T7 (VAT included in KSA, added in UAE) ---
    Topic(
        doc_id="postpaid-billing-ksa",
        title="Postpaid Billing and Payments (KSA)",
        doc_type="policy",
        market="KSA",
        effective_from=date(2026, 1, 1),
        facts=[
            "Postpaid bills are issued on the 1st of each month.",
            "Payment is due within 21 days of the bill date.",
            "A late payment fee of 25 SAR applies after the due date.",
            "Outgoing services are suspended if a bill is unpaid 45 days after the bill date.",
            "Electronic bills are free; paper bills cost 10 SAR per month.",
            "VAT of 15% is already included in all advertised plan prices in KSA.",
            "Bills can be paid in the TelcoX app, by bank transfer, or through SADAD.",
        ],
    ),
    Topic(
        doc_id="postpaid-billing-uae",
        title="Postpaid Billing and Payments (UAE)",
        doc_type="policy",
        market="UAE",
        effective_from=date(2026, 1, 1),
        facts=[
            "Postpaid bills are issued on the 5th of each month.",
            "Payment is due within 20 days of the bill date.",
            "A late payment fee of 30 AED applies after the due date.",
            "Outgoing services are suspended if a bill is unpaid 40 days after the bill date.",
            "Electronic bills are free; paper bills cost 10 AED per month.",
            "VAT of 5% is added on top of advertised plan prices in the UAE.",
            "Bills can be paid in the TelcoX app, by card, or at TelcoX stores.",
        ],
    ),
    Topic(
        doc_id="out-of-bundle-charges-ksa",
        title="Charges Outside Your Bundle (KSA)",
        doc_type="policy",
        market="KSA",
        effective_from=date(2026, 1, 1),
        facts=[
            "Local calls after the bundle's minutes are used cost 0.5 SAR per minute.",
            "SMS after the bundle's allowance cost 0.25 SAR each.",
            "Mobile data stops when the bundle's data runs out, unless Data Overage is enabled.",
            "With Data Overage enabled, extra data costs 5 SAR per 100 MB.",
            "Data Overage can be switched on or off in the TelcoX app at any time.",
        ],
    ),
    Topic(
        doc_id="out-of-bundle-charges-uae",
        title="Charges Outside Your Bundle (UAE)",
        doc_type="policy",
        market="UAE",
        effective_from=date(2026, 1, 1),
        facts=[
            "Local calls after the bundle's minutes are used cost 0.4 AED per minute.",
            "SMS after the bundle's allowance cost 0.2 AED each.",
            "Mobile data stops when the bundle's data runs out, unless Data Overage is enabled.",
            "With Data Overage enabled, extra data costs 4 AED per 100 MB.",
            "Data Overage can be switched on or off in the TelcoX app at any time.",
        ],
    ),
    Topic(
        doc_id="international-calling-ksa",
        title="International Calling Rates (KSA)",
        doc_type="policy",
        market="KSA",
        effective_from=date(2026, 1, 1),
        facts=[
            "Calls to GCC countries cost 1.5 SAR per minute.",
            "Calls to India, Pakistan and Bangladesh cost 0.9 SAR per minute.",
            "Calls to the UK and the US cost 1.2 SAR per minute.",
            "Calls to all other countries cost 3 SAR per minute.",
            "The International Saver add-on costs 30 SAR for 100 international minutes, "
            "valid for 30 days, to India, Pakistan, Bangladesh, the Philippines and Egypt.",
        ],
    ),
    Topic(
        doc_id="international-calling-uae",
        title="International Calling Rates (UAE)",
        doc_type="policy",
        market="UAE",
        effective_from=date(2026, 1, 1),
        facts=[
            "Calls to GCC countries cost 1.2 AED per minute.",
            "Calls to India, Pakistan and Bangladesh cost 0.8 AED per minute.",
            "Calls to the UK and the US cost 1.0 AED per minute.",
            "Calls to all other countries cost 2.5 AED per minute.",
            "The International Saver add-on costs 25 AED for 100 international minutes, "
            "valid for 30 days, to India, Pakistan, Bangladesh, the Philippines and Egypt.",
        ],
    ),
    # --- Account and line management ---
    Topic(
        doc_id="prepaid-recharge-validity",
        title="Prepaid Recharge, Renewal and Validity",
        doc_type="policy",
        market="ALL",
        effective_from=date(2026, 1, 1),
        facts=[
            "Prepaid lines can be recharged in the TelcoX app, on the website, "
            "or at retail stores.",
            "A prepaid plan renews automatically at the end of its validity if the balance is "
            "enough to pay for it.",
            "If the balance is not enough, the line enters a 7-day grace period in which only "
            "incoming calls work.",
            "A line with no recharge for 90 days after the grace period is deactivated.",
            "Unused main balance does not expire while the line is active.",
            "Customers can check their balance by dialling *100#.",
        ],
    ),
    Topic(
        doc_id="sim-replacement-esim",
        title="SIM Replacement and eSIM",
        doc_type="policy",
        market="ALL",
        effective_from=date(2026, 1, 1),
        facts=[
            "A physical SIM replacement is free once every 12 months.",
            "Further replacements cost 25 SAR in KSA and 25 AED in the UAE.",
            "eSIM can be activated in the TelcoX app and usually takes about 10 minutes.",
            "An eSIM can be moved to a new device at most 3 times per month.",
            "A valid ID is required for any SIM replacement or eSIM activation.",
            "A lost or stolen SIM should be blocked immediately in the app or by calling 900.",
        ],
    ),
    Topic(
        doc_id="number-porting-ksa",
        title="Moving Your Number to TelcoX (KSA)",
        doc_type="policy",
        market="KSA",
        effective_from=date(2026, 1, 1),
        facts=[
            "Moving a number to TelcoX from another operator is free.",
            "Porting takes up to 3 working days.",
            "The customer must have no unpaid bills with the previous operator.",
            "The name on the ID must match the name registered with the previous operator.",
            "A number can be ported out of TelcoX only after 60 days from activation.",
        ],
    ),
    Topic(
        doc_id="refunds-cancellation",
        title="Refunds and Cancellation",
        doc_type="policy",
        market="ALL",
        effective_from=date(2026, 1, 1),
        facts=[
            "Prepaid plan purchases cannot be refunded after the plan is activated.",
            "Postpaid lines can be cancelled with 30 days' notice.",
            "Cancelling a 12-month postpaid contract early costs 50% of the remaining "
            "monthly fees.",
            "Device instalments must be paid in full when a postpaid line is cancelled.",
            "Approved refunds are processed within 14 working days.",
        ],
    ),
    Topic(
        doc_id="plan-change-rules",
        title="Changing Your Plan",
        doc_type="policy",
        market="ALL",
        effective_from=date(2026, 1, 1),
        facts=[
            "Upgrading to a more expensive plan takes effect immediately; the customer pays the "
            "prorated difference for the rest of the cycle.",
            "Downgrading takes effect at the next renewal.",
            "Only one downgrade is allowed per billing cycle.",
            "Moving from prepaid to postpaid requires a credit check.",
            "Unused data from the old plan is not carried over after a plan change.",
        ],
    ),
    Topic(
        doc_id="data-rollover",
        title="Data Rollover",
        doc_type="policy",
        market="ALL",
        effective_from=date(2026, 1, 1),
        facts=[
            "Postpaid plans priced at 150 SAR or 150 AED per month or more roll over unused data.",
            "Up to 50% of the monthly data allowance can roll over, to the next cycle only.",
            "Rolled-over data is used before the new cycle's data.",
            "Prepaid plans and unlimited plans do not roll over data.",
        ],
    ),
    Topic(
        doc_id="5g-coverage-devices",
        title="5G Coverage and Devices",
        doc_type="policy",
        market="ALL",
        effective_from=date(2026, 1, 1),
        facts=[
            "In KSA, 5G is available in Riyadh, Jeddah and Dammam.",
            "In the UAE, 5G is available in Dubai, Abu Dhabi and Sharjah.",
            "5G needs a 5G-capable device and a plan that includes the 5G feature.",
            "There is no extra charge for 5G.",
            "Where 5G is unavailable, the phone switches to 4G automatically.",
        ],
    ),
    Topic(
        doc_id="customer-id-requirements",
        title="ID Requirements for New Lines",
        doc_type="policy",
        market="ALL",
        effective_from=date(2026, 1, 1),
        facts=[
            "In KSA, a National ID or Iqama verified by fingerprint is required.",
            "In the UAE, an Emirates ID is required.",
            "A maximum of 5 lines can be registered to one ID.",
            "Visitors can get a prepaid line with a passport and a valid visa.",
            "Visitor lines are prepaid only and stay active for 30 days unless renewed.",
        ],
    ),
    Topic(
        doc_id="parental-controls",
        title="Parental Controls",
        doc_type="policy",
        market="ALL",
        effective_from=date(2026, 1, 1),
        target_words=250,
        facts=[
            "Parental controls are free on all lines.",
            "They block adult content on mobile data.",
            "They are set up in the TelcoX app and protected by a 4-digit PIN.",
            "Parental controls apply to mobile data only, not to Wi-Fi.",
        ],
    ),
    Topic(
        doc_id="customer-support-contact",
        title="Contacting Customer Support",
        doc_type="policy",
        market="ALL",
        effective_from=date(2026, 1, 1),
        target_words=250,
        facts=[
            "Chat support in the TelcoX app is available 24/7.",
            "Calling 900 from a TelcoX line is free.",
            "From other phones, customers can call 800 1234 567.",
            "TelcoX stores are open from 10:00 to 22:00, Saturday to Thursday.",
        ],
    ),
    # --- T8: long document with a buried fact ---
    Topic(
        doc_id="terms-and-conditions",
        title="General Terms and Conditions",
        doc_type="policy",
        market="ALL",
        effective_from=date(2026, 1, 1),
        target_words=2000,
        outline=[
            "About these terms",
            "Definitions",
            "Who can hold an account",
            "Your TelcoX account",
            "Using the service",
            "Your responsibilities",
            "Charges and payments",
            "Billing questions and disputes",
            "Network coverage and service quality",
            "Changes to these terms",
            "Suspension and ending the service",
            "Personal data",
            "General provisions",
            "Governing law",
        ],
        facts=[
            "Account holders must be at least 18 years old.",
            "TelcoX may change these terms with 30 days' notice by SMS or email.",
            "Customers are responsible for all usage on their line, including by other people.",
            "Billing disputes must be raised within 60 days of the bill date.",
            "Customers aged 65 and above get a 20% discount on any postpaid plan "
            "after presenting a valid ID in a TelcoX store.",
            "TelcoX may suspend a line used for fraud, spam or illegal activity.",
            "Service availability is not guaranteed in remote areas or inside some buildings.",
            "Personal data is processed according to the TelcoX Privacy Notice.",
            "These terms are governed by the laws of the market in which the line is registered.",
        ],
    ),
    # --- FAQs (only current facts, consistent with the policies above) ---
    Topic(
        doc_id="faq-prepaid",
        title="Prepaid FAQs",
        doc_type="faq",
        market="ALL",
        effective_from=date(2026, 1, 1),
        target_words=450,
        facts=[
            "Check your balance by dialling *100#.",
            "Recharge in the TelcoX app, on the website, or at retail stores.",
            "If the balance is too low at renewal, a 7-day grace period starts in which only "
            "incoming calls work.",
            "Prepaid plans do not roll over unused data.",
            "Prepaid plan purchases cannot be refunded after activation.",
            "eSIM activation in the app usually takes about 10 minutes.",
            "Upgrades take effect immediately; downgrades take effect at the next renewal.",
        ],
    ),
    Topic(
        doc_id="faq-postpaid",
        title="Postpaid FAQs",
        doc_type="faq",
        market="ALL",
        effective_from=date(2026, 1, 1),
        target_words=450,
        facts=[
            "Bills are issued on the 1st of each month in KSA and on the 5th in the UAE.",
            "The late payment fee is 25 SAR in KSA and 30 AED in the UAE.",
            "Postpaid plans priced at 150 or more per month roll over up to 50% of unused data "
            "to the next cycle.",
            "Postpaid lines can be cancelled with 30 days' notice.",
            "Electronic bills are free.",
            "VAT is included in prices in KSA and added on top in the UAE.",
        ],
    ),
    Topic(
        doc_id="faq-roaming",
        title="Roaming FAQs",
        doc_type="faq",
        market="ALL",
        effective_from=date(2026, 7, 1),
        target_words=450,
        facts=[
            "Data roaming is disabled by default on new lines.",
            "Turn on roaming in the TelcoX app or by dialling *123#.",
            "The GCC Day Pass gives 1 GB for 24 hours: 25 SAR in KSA, 20 AED in the UAE.",
            "Roaming data is suspended at 500 SAR in KSA or 400 AED in the UAE per billing cycle, "
            "unless you opt out in the app.",
            "Roaming add-on plans give a fixed amount of roaming data in specific countries; "
            "see the add-on plan pages.",
        ],
    ),
    Topic(
        doc_id="faq-network-devices",
        title="Network and Device FAQs",
        doc_type="faq",
        market="ALL",
        effective_from=date(2026, 1, 1),
        target_words=400,
        facts=[
            "5G is available in Riyadh, Jeddah and Dammam in KSA, and in Dubai, Abu Dhabi and "
            "Sharjah in the UAE.",
            "5G has no extra charge but needs a 5G-capable device and a plan with the 5G feature.",
            "Tethering is allowed on all plans; on unlimited plans it is capped at "
            "20 GB per cycle.",
            "After the fair usage threshold on unlimited plans, speed drops to 1 Mbps.",
            "A lost SIM should be blocked immediately in the app or by calling 900.",
        ],
    ),
]

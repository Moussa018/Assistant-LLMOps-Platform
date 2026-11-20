"""Generate the synthetic demo datasets for the two demo tenants.

    python scripts/make_demo_data.py

Real tenants bring their own data/tenants/<name>/tickets.jsonl,
one {"text": ..., "label": ...} object per line.
"""

import json
import random
from pathlib import Path

SLOTS = {
    "order": ["#48213", "#50991", "A-7781", "#33017", "B-1204", "#61530"],
    "amount": ["$24.99", "$89", "$12.50", "$149.00", "$7.99", "$310"],
    "code": ["SPRING20", "WELCOME10", "FREESHIP", "VIP15"],
    "item": [
        "running shoes",
        "blender",
        "winter jacket",
        "phone case",
        "desk lamp",
        "backpack",
        "headphones",
        "coffee maker",
    ],
    "city": ["Lyon", "Casablanca", "Toronto", "Madrid", "Dubai", "Austin"],
    "days": ["3", "5", "8", "10", "two", "four"],
    "day": ["Monday", "Friday", "yesterday", "this morning"],
    "color": ["black", "navy", "red", "white"],
    "size": ["medium", "size 42", "large", "small"],
    "device": ["iPhone 15", "a Pixel 8", "my MacBook", "Android tablets"],
    "email": ["sam.k@mail.com", "my work address", "a new Gmail"],
    "page": ["Reports", "Dashboards", "Users", "Billing", "Funnels"],
    "month": ["March", "last quarter", "September", "this week"],
    "err": ["500", "E-1042", "403", "timeout"],
    "format": ["Excel", "PDF", "Parquet", "Google Sheets"],
    "team": ["finance", "sales", "support"],
    "time": ["9am", "an hour ago", "this morning", "noon UTC"],
    "region": ["Europe", "Asia", "the US east coast"],
    "tool": ["Salesforce", "HubSpot", "Slack", "BigQuery", "Segment", "Zapier"],
    "idp": ["Okta", "Azure AD", "Google Workspace"],
    "seats": ["50", "120", "300", "25"],
}

TENANTS = {
    "acme": {
        "billing": [
            "I was charged twice for order {order}",
            "My card was charged {amount} but the order still shows unpaid",
            "Can I get an invoice for order {order}?",
            "Why is there an extra fee of {amount} on my bill?",
            "The payment failed but the money left my account",
            "How do I change the payment method on my order?",
            "The promo code {code} didn't apply and I paid full price",
            "I need a VAT invoice for a company purchase",
            "You billed me {amount} for a subscription I cancelled",
        ],
        "shipping": [
            "Where is my package? Order {order} hasn't arrived",
            "Tracking hasn't updated in {days} days",
            "Do you ship to {city}?",
            "The courier marked it delivered but I got nothing",
            "How long does delivery to {city} take?",
            "Can I change the delivery address for order {order}?",
            "My parcel seems stuck at customs",
            "Delivery was supposed to be {day} and nobody came",
        ],
        "returns": [
            "I want to return the {item} I bought",
            "The {item} arrived broken, how do I send it back?",
            "How many days do I have to return a {item}?",
            "I sent my return {days} days ago, when do I get my money back?",
            "Wrong size on the {item}, can I exchange it?",
            "Do I have to pay for return shipping?",
            "The return label link doesn't work",
        ],
        "account": [
            "I can't log in to my account",
            "The password reset email never arrives",
            "How do I delete my account?",
            "I want to change the email on my account to {email}",
            "My account got locked after too many attempts",
            "Someone changed my password, I think I was hacked",
            "How do I stop getting your marketing emails?",
        ],
        "product_question": [
            "Is the {item} waterproof?",
            "Does the {item} come with a warranty?",
            "What's the difference between the {item} and the {item} pro?",
            "Is the {item} available in {color}?",
            "Which {item} size should I get if I'm usually a {size}?",
            "When will the {item} be back in stock?",
            "Is the {item} compatible with {device}?",
        ],
    },
    "globex": {
        "bug_report": [
            "The export button does nothing on the {page} page",
            "Charts show wrong totals for {month}",
            "Getting error {err} when saving a report",
            "Filters reset every time I refresh {page}",
            "The date picker is off by one day",
            "Duplicate rows appear in the {page} table",
            "Sorting by date is broken on {page}",
        ],
        "feature_request": [
            "Could you add a dark mode?",
            "It would be great to schedule reports weekly",
            "Please support exporting to {format}",
            "Can we get role-based permissions for the {team} team?",
            "Add an option to share dashboards with a public link",
            "We'd love keyboard shortcuts in the editor",
            "Any plans for custom alerts on {page}?",
        ],
        "outage": [
            "The whole app is down for our team",
            "Nothing loads since {time}, is there an incident?",
            "The API returns 503 for every request",
            "Nobody can log in, the login page times out",
            "{page} is completely unreachable from {region}",
            "All dashboards are blank right now",
        ],
        "integration": [
            "How do I connect {tool} to Globex?",
            "The {tool} sync stopped importing data",
            "Our webhook to {tool} isn't firing",
            "Is there an API endpoint to push events?",
            "SSO with {idp} fails with an invalid assertion",
            "Can we import historical data from {tool}?",
        ],
        "pricing": [
            "How much is the enterprise plan?",
            "Can we get a discount for {seats} seats?",
            "What happens if we go over the event limit?",
            "Do you offer nonprofit pricing?",
            "We want to downgrade to the starter plan",
            "Is annual billing cheaper than monthly?",
            "How is the price calculated for {seats} users?",
        ],
    },
}

PREFIX = ["", "", "", "Hi, ", "Hello team, ", "Urgent: ", "Hey, ", "Good morning. "]
SUFFIX = ["", "", "", " Thanks", " Please help.", "!!", " Any update?", " Thank you in advance."]


def fill(template: str, rng: random.Random) -> str:
    out = template
    for key, values in SLOTS.items():
        if "{" + key + "}" in out:
            out = out.replace("{" + key + "}", rng.choice(values))
    text = rng.choice(PREFIX) + out + rng.choice(SUFFIX)
    return text.lower() if rng.random() < 0.15 else text


def main(per_label: int = 40, seed: int = 7) -> None:
    rng = random.Random(seed)
    for tenant, labels in TENANTS.items():
        rows, seen = [], set()
        for label, templates in labels.items():
            tries = 0
            count = 0
            while count < per_label and tries < per_label * 50:
                tries += 1
                text = fill(rng.choice(templates), rng)
                if text.lower() in seen:
                    continue
                seen.add(text.lower())
                rows.append({"text": text, "label": label})
                count += 1
        rng.shuffle(rows)
        out = Path("data/tenants") / tenant / "tickets.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        print(f"{tenant}: {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()

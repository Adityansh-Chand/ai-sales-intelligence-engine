FEATURE_ORDER = [
    "visits",
    "spend",
    "account_age_days",
    "usage_frequency",
    "support_tickets",
    "renewal_days",
]


def build_features(customer):
    return [
        float(customer.get("visits", 0)),
        float(customer.get("spend", 0)),
        float(customer.get("account_age_days", 0)),
        float(customer.get("usage_frequency", 0)),
        float(customer.get("support_tickets", 0)),
        float(customer.get("renewal_days", 365)),
    ]


def features_to_dict(features):
    return dict(zip(FEATURE_ORDER, features))

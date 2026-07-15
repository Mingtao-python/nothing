def find_failure_cases(query):
    return [
        "Ambiguous word: bank (river vs finance)",
        "Out-of-domain query",
        "Very short text",
        "Noisy text",
        "Rare words not in training data"
    ]

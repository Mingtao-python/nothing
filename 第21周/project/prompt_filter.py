def filter_prompt(p):
    if len(p) > 300:
        return False, "Input too long"
    bad = ["ignore", "system", "override", "bypass"]
    if any(w in p.lower() for w in bad):
        return False, "Unsafe prompt detected"
    return True, p

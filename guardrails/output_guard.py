def validate_output(output, min_length=20):
    if output is None:
        return False

    output_text = str(output).strip()
    if len(output_text) < min_length:
        return False

    lowered = output_text.lower()
    
    blocked_starts = [
        "sorry",
        "i cannot",
        "i don't know"
    ]

    if any(lowered.startswith(p) for p in blocked_starts):
        return False

    return True
 
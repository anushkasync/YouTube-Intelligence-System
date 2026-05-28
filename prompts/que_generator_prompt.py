QA_PROMPT = """
Generate exam questions.

Mode:
raw: 3 basic questions
medium: 3–6 conceptual questions
long: 6–12 mixed (fact + concept + application)

Rules:
- no answers
- no repetition
- only important concepts
- clear exam-style questions

Content:
{content}

Mode:
{mode}

Questions:
"""
from prompts.summary_prompt import SUMMARY_PROMPT
from prompts.que_generator_prompt import QA_PROMPT
from prompts.keypoints_prompt import KEYPOINTS_PROMPT


def summarize_small(chunks, llm):
    text = "\n".join(chunks[:3])
    mode = "raw"

    prompt = SUMMARY_PROMPT.format(content=text, mode=mode)

    return llm.invoke(prompt).content


def summarize_medium(chunks, llm):

    text = "\n".join(chunks)
    mode = "medium"

    prompt = SUMMARY_PROMPT.format(content=text, mode=mode)

    return llm.invoke(prompt).content

def summarize_long(chunks, llm):

    if not chunks:
        return "Clustering failed. Unable to summarize long transcript safely."

    merged_text = "\n\n".join(chunks)
    merged_text = merged_text
    
    prompt = SUMMARY_PROMPT.format(
        content=merged_text,
        mode="long"
    )

    return llm.invoke(prompt).content

def generate_summary(processed_chunks, llm, mode):

    if mode in ("raw", "small"):

        return summarize_small(
            processed_chunks["raw"],
            llm
        )

    elif mode == "medium":

        return summarize_medium(
            processed_chunks["medium"],
            llm
        )

    elif mode == "long":

        return summarize_long(
            processed_chunks["long"],
            llm
        )

    else:
        raise ValueError(
            f"Unsupported summary mode: {mode}"
        )
        
def generate_questions(processed_chunks, llm, mode="medium"):
    """
    Generate questions based on processed chunks from Chunker.

    mode:
    - small → raw chunks
    - medium → top-k chunks
    - long → KMeans representative chunks
    """

    if mode == "small":
        text = "\n".join(processed_chunks["raw"])
        prompt = QA_PROMPT.format(content=text, mode = mode)

    elif mode == "medium":
        text = "\n".join(processed_chunks["medium"])

        prompt = QA_PROMPT.format(content=text, mode = mode)

    else:
        text = "\n".join(processed_chunks["long"])

        prompt = QA_PROMPT.format(content=text, mode = mode)

    return llm.invoke(prompt).content

def generate_keypoints(processed_chunks, llm, mode="medium"):

    if mode == "small":
        text = "\n".join(processed_chunks["raw"])
        prompt = KEYPOINTS_PROMPT.format(content=text, mode = mode)

    elif mode == "medium":
        text = "\n".join(processed_chunks["medium"])
        prompt = KEYPOINTS_PROMPT.format(content=text, mode = mode)

    else:
        text = "\n".join(processed_chunks["long"])
        prompt = KEYPOINTS_PROMPT.format(content=text, mode = mode)

    return llm.invoke(prompt).content
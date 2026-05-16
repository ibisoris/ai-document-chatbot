import os

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError


NO_CONTEXT_ANSWER = "I could not find that information in the uploaded document."
DEFAULT_MODEL = "gpt-4.1-mini"
QUOTA_ERROR_ANSWER = (
    "The app reached OpenAI, but your OpenAI account does not currently have "
    "available API quota. Please check your OpenAI billing or usage limits, "
    "then try again."
)


class MissingOpenAIKeyError(Exception):
    """Raised when OPENAI_API_KEY is missing from the environment."""


def get_openai_api_key() -> str:
    """Load the OpenAI API key from the .env file."""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise MissingOpenAIKeyError(
            "OPENAI_API_KEY was not found. Add it to your .env file."
        )

    return api_key


def build_rag_prompt(question: str, retrieved_chunks: list[str]) -> str:
    """Build a prompt that keeps the LLM grounded in retrieved document text."""
    context = "\n\n---\n\n".join(
        f"Context section {index}:\n{chunk}"
        for index, chunk in enumerate(retrieved_chunks, start=1)
    )

    return f"""
Use only the context sections below to answer the user's question.

If the answer is not clearly present in the context, return exactly:
{NO_CONTEXT_ANSWER}

Write a clear, helpful answer in plain English. Do not mention information that
is not supported by the context.

Formatting rules:
- Use short paragraphs.
- Use bullet points when they make the answer easier to scan.
- Keep the answer concise unless the user asks for detail.

Context:
{context}

Question:
{question}
""".strip()


def generate_answer(question: str, retrieved_chunks: list[str]) -> str:
    """Generate a natural-language answer using retrieved document chunks."""
    if not retrieved_chunks:
        return NO_CONTEXT_ANSWER

    api_key = get_openai_api_key()
    client = OpenAI(api_key=api_key)
    prompt = build_rag_prompt(question, retrieved_chunks)

    try:
        response = client.responses.create(
            model=os.getenv("OPENAI_MODEL", DEFAULT_MODEL),
            instructions=(
                "You are a document question-answering assistant. Answer only "
                "from the provided context. Be clear, concise, and easy to read."
            ),
            input=prompt,
            temperature=0.2,
        )
    except OpenAIError as error:
        error_text = str(error).lower()
        if "insufficient_quota" in error_text or "exceeded your current quota" in error_text:
            return QUOTA_ERROR_ANSWER

        return f"OpenAI API error: {error}"

    answer = response.output_text.strip()
    return answer or NO_CONTEXT_ANSWER

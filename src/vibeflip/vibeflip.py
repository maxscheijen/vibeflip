import os
from typing import Type, TypeVar

from openai import OpenAI
from pydantic import BaseModel


class VibeflipError(Exception):
    """Raised when vibeflip cannot parse a valid response."""


class VibeflipResponse(BaseModel):
    flip: bool


T = TypeVar("T", bound=BaseModel)


def structured_output(content: str, response_format: Type[T], model: str) -> T:
    """
    Calls OpenAI API and parses the response into the given Pydantic model.

    Args:
        content (str): User message to send to the model.
        response_format (Type[T]): Pydantic model class to parse the output into.
        model (str): OpenAI model name.

    Returns:
        T: An instance of response_format containing the parsed response.

    Raises:
        VibeflipError: If the response could not be parsed.
    """

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise VibeflipError("The OPENAI_API_KEY environment variable is not set")

    client = OpenAI()

    response = client.chat.completions.parse(
        model=model,
        temperature=0.0,
        messages=[{"role": "user", "content": content}],
        response_format=response_format,
    )

    response_model = response.choices[0].message.parsed

    if response_model:
        return response_model

    raise VibeflipError("`vibeflip` failed: no valid response parsed")

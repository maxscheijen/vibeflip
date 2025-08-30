from . import vibeflip as vf


def vibeflip(prompt: str = "True or False", model: str = "gpt-4.1-mini") -> bool:
    """
    Coin flip library that uses GPT for true randomness

    Args:
        prompt (str): Prompt for the AI to answer True or False. Defaults to "True or False".
        model (str): OpenAI model to use. Defaults to "gpt-4.1-mini".

    Returns:
        bool: The random True or False decision.

    Raises:
        vf.VibeflipError: If the API call fails or the response cannot be parsed.
    """
    return vf.structured_output(
        content=prompt, response_format=vf.VibeflipResponse, model=model
    ).flip

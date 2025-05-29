import logging
import time
from typing import Any, List, Dict, Optional


class LlmHandler:
    """
    Minimal unified helper for OpenAI-style and Hugging Face chat-completion APIs.
    """

    def __init__(
        self,
        *,
        backend: str = "openai",
        max_attempts: int = 10,
        seed: int = 14,
        **client_kwargs: Any,
    ) -> None:
        self.max_attempts, self.seed = max_attempts, seed

        # pick backend
        if backend == "openai":
            import openai, httpx  # lazy import

            transport = httpx.HTTPTransport(verify=False)
            client_kwargs["http_client"] = httpx.Client(verify=False)

            self._chat = openai.OpenAI(**client_kwargs).chat.completions.create

        elif backend == "hf":
            from huggingface_hub import InferenceClient  # lazy import

            self._chat = InferenceClient(**client_kwargs).chat.completions.create

        else:
            raise ValueError("backend must be 'auto', 'openai', or 'hf'")

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: str,
        **kwargs: Any,
    ) -> Optional[str]:
        for attempt in range(self.max_attempts):
            try:
                logging.info("Attempt %d", attempt + 1)
                resp = self._chat(model=model, messages=messages, seed=self.seed, **kwargs)
                return resp.choices[0].message.content
            except Exception as exc:  # noqa: BLE001
                logging.error("Error: %s", exc, exc_info=True)
                time.sleep(2**attempt)  # exponential back-off
        logging.error("All attempts failed.")
        return None

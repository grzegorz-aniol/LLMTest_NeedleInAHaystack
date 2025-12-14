import os
from operator import itemgetter
from typing import Optional

from openai import AsyncOpenAI
from langchain_openai import ChatOpenAI
from langchain_classic.prompts import PromptTemplate
import tiktoken
from transformers import AutoTokenizer

from .model import ModelProvider


class OpenAI(ModelProvider):
    """Provider for OpenAI-hosted models using the official OpenAI client.

    This class uses ``tiktoken`` for tokenisation, matching OpenAI's models.
    It remains the default provider when ``--provider openai`` is used.
    """

    DEFAULT_MODEL_KWARGS: dict = dict(max_tokens=300, temperature=0)

    def __init__(
        self,
        model_name: str = "gpt-5-mini",
        model_kwargs: dict = DEFAULT_MODEL_KWARGS,
        base_url: Optional[str] = None,
    ) -> None:
        """Initialise the OpenAI provider.

        Args:
            model_name: Name of the OpenAI model to use.
            model_kwargs: Model configuration (e.g. ``max_tokens``, ``temperature``).
            base_url: Optional alternate base URL for an OpenAI-compatible endpoint.
        """
        api_key = os.getenv("NIAH_MODEL_API_KEY")
        if not api_key:
            raise ValueError("NIAH_MODEL_API_KEY must be in env.")

        self.model_name = model_name
        self.model_kwargs = model_kwargs
        self.api_key = api_key
        self.base_url = base_url

        if self.base_url:
            self.model = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
        else:
            self.model = AsyncOpenAI(api_key=self.api_key)

        # Use OpenAI's tokenizer mapping
        self.tokenizer = tiktoken.encoding_for_model(self.model_name)

    async def evaluate_model(self, prompt: str) -> str:
        """Evaluate a given prompt using the OpenAI model and return the response text."""
        response = await self.model.chat.completions.create(
            model=self.model_name,
            messages=prompt,
            **self.model_kwargs,
        )
        return response.choices[0].message.content

    def generate_prompt(self, context: str, retrieval_question: str) -> list[dict[str, str]]:
        """Generate a chat-style prompt from context and retrieval question."""
        return [
            {
                "role": "system",
                "content": "You are a helpful AI bot that answers questions for a user. Keep your response short and direct",
            },
            {
                "role": "user",
                "content": context,
            },
            {
                "role": "user",
                "content": f"{retrieval_question} Don't give information outside the document or repeat your findings",
            },
        ]

    def encode_text_to_tokens(self, text: str) -> list[int]:
        """Encode text into tokens using ``tiktoken``."""
        return self.tokenizer.encode(text)

    def decode_tokens(self, tokens: list[int], context_length: Optional[int] = None) -> str:
        """Decode tokens back into text using ``tiktoken``."""
        return self.tokenizer.decode(tokens[:context_length])

    def get_langchain_runnable(self, context: str):
        """Create a LangChain runnable that queries the OpenAI model."""

        template = """You are a helpful AI bot that answers questions for a user. Keep your response short and direct" \n
        \n ------- \n 
        {context} 
        \n ------- \n
        Here is the user question: \n --- --- --- \n {question} \n Don't give information outside the document or repeat your findings."""

        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"],
        )

        if getattr(self, "base_url", None):
            model = ChatOpenAI(temperature=0, model=self.model_name, base_url=self.base_url)
        else:
            model = ChatOpenAI(temperature=0, model=self.model_name)

        chain = (
            {"context": lambda x: context, "question": itemgetter("question")}  # type: ignore[arg-type]
            | prompt
            | model
        )
        return chain


class LocalOpenAI(ModelProvider):
    """Provider for *local* OpenAI-compatible endpoints using a Hugging Face tokenizer.

    This variant is intended to be used when ``--provider local`` is selected.
    It talks to an OpenAI-compatible HTTP server via ``AsyncOpenAI`` but uses
    ``transformers.AutoTokenizer`` so that tokenisation matches the local model
    (e.g. a Hugging Face model name or path).
    """

    DEFAULT_MODEL_KWARGS: dict = dict(max_tokens=300, temperature=0)

    def __init__(
        self,
        model_name: str,
        model_kwargs: dict = DEFAULT_MODEL_KWARGS,
        base_url: Optional[str] = None,
        tokenizer_hf_id: Optional[str] = None,
    ) -> None:
        """Initialise LocalOpenAI.

        Args:
            model_name: The model identifier passed to the OpenAI-compatible server.
            model_kwargs: Model configuration (e.g. ``max_tokens``, ``temperature``).
            base_url: Base URL for the OpenAI-compatible HTTP endpoint.
            tokenizer_hf_id: Optional Hugging Face model id to use for tokenisation.
                If provided, this value is passed to ``AutoTokenizer.from_pretrained``.
                If omitted, ``model_name`` is used instead.
        """
        api_key = os.getenv("NIAH_MODEL_API_KEY")
        if not api_key:
            raise ValueError("NIAH_MODEL_API_KEY must be in env.")

        if not base_url:
            raise ValueError(
                "base_url must be provided when using LocalOpenAI (local provider)."
            )

        self.model_name = model_name
        self.model_kwargs = model_kwargs
        self.api_key = api_key
        self.base_url = base_url
        self.tokenizer_hf_id = tokenizer_hf_id

        # AsyncOpenAI client pointed at the local OpenAI-compatible server.
        self.model = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

        # Use a Hugging Face tokenizer. Prefer an explicit HF id if provided,
        # otherwise fall back to the model_name that is sent to the server.
        hf_id = self.tokenizer_hf_id or self.model_name
        self.tokenizer = AutoTokenizer.from_pretrained(hf_id)

    async def evaluate_model(self, prompt: str) -> str:
        """Evaluate a given prompt using the local OpenAI-compatible model."""
        response = await self.model.chat.completions.create(
            model=self.model_name,
            messages=prompt,
            **self.model_kwargs,
        )
        return response.choices[0].message.content

    def generate_prompt(self, context: str, retrieval_question: str) -> list[dict[str, str]]:
        """Generate a chat-style prompt from context and retrieval question."""
        return [
            {
                "role": "system",
                "content": "You are a helpful AI bot that answers questions for a user. Keep your response short and direct",
            },
            {
                "role": "user",
                "content": context,
            },
            {
                "role": "user",
                "content": f"{retrieval_question} Don't give information outside the document or repeat your findings",
            },
        ]

    def encode_text_to_tokens(self, text: str) -> list[int]:
        """Encode text into tokens using the Hugging Face tokenizer.

        ``add_special_tokens`` is disabled so that token counts align more
        closely with the raw input text, which is usually what the benchmark
        cares about.
        """
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode_tokens(self, tokens: list[int], context_length: Optional[int] = None) -> str:
        """Decode tokens back into text using the Hugging Face tokenizer."""
        slice_tokens = tokens[:context_length] if context_length is not None else tokens
        return self.tokenizer.decode(slice_tokens, skip_special_tokens=True)

    def get_langchain_runnable(self, context: str):
        """Create a LangChain runnable that queries the local endpoint."""

        template = """You are a helpful AI bot that answers questions for a user. Keep your response short and direct" \n
        \n ------- \n 
        {context} 
        \n ------- \n
        Here is the user question: \n --- --- --- \n {question} \n Don't give information outside the document or repeat your findings."""

        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"],
        )

        model = ChatOpenAI(
            temperature=0,
            model=self.model_name,
            base_url=self.base_url,
        )

        chain = (
            {"context": lambda x: context, "question": itemgetter("question")}  # type: ignore[arg-type]
            | prompt
            | model
        )
        return chain

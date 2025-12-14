import os
from typing import Optional

from .evaluator import Evaluator
from .cache import EvaluationCache

from langchain_classic.evaluation import load_evaluator
from langchain_openai import ChatOpenAI


class OpenAIEvaluator(Evaluator):
    DEFAULT_MODEL_KWARGS: dict = dict(temperature=0)
    CRITERIA = {
        "accuracy": """
                Score 1: The answer is completely unrelated to the reference.
                Score 3: The answer has minor relevance but does not align with the reference.
                Score 5: The answer has moderate relevance but contains inaccuracies.
                Score 7: The answer aligns with the reference but has minor omissions.
                Score 10: The answer is completely accurate and aligns perfectly with the reference.
                Only respond with a numerical score"""
    }

    def __init__(
        self,
        model_name: str = "gpt-5-mini",
        model_kwargs: dict = DEFAULT_MODEL_KWARGS,
        true_answer: str | None = None,
        question_asked: str | None = None,
        base_url: Optional[str] = None,
        use_cache: bool = True,
        cache: Optional[EvaluationCache] = None,
    ) -> None:
        """Initialise the OpenAI evaluator.

        Args:
            model_name: The name of the model.
            model_kwargs: Model configuration. Default is {temperature: 0}.
            true_answer: The true answer to the question asked.
            question_asked: The question asked to the model.
            base_url: Optional base URL for an OpenAI-compatible endpoint.
            use_cache: Whether to enable in-memory caching of evaluation scores.
            cache: Optional shared cache instance. If not provided and
                ``use_cache`` is True, a new :class:`EvaluationCache` will be
                created for this evaluator.
        """

        if (not true_answer) or (not question_asked):
            raise ValueError("true_answer and question_asked must be supplied with init.")

        self.model_name = model_name
        self.model_kwargs = model_kwargs
        self.true_answer = true_answer
        self.question_asked = question_asked

        api_key = os.getenv("NIAH_EVALUATOR_API_KEY")
        if not api_key:
            raise ValueError("NIAH_EVALUATOR_API_KEY must be in env for using openai evaluator.")

        self.api_key = api_key
        self.base_url = base_url

        # Cache configuration
        self.use_cache = use_cache
        self._cache: Optional[EvaluationCache] = cache if cache is not None else (
            EvaluationCache() if use_cache else None
        )

        if self.base_url:
            # ChatOpenAI from langchain_community does not currently accept a base_url argument
            # directly for all versions, but it will honor the OPENAI_BASE_URL environment variable.
            os.environ["OPENAI_BASE_URL"] = self.base_url

        self.evaluator = ChatOpenAI(
            model=self.model_name,
            openai_api_key=self.api_key,
            **self.model_kwargs,
        )

    def evaluate_response(self, response: str) -> int:
        """Evaluate a model response and return an integer score.

        When caching is enabled, this method will first attempt to read from
        the cache using a **strict** key consisting of
        ``(prediction, reference, input)`` which map to::

            prediction -> response (the model's answer)
            reference  -> self.true_answer (ground-truth answer)
            input      -> self.question_asked (the question asked)
        """
        prediction = response
        reference = self.true_answer
        input_ = self.question_asked

        # Try cache first, if enabled
        if self._cache is not None:
            cached_score = self._cache.get(prediction, reference, input_)
            if cached_score is not None:
                return cached_score

        evaluator = load_evaluator(
            "labeled_score_string",
            criteria=self.CRITERIA,
            llm=self.evaluator,
        )

        eval_result = evaluator.evaluate_strings(
            # The model's response
            prediction=prediction,
            # The actual answer
            reference=reference,
            # The question asked
            input=input_,
        )

        score = int(eval_result["score"])

        # Store result in cache for this (prediction, reference, input_) triplet
        if self._cache is not None:
            self._cache.set(prediction, reference, input_, score)

        return score

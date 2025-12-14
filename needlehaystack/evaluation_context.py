from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel


class EvaluationContext(BaseModel):
    """Universal evaluation result model for single and multi-needle tests.

    Attributes
    ----------
    model:
        Name of the model under test.
    context_length:
        Total context length in tokens used for the test.
    depth_percent:
        Target depth percentage used for needle insertion.
    version:
        Logical run/version identifier.
    needles:
        Needles placed in the haystack. For single-needle tests this is a one-element list.
    model_response:
        Raw model response text.
    score:
        Evaluator score for the response.
    test_duration_seconds:
        Wall-clock duration of the evaluation call.
    test_timestamp_utc:
        UTC timestamp string when the test finished.
    file_name:
        Base filename used when persisting context/results.
    insertion_points:
        Token indices at which needles were inserted into the context. Single-needle tests
        will contain exactly one element; multi-needle tests will contain one per needle.
    """

    model: str
    context_length: int
    depth_percent: float
    version: int

    needles: List[str]

    model_response: Optional[str] = None
    score: Optional[float] = None

    test_duration_seconds: Optional[float] = None
    test_timestamp_utc: Optional[str] = None

    file_name: Optional[str] = None

    insertion_points: List[int] = []

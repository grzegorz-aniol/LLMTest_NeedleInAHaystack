from unittest.mock import patch, MagicMock, call, ANY

from needlehaystack.evaluators import OpenAIEvaluator
from needlehaystack.evaluators.cache import EvaluationCache

QUESTION_ASKED = "What is the color of the sky?"
QUESTION_ASKER_2 = "What color is the sky?"  # Slightly different wording
QUESTION_ANSWER = "Sky is blue"
QUESTION_ANSWER_2 = "The sky is blue"  # Slight variation
API_KEY = "abc"
SCORE = 123
SCORE_2 = 456
TEMPERATURE = 0
MODEL = "gpt-5-mini"


@patch("needlehaystack.evaluators.openai.ChatOpenAI")
@patch("needlehaystack.evaluators.openai.load_evaluator")
def test_openai(mock_load_evaluator, mock_chat_open_ai, monkeypatch):
    monkeypatch.setenv("NIAH_EVALUATOR_API_KEY", API_KEY)

    mock_evaluator = MagicMock()
    mock_evaluator.evaluate_strings.return_value = {"score": str(SCORE)}

    mock_load_evaluator.return_value = mock_evaluator

    evaluator = OpenAIEvaluator(question_asked=QUESTION_ASKED, true_answer=QUESTION_ANSWER)
    result = evaluator.evaluate_response("Something")

    assert mock_chat_open_ai.call_args == call(
        model=MODEL, temperature=TEMPERATURE, openai_api_key=API_KEY
    )
    assert mock_load_evaluator.call_args == call(
        "labeled_score_string", criteria=OpenAIEvaluator.CRITERIA, llm=ANY
    )

    assert result == SCORE


@patch("needlehaystack.evaluators.openai.ChatOpenAI")
@patch("needlehaystack.evaluators.openai.load_evaluator")
def test_openai_evaluator_uses_cache_for_identical_triplet(
    mock_load_evaluator, mock_chat_open_ai, monkeypatch
):
    """Same (prediction, reference, input) triplet should hit cache."""
    monkeypatch.setenv("NIAH_EVALUATOR_API_KEY", API_KEY)

    mock_evaluator = MagicMock()
    mock_evaluator.evaluate_strings.return_value = {"score": str(SCORE)}
    mock_load_evaluator.return_value = mock_evaluator

    evaluator = OpenAIEvaluator(
        question_asked=QUESTION_ASKED,
        true_answer=QUESTION_ANSWER,
        use_cache=True,
    )

    # First call populates cache
    first_result = evaluator.evaluate_response("Something")
    # Second call with identical prediction/reference/input should use cache
    second_result = evaluator.evaluate_response("Something")

    assert first_result == SCORE
    assert second_result == SCORE

    # Under caching, evaluate_strings should be called only once
    mock_evaluator.evaluate_strings.assert_called_once()


@patch("needlehaystack.evaluators.openai.ChatOpenAI")
@patch("needlehaystack.evaluators.openai.load_evaluator")
def test_openai_evaluator_cache_key_changes_with_different_reference(
    mock_load_evaluator, mock_chat_open_ai, monkeypatch
):
    """Different reference should NOT reuse cache even if prediction/input match."""
    monkeypatch.setenv("NIAH_EVALUATOR_API_KEY", API_KEY)

    cache = EvaluationCache()

    mock_evaluator = MagicMock()
    # Use side_effect to simulate different scores, though we mostly care about call count
    mock_evaluator.evaluate_strings.side_effect = [
        {"score": str(SCORE)},
        {"score": str(SCORE_2)},
    ]
    mock_load_evaluator.return_value = mock_evaluator

    evaluator1 = OpenAIEvaluator(
        question_asked=QUESTION_ASKED,
        true_answer=QUESTION_ANSWER,
        use_cache=True,
        cache=cache,
    )

    evaluator2 = OpenAIEvaluator(
        question_asked=QUESTION_ASKED,
        true_answer=QUESTION_ANSWER_2,
        use_cache=True,
        cache=cache,
    )

    # Same prediction and input, but different reference
    result1 = evaluator1.evaluate_response("Something")
    result2 = evaluator2.evaluate_response("Something")

    assert result1 == SCORE
    assert result2 == SCORE_2

    # Because reference differs, cache keys differ; evaluator should be called twice
    assert mock_evaluator.evaluate_strings.call_count == 2


@patch("needlehaystack.evaluators.openai.ChatOpenAI")
@patch("needlehaystack.evaluators.openai.load_evaluator")
def test_openai_evaluator_cache_key_changes_with_different_input(
    mock_load_evaluator, mock_chat_open_ai, monkeypatch
):
    """Different input should NOT reuse cache even if prediction/reference match."""
    monkeypatch.setenv("NIAH_EVALUATOR_API_KEY", API_KEY)

    cache = EvaluationCache()

    mock_evaluator = MagicMock()
    mock_evaluator.evaluate_strings.side_effect = [
        {"score": str(SCORE)},
        {"score": str(SCORE_2)},
    ]
    mock_load_evaluator.return_value = mock_evaluator

    evaluator1 = OpenAIEvaluator(
        question_asked=QUESTION_ASKED,
        true_answer=QUESTION_ANSWER,
        use_cache=True,
        cache=cache,
    )

    evaluator2 = OpenAIEvaluator(
        question_asked=QUESTION_ASKER_2,
        true_answer=QUESTION_ANSWER,
        use_cache=True,
        cache=cache,
    )

    # Same prediction and reference, but different input
    result1 = evaluator1.evaluate_response("Something")
    result2 = evaluator2.evaluate_response("Something")

    assert result1 == SCORE
    assert result2 == SCORE_2

    # Because input differs, cache keys differ; evaluator should be called twice
    assert mock_evaluator.evaluate_strings.call_count == 2


@patch("needlehaystack.evaluators.openai.ChatOpenAI")
@patch("needlehaystack.evaluators.openai.load_evaluator")
def test_openai_evaluator_no_cache_when_flag_disabled(
    mock_load_evaluator, mock_chat_open_ai, monkeypatch
):
    """When use_cache is False, evaluator should be called each time."""
    monkeypatch.setenv("NIAH_EVALUATOR_API_KEY", API_KEY)

    mock_evaluator = MagicMock()
    mock_evaluator.evaluate_strings.return_value = {"score": str(SCORE)}
    mock_load_evaluator.return_value = mock_evaluator

    evaluator = OpenAIEvaluator(
        question_asked=QUESTION_ASKED,
        true_answer=QUESTION_ANSWER,
        use_cache=False,
    )

    evaluator.evaluate_response("Something")
    evaluator.evaluate_response("Something")

    # With caching disabled, evaluate_strings should be called twice
    assert mock_evaluator.evaluate_strings.call_count == 2

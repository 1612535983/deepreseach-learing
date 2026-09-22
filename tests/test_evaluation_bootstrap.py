from deepresearch.evaluation.bootstrap import get_evaluation_provider
from deepresearch.evaluation.config import EvaluationConfig
from deepresearch.evaluation.jev import JevDecisionProvider


def test_disabled_evaluation_does_not_construct_provider() -> None:
    assert get_evaluation_provider(EvaluationConfig()) is None


def test_jev_config_constructs_jev_provider() -> None:
    provider = get_evaluation_provider(
        EvaluationConfig(use="jev", api_key="test-secret")
    )
    assert isinstance(provider, JevDecisionProvider)

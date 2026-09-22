"""Failures surfaced by probabilistic decision providers."""


class EvaluationProviderError(RuntimeError):
    """The remote provider could not return a usable decision batch."""


class EvaluationResponseError(EvaluationProviderError):
    """The provider response violated the expected typed contract."""

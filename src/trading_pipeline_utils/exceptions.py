class PipelineError(Exception):
    """Base for all pipeline errors."""


class ConfigError(PipelineError):
    """Config loading or validation errors."""


class DataFetchError(PipelineError):
    """External data fetch failures."""


class ValidationError(PipelineError):
    """Data or output validation failures."""


class ModelError(PipelineError):
    """Model load or inference failures."""


class ArtifactMissingError(PipelineError):
    """Required upstream artifact not found."""

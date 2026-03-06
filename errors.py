class PluginError(Exception):
    """Base plugin error."""


class ExternalServiceError(PluginError):
    """Raised when an upstream service cannot be reached."""


class InvalidQueryError(PluginError):
    """Raised when a user query is invalid."""


class InvalidUrlError(PluginError):
    """Raised when a URL is invalid."""

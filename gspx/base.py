"""Base classes."""

from abc import abstractmethod, ABC


class Signal(ABC):
    """Base class for graph signals."""

    @abstractmethod
    def to_rgba(self, **kwargs):
        """Get an RGBA representation of the signal."""
        raise NotImplementedError('Abstract method not implemented.')

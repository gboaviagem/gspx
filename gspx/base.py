"""Base classes."""

from abc import abstractmethod, ABC


class Signal(ABC):
    """Base class for graph signals."""

    @abstractmethod
    def to_rgb(self, **kwargs):
        """Get an RGB representation of the signal."""
        raise NotImplementedError('Abstract method not implemented.')

    @abstractmethod
    def to_array(self, **kwargs):
        """Create a pure-numpy array representation."""
        raise NotImplementedError('Abstract method not implemented.')

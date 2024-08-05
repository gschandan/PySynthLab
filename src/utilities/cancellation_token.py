import threading
from contextlib import contextmanager


class CancellationError(Exception):
    pass


class GlobalCancellationToken:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._is_cancelled = threading.Event()
        return cls._instance

    def cancel(self):
        self._is_cancelled.set()

    @property
    def is_cancelled(self):
        return self._is_cancelled.is_set()

    @staticmethod
    def check_cancellation():
        if GlobalCancellationToken().is_cancelled:
            raise CancellationError("Operation was cancelled")

    @staticmethod
    @contextmanager
    def cancellable():
        try:
            yield
        finally:
            GlobalCancellationToken.check_cancellation()

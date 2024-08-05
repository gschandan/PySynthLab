import unittest
import threading
import time
from src.utilities.cancellation_token import GlobalCancellationToken, CancellationError

@unittest.skip("Global cancellation token")
class TestGlobalCancellationToken(unittest.TestCase):

    def setUp(self):
        GlobalCancellationToken._instance = None

    def test_cancel(self):
        token = GlobalCancellationToken()
        self.assertFalse(token.is_cancelled)
        token.cancel()
        self.assertTrue(token.is_cancelled)

    def test_is_cancelled(self):
        token = GlobalCancellationToken()
        self.assertFalse(token.is_cancelled)
        token.cancel()
        self.assertTrue(token.is_cancelled)

        another_token = GlobalCancellationToken()
        self.assertTrue(another_token.is_cancelled)

    def test_check_cancellation(self):
        token = GlobalCancellationToken()

        try:
            GlobalCancellationToken.check_cancellation()
        except CancellationError:
            self.fail("check_cancellation() raised CancellationError")

        token.cancel()

        with self.assertRaises(CancellationError):
            GlobalCancellationToken.check_cancellation()

    def test_cancellable(self):
        token = GlobalCancellationToken()

        with GlobalCancellationToken.cancellable():
            pass

        token.cancel()

        with self.assertRaises(CancellationError):
            with GlobalCancellationToken.cancellable():
                pass

    def test_cancellation_in_thread(self):
        def target():
            try:
                while True:
                    with GlobalCancellationToken.cancellable():
                        time.sleep(0.1)
            except CancellationError:
                return "Cancelled"

        thread = threading.Thread(target=target)
        thread.start()

        time.sleep(0.3)
        GlobalCancellationToken().cancel()

        thread.join(timeout=1)
        self.assertFalse(thread.is_alive())

    def test_behaviour(self):
        token1 = GlobalCancellationToken()
        token2 = GlobalCancellationToken()
        self.assertIs(token1, token2)

        token1.cancel()
        self.assertTrue(token2.is_cancelled)

if __name__ == '__main__':
    unittest.main()
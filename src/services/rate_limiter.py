import threading
import time
from src.config import TPM_LIMIT, RPM_LIMIT

class RateLimiter:
    """A thread-safe class to manage TPM and RPM limits using a fixed window."""
    def __init__(self, tpm_limit: int, rpm_limit: int):
        self.tpm_limit = tpm_limit
        self.rpm_limit = rpm_limit
        self.lock = threading.Lock()
        self.tokens_in_window = 0
        self.requests_in_window = 0
        self.window_start_time = time.time()

    def acquire(self, num_tokens: int):
        with self.lock:
            now = time.time()
            elapsed = now - self.window_start_time

            if elapsed > 60:
                # New window
                self.tokens_in_window = 0
                self.requests_in_window = 0
                self.window_start_time = now

            if self.requests_in_window + 1 > self.rpm_limit or self.tokens_in_window + num_tokens > self.tpm_limit:
                wait_time = 60 - elapsed
                if wait_time > 0:
                    print(f"RPM/TPM limit reached. Waiting for {wait_time:.2f} seconds...")
                    time.sleep(wait_time)
                # After waiting, we are in a new window
                self.tokens_in_window = 0
                self.requests_in_window = 0
                self.window_start_time = time.time()
            
            self.requests_in_window += 1
            self.tokens_in_window += num_tokens

# Global rate limiter instance
rate_limiter = RateLimiter(tpm_limit=TPM_LIMIT, rpm_limit=RPM_LIMIT)


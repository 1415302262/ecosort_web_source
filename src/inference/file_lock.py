"""Exclusive process lock; closing the file releases it on Linux and Windows."""

import os
import time

try:
    from fcntl import LOCK_EX, flock
except ImportError:
    import msvcrt

    LOCK_EX = 2

    def flock(fd, operation):
        if operation != LOCK_EX:
            raise ValueError("Only exclusive locks are supported")
        deadline = time.monotonic() + 60
        while True:
            try:
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                return
            except OSError:
                if time.monotonic() >= deadline:
                    raise TimeoutError("File lock timed out") from None
                time.sleep(0.02)

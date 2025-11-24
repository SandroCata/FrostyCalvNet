import io
import os
import sys
from contextlib import contextmanager


class _Tee(io.TextIOBase):
    """A simple tee that writes to two text streams."""

    def __init__(self, primary, secondary):
        self.primary = primary
        self.secondary = secondary

    def write(self, data):
        self.primary.write(data)
        self.secondary.write(data)
        return len(data)

    def flush(self):
        self.primary.flush()
        self.secondary.flush()


@contextmanager
def tee_stdout_stderr(log_path):
    """Redirect stdout and stderr to both the terminal and ``log_path``.

    The log file is opened in append mode and the parent directory is created if
    necessary.
    """

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a", buffering=1) as log_file:
        stdout_tee = _Tee(sys.__stdout__, log_file)
        stderr_tee = _Tee(sys.__stderr__, log_file)
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = stdout_tee, stderr_tee
        try:
            yield log_file
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            sys.stdout, sys.stderr = old_stdout, old_stderr
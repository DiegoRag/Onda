"""Simulated transmission channels (AWGN for now). Test-only — never wired
into the production transmit/receive pipeline.
"""

from core.channel.awgn import AWGNChannel

__all__ = ["AWGNChannel"]

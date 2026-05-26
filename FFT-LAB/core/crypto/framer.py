"""Binary framing: serializes the metadata that travels alongside ciphertext.

A "frame" in this project is a single self-contained transmission unit:

    +------------------+--------------+------------+---------------------+----------------+
    | nonce (16 bytes) | sample_rate  | num_samples| ciphertext_length   | ciphertext     |
    |                  | uint32 LE    | uint32 LE  | uint32 LE           | N bytes        |
    +------------------+--------------+------------+---------------------+----------------+
    |<------------------ HEADER (28 bytes) --------------------------->|

The header carries everything the receiver needs to reconstruct the original
audio:
    - `nonce`             — pairs with the password-derived AES key.
    - `sample_rate`       — the voice sample rate (e.g., 16000 Hz).
    - `num_samples`       — exact original sample count (so trailing
                            zero-padding from OFDM framing can be trimmed).
    - `ciphertext_length` — bytes of AES output to read after the header.

Everything is little-endian (struct format '<16sIII') for cross-platform
predictability; modern x86/ARM are LE natively.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import global_configs


@dataclass(frozen=True, slots=True)
class FrameHeader:
    """Immutable header value-object.

    `frozen=True` prevents accidental mutation after construction; `slots=True`
    keeps the memory layout tight (no per-instance __dict__).
    """

    nonce: bytes              # exactly AES_NONCE_SIZE_BYTES bytes
    sample_rate: int          # source audio sample rate (Hz)
    num_samples: int          # exact original sample count
    ciphertext_length: int    # bytes of AES output after the header

    def __post_init__(self) -> None:
        # Validate at construction time so a corrupt header can never travel
        # silently. We hit the underlying frozen-dataclass dict directly to
        # avoid AttributeError from the frozen setter; no mutation happens.
        if len(self.nonce) != global_configs.AES_NONCE_SIZE_BYTES:
            raise ValueError(
                f"nonce must be {global_configs.AES_NONCE_SIZE_BYTES} bytes, "
                f"got {len(self.nonce)}."
            )
        for field_name in ("sample_rate", "num_samples", "ciphertext_length"):
            value = getattr(self, field_name)
            if value < 0 or value > 0xFFFF_FFFF:
                raise ValueError(
                    f"{field_name}={value} does not fit in uint32."
                )


class Framer:
    """Builds and parses the 28-byte header that prefixes every transmission.

    Stateless utility — could be free functions, but a class lets us bind the
    struct format and header size as attributes and makes the API discoverable
    via autocomplete.
    """

    # struct format: 16-byte nonce, then three little-endian uint32.
    _STRUCT_FORMAT: str = "<16sIII"

    HEADER_SIZE_BYTES: int = global_configs.HEADER_SIZE_BYTES

    def __init__(self) -> None:
        # struct.Struct is precompiled — slightly faster than calling
        # struct.pack/unpack repeatedly.
        self._packer: struct.Struct = struct.Struct(self._STRUCT_FORMAT)
        assert self._packer.size == self.HEADER_SIZE_BYTES, (
            f"struct format {self._STRUCT_FORMAT!r} yields "
            f"{self._packer.size} bytes; expected {self.HEADER_SIZE_BYTES}."
        )

    # ------------------------------------------------------------------
    # Build / parse
    # ------------------------------------------------------------------
    def build(self, header: FrameHeader, ciphertext: bytes) -> bytes:
        """Concatenate header + ciphertext into the wire-format frame."""
        if header.ciphertext_length != len(ciphertext):
            raise ValueError(
                f"header.ciphertext_length={header.ciphertext_length} does "
                f"not match len(ciphertext)={len(ciphertext)}."
            )
        header_bytes = self._packer.pack(
            header.nonce,
            header.sample_rate,
            header.num_samples,
            header.ciphertext_length,
        )
        return header_bytes + ciphertext

    def parse_header(self, payload_prefix: bytes) -> FrameHeader:
        """Decode the first HEADER_SIZE_BYTES bytes of a frame into a header.

        Caller must supply at least HEADER_SIZE_BYTES bytes (extra is ignored).
        """
        if len(payload_prefix) < self.HEADER_SIZE_BYTES:
            raise ValueError(
                f"need at least {self.HEADER_SIZE_BYTES} bytes to parse "
                f"header, got {len(payload_prefix)}."
            )
        nonce, sample_rate, num_samples, ct_len = self._packer.unpack(
            payload_prefix[: self.HEADER_SIZE_BYTES]
        )
        return FrameHeader(
            nonce=nonce,
            sample_rate=sample_rate,
            num_samples=num_samples,
            ciphertext_length=ct_len,
        )

    def extract_ciphertext(self, payload: bytes, ciphertext_length: int) -> bytes:
        """Slice the ciphertext out of a full frame, given the declared length.

        Truncates anything past `ciphertext_length` (OFDM framing may pad).
        """
        start = self.HEADER_SIZE_BYTES
        end = start + ciphertext_length
        if len(payload) < end:
            raise ValueError(
                f"payload too short: need {end} bytes "
                f"({self.HEADER_SIZE_BYTES} header + {ciphertext_length} "
                f"ciphertext), got {len(payload)}."
            )
        return payload[start:end]

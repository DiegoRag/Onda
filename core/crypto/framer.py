"""Empacotamento binário: serializa os metadados que viajam junto com o ciphertext.

Um "frame" neste projeto é uma unidade de transmissão autocontida:

    +------------------+--------------+------------+---------------------+----------------+
    | nonce (16 bytes) | sample_rate  | num_samples| ciphertext_length   | ciphertext     |
    |                  | uint32 LE    | uint32 LE  | uint32 LE           | N bytes        |
    +------------------+--------------+------------+---------------------+----------------+
    |<------------------ CABEÇALHO (28 bytes) ------------------------>|

O cabeçalho carrega tudo que o receptor precisa para reconstruir o áudio original:
    - `nonce`             — combina com a chave AES derivada da senha.
    - `sample_rate`       — a taxa de amostragem da voz (ex.: 16000 Hz).
    - `num_samples`       — contagem exata de amostras originais (para o padding de
                            zeros do frame OFDM poder ser cortado).
    - `ciphertext_length` — bytes de saída do AES a ler depois do cabeçalho.

Tudo é little-endian (formato struct '<16sIII') por previsibilidade entre plataformas;
x86/ARM modernos são LE nativamente.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import global_configs


@dataclass(frozen=True, slots=True)
class FrameHeader:
    """Objeto-valor imutável do cabeçalho.

    `frozen=True` impede mutação acidental após a construção; `slots=True` mantém o
    layout de memória enxuto (sem __dict__ por instância).
    """

    nonce: bytes              # exatamente AES_NONCE_SIZE_BYTES bytes
    sample_rate: int          # taxa de amostragem da fonte (Hz)
    num_samples: int          # contagem exata de amostras originais
    ciphertext_length: int    # bytes de saída do AES depois do cabeçalho

    def __post_init__(self) -> None:
        # Valida no momento da construção para um cabeçalho corrompido nunca viajar
        # silenciosamente. Só leitura aqui (getattr), nenhuma mutação acontece —
        # por isso não conflita com o dataclass frozen.
        if len(self.nonce) != global_configs.AES_NONCE_SIZE_BYTES:
            raise ValueError(
                f"nonce must be {global_configs.AES_NONCE_SIZE_BYTES} bytes, "
                f"got {len(self.nonce)}."
            )
        # Os três inteiros precisam caber em uint32 (4 bytes) para o struct empacotar.
        for field_name in ("sample_rate", "num_samples", "ciphertext_length"):
            value = getattr(self, field_name)
            if value < 0 or value > 0xFFFF_FFFF:
                raise ValueError(
                    f"{field_name}={value} does not fit in uint32."
                )


class Framer:
    """Constrói e analisa o cabeçalho de 28 bytes que prefixa cada transmissão.

    Utilitário sem estado — poderiam ser funções soltas, mas uma classe deixa amarrar
    o formato do struct e o tamanho do cabeçalho como atributos e torna a API
    descobrível pelo autocomplete.
    """

    # formato struct: nonce de 16 bytes, depois três uint32 little-endian.
    _STRUCT_FORMAT: str = "<16sIII"

    HEADER_SIZE_BYTES: int = global_configs.HEADER_SIZE_BYTES

    def __init__(self) -> None:
        # struct.Struct é pré-compilado — um pouco mais rápido que chamar
        # struct.pack/unpack repetidamente.
        self._packer: struct.Struct = struct.Struct(self._STRUCT_FORMAT)
        # Sanidade: o formato precisa render exatamente HEADER_SIZE_BYTES (28).
        assert self._packer.size == self.HEADER_SIZE_BYTES, (
            f"struct format {self._STRUCT_FORMAT!r} yields "
            f"{self._packer.size} bytes; expected {self.HEADER_SIZE_BYTES}."
        )

    # ------------------------------------------------------------------
    # Construir / analisar
    # ------------------------------------------------------------------
    def build(self, header: FrameHeader, ciphertext: bytes) -> bytes:
        """Concatena cabeçalho + ciphertext no frame em formato de transmissão."""
        # O tamanho declarado no cabeçalho precisa bater com o ciphertext real.
        if header.ciphertext_length != len(ciphertext):
            raise ValueError(
                f"header.ciphertext_length={header.ciphertext_length} does "
                f"not match len(ciphertext)={len(ciphertext)}."
            )
        # Empacota os 4 campos do cabeçalho e cola o ciphertext em seguida.
        header_bytes = self._packer.pack(
            header.nonce,
            header.sample_rate,
            header.num_samples,
            header.ciphertext_length,
        )
        return header_bytes + ciphertext

    def parse_header(self, payload_prefix: bytes) -> FrameHeader:
        """Decodifica os primeiros HEADER_SIZE_BYTES de um frame em um cabeçalho.

        O chamador precisa fornecer ao menos HEADER_SIZE_BYTES bytes (o excesso é
        ignorado).
        """
        # Sem bytes suficientes não dá para ler o cabeçalho inteiro.
        if len(payload_prefix) < self.HEADER_SIZE_BYTES:
            raise ValueError(
                f"need at least {self.HEADER_SIZE_BYTES} bytes to parse "
                f"header, got {len(payload_prefix)}."
            )
        # Desempacota os 4 campos a partir dos primeiros 28 bytes.
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
        """Recorta o ciphertext de um frame completo, dado o tamanho declarado.

        Trunca qualquer coisa além de `ciphertext_length` (o frame OFDM pode ter
        padding).
        """
        # O ciphertext começa logo após o cabeçalho e tem ciphertext_length bytes.
        start = self.HEADER_SIZE_BYTES
        end = start + ciphertext_length
        if len(payload) < end:
            raise ValueError(
                f"payload too short: need {end} bytes "
                f"({self.HEADER_SIZE_BYTES} header + {ciphertext_length} "
                f"ciphertext), got {len(payload)}."
            )
        return payload[start:end]

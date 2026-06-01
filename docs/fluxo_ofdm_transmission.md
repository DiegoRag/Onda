# Fluxo exato da aba "OFDM Transmissão" (FFT-LAB)

> Documento de referência para explicação. Descreve, passo a passo, o caminho de
> execução das funções quando o usuário interage com a aba de transmissão/recepção
> OFDM de voz. A aba grava voz, **criptografa com AES-256-CTR**, modula os bytes
> cifrados em **OFDM usando uma FFT/IFFT implementada manualmente** (sem biblioteca),
> e transmite como áudio (WAV ou alto-falante). A recepção faz o caminho inverso.

## Visão geral da pipeline

```
TX: voz (int16) -> bytes -> AES-256-CTR -> framing -> embaralhamento ->
    OFDM (IFFT manual) -> + chirp de sincronização -> WAV/alto-falante @ 48 kHz

RX: WAV/microfone @ 48 kHz -> detectar chirp -> OFDM (FFT manual) ->
    desembaralhar -> remontar frame -> AES decrypt -> voz (int16)
```

Duas camadas independentes:
- **AES** fornece a confidencialidade real (depende da chave/senha).
- **Fourier (IFFT/FFT)** é só o transporte: transforma bytes em som e vice-versa.

## Mapa de arquivos

| Camada | Arquivo | Papel |
|---|---|---|
| UI da aba | `ui/views/ofdm_transmission_view.py` | botões, campos, threads, espectrograma |
| Orquestração TX | `core/pipeline/transmitter.py` | classe `Transmitter` |
| Orquestração RX | `core/pipeline/receiver.py` | classe `Receiver` |
| Cifra | `core/crypto/cipher.py` | `AESCipher` (AES-256-CTR) |
| Framing | `core/crypto/framer.py` | `FrameHeader`, `Framer` |
| Modulação | `core/modulation/ofdm.py` | `OFDMModem` (modula/demodula) |
| Mapeamento bits↔símbolos | `core/modulation/qpsk.py` | `QPSKModem` (QPSK) |
| Embaralhamento | `core/modulation/scrambler.py` | `Scrambler` (permutação por chave) |
| Sincronização | `core/modulation/sync.py` | `PreambleSync` (chirp + correlação) |
| **FFT manual** | `core/modulation/fft_from_scratch.py` | `fft_iterative`, `ifft_iterative` |
| Áudio | `core/audio/recorder.py`, `player.py`, `wav_io.py` | gravar, tocar, ler/escrever WAV |
| Constantes | `global_configs.py` | parâmetros (banda, OFDM, AES, etc.) |

## Parâmetros relevantes (de `global_configs.py`)

- `FS = 48000` — taxa do sinal transmitido.
- `F_MIN = 6000`, `F_MAX = 10000` — banda audível das subportadoras.
- `N_FFT = 256`, `N_CP = 64` → frame OFDM = **320 amostras**.
- `PILOT_BIN = 32`; `DATA_BINS = 33..53` (21 subportadoras de dados).
- `BITS_PER_OFDM_FRAME = 42` (21 símbolos QPSK × 2 bits).
- `AES_KEY_SIZE_BYTES = 32`, `AES_NONCE_SIZE_BYTES = 16`, `HEADER_SIZE_BYTES = 28`.

---

## 1) TRANSMISSÃO

### 1.1 Gatilhos na UI
- Botão "💾 Salvar WAV de transmissão" → `_on_tx_to_wav`
- Botão "📢 Transmitir via alto-falante" → `_on_tx_to_speaker`

Ambos chamam o núcleo comum `_build_tx_signal()`.

### 1.2 `_build_tx_signal()` (na view)
```
_build_tx_signal
  ├─ _tx_password()                         # lê/valida a senha de transmissão
  ├─ Transmitter(password)                  # constrói a pipeline (ver 1.3)
  └─ transmitter.build_signal(voz, taxa_voz)  # roda a pipeline (ver 1.4)
       -> retorna (signal_48k, TransmissionResult)
```

### 1.3 `Transmitter.__init__(password)`
```
AESCipher.from_password(password)            # chave = SHA-256(senha) -> 32 bytes
Framer()
OFDMModem()                                  # usa QPSKModem internamente
Scrambler()
PreambleSync()                               # gera o chirp uma vez (cache)
AudioPlayer()
permutation = Scrambler.permutation_for_password(password)
   ├─ derive_seed(password)  = SHA-256(senha + b"scramble-v1")[:8]
   └─ build_permutation(seed) = np.random.default_rng(seed).permutation(21)
```

### 1.4 `Transmitter.build_signal(voice_samples, voice_sample_rate)`
```
Estágio 1 — Serialização:
   WavIO.int16_to_bytes(voice_samples)  -> plaintext (bytes, little-endian)

Estágio 2 — Criptografia:
   AESCipher.encrypt(plaintext)         -> (nonce, ciphertext)
   # CTR: ciphertext tem o mesmo tamanho do plaintext; nonce aleatório por chamada

Estágio 3 — Framing:
   header = FrameHeader(nonce, sample_rate, num_samples, ciphertext_length)
   payload = Framer.build(header, ciphertext)   # 28 bytes de cabeçalho + ciphertext

Estágio 4 — Modulação OFDM:
   ofdm_signal = OFDMModem.bytes_to_signal(payload, permutation)   # ver 1.5

Estágio 5 — Montagem final:
   full = [silêncio] + PreambleSync.preamble + ofdm_signal + [silêncio]
   # PreambleSync.preamble = chirp linear 6->10 kHz, 50 ms, com rampas de Hann
```

### 1.5 `OFDMModem.bytes_to_signal(payload, permutation)`
```
bits = np.unpackbits(payload)                 # bytes -> bits (MSB primeiro)
padding até múltiplo de 42 bits (1 frame)
para cada frame (42 bits):
   symbols = QPSKModem.modulate(42 bits)      # -> 21 símbolos complexos
   frame   = OFDMModem.modulate_frame(symbols, permutation)   # ver 1.6
   concatena frame (320 amostras) no sinal
```

### 1.6 `OFDMModem.modulate_frame(symbols, permutation)` — onde entra a IFFT manual
```
1. spectrum = zeros(256) (complexo)
2. spectrum[PILOT_BIN] = 1+0j               # piloto conhecido (referência)
3. coloca os 21 símbolos embaralhados nos DATA_BINS (via permutation)
4. simetria conjugada: spectrum[N-k] = conj(spectrum[k])   # garante saída real
5. time = ifft_iterative(spectrum)          # <<< IFFT MANUAL (sem np.fft) >>>
6. real = parte real de time
7. prefixo cíclico: copia as últimas 64 amostras pro início -> 320 amostras
```

### 1.7 Saídas
```
_on_tx_to_wav:    WavIO.write(path, full, 48000)  + _plot_spectrogram(full)
_on_tx_to_speaker: (thread) AudioPlayer.play(full, 48000)  + _plot_spectrogram(full)
```

---

## 2) RECEPÇÃO

### 2.1 Gatilhos na UI
- Botão "🎤 Receber via microfone" → `_on_rx_from_mic`
- Botão "📂 Receber de WAV" → `_on_rx_from_wav`

Ambos leem a senha de recepção (`_rx_password()`) e rodam numa thread.

### 2.2 Fontes
```
Receiver(password)                            # mesmos componentes do Transmitter,
                                              # mesma permutation derivada da senha
from_microphone(capture_s):
   AudioRecorder.record(capture_s) @ 48 kHz -> decode_signal(sinal)
from_wav(path):
   WavIO.read(path) -> decode_signal(sinal)
```

### 2.3 `Receiver.decode_signal(signal)`
```
Estágio 1 — Sincronização:
   ofdm_start = PreambleSync.detect(signal)
   # correlação cruzada do sinal com o chirp conhecido; pico = início dos dados
   # (se não achar o chirp -> ReceptionError)

Estágio 2 — Lê o cabeçalho:
   header_bytes = OFDMModem.signal_to_bytes(parte_inicial, 28, permutation)  # ver 2.4
   header = Framer.parse_header(header_bytes)   # nonce, sample_rate, num_samples, ct_len

Estágio 3 — Demodula o payload completo:
   total = 28 + header.ciphertext_length
   payload_bytes = OFDMModem.signal_to_bytes(sinal, total, permutation)

Estágio 4 — Decriptação:
   ciphertext = Framer.extract_ciphertext(payload_bytes, header.ciphertext_length)
   plaintext  = AESCipher.decrypt(header.nonce, ciphertext)
   # senha errada -> lixo silencioso (CTR não verifica integridade)

Estágio 5 — Remontagem:
   samples = WavIO.bytes_to_int16(plaintext)[: header.num_samples]   # voz recuperada
```

### 2.4 `OFDMModem.signal_to_bytes(...)` e `demodulate_frame` — onde entra a FFT manual
```
para cada frame (320 amostras):
   useful   = frame[64:]                      # descarta prefixo cíclico
   spectrum = fft_iterative(useful)           # <<< FFT MANUAL (sem np.fft) >>>
   H        = spectrum[PILOT_BIN] / (1+0j)    # estima o canal pelo piloto
   eq       = spectrum[DATA_BINS] / H         # equaliza
   symbols  = desembaralha(eq, permutation)
   bits     = QPSKModem.demodulate(symbols)   # 21 símbolos -> 42 bits
junta os bits -> np.packbits -> bytes (truncado para o tamanho pedido)
```

### 2.5 Saídas
```
_on_play_recovered:  AudioPlayer.play(amostras_recuperadas, taxa_da_voz)
_on_save_recovered:  WavIO.write(path, amostras, taxa_da_voz)
```

---

## 3) A FFT manual (`core/modulation/fft_from_scratch.py`)

A pipeline de voz usa **estas funções**, não `numpy.fft`:

```
fft_iterative(x)            # Cooley-Tukey radix-2, iterativa (N deve ser potência de 2)
  1. _bit_reverse_indices(N): reordena a entrada em ordem bit-reversa
  2. log2(N) estágios de "borboletas":
        para size = 2, 4, ..., N:
           w_step = exp(-2j*pi/size)
           combina pares: X[i]      = par + w*ímpar
                          X[i+size/2]= par - w*ímpar
  -> O(N log N)

ifft_iterative(X)           # inverso pelo truque do conjugado:
  return conj( fft_iterative( conj(X) ) ) / N
```

O módulo também tem `dft_naive`/`idft_naive` (definição direta O(N²)) e
`fft_recursive`/`ifft_recursive`, além de `verify_against_numpy()` que prova que
as implementações batem com o `numpy.fft` (diferença ~1e-12).

---

## 4) Resumo de "onde Fourier aparece"

| Etapa | Função | Motor de Fourier |
|---|---|---|
| Modulação (síntese do sinal) | `OFDMModem.modulate_frame` | **`ifft_iterative` (manual)** |
| Demodulação (análise do sinal) | `OFDMModem.demodulate_frame` | **`fft_iterative` (manual)** |
| Sincronização (achar o chirp) | `PreambleSync.detect` | correlação cruzada (ainda via biblioteca) |
| Espectrograma (visualização) | `_plot_spectrogram` | `scipy.signal.spectrogram` (biblioteca) |

**Nota:** a pipeline de voz (TX/RX) é 100% FFT manual. A sincronização e o
espectrograma ainda usam FFT de biblioteca (são candidatos a conversão futura).
```

"""Fourier Transform implementations — from scratch, no FFT library.

This module exists for the academic report: it demonstrates that we understand
the Transformada de Fourier from first principles, not just as a black-box
library call. Four implementations are provided, in order of growing
sophistication:

    1. dft_naive(x)       — direct application of the DFT formula. O(N^2).
                            Pure mathematics, no algorithmic cleverness.

    2. idft_naive(X)      — direct application of the inverse formula.

    3. fft_recursive(x)   — Cooley-Tukey radix-2 by divide-and-conquer.
                            O(N log N). Pure Python + stdlib `cmath`.

    4. fft_iterative(x)   — same algorithm, in iterative form with explicit
                            bit-reversal. Used in real implementations
                            (this is what numpy/scipy do internally, in C).

All functions accept any iterable of numbers (int, float, complex). They
return Python lists of complex numbers — no numpy required for the math
itself. A numpy-array wrapper at the bottom (`fft_from_scratch`) is provided
for convenience so the result can drop into the rest of the codebase.

The only dependency is `cmath` (stdlib): we need `cmath.exp` to compute the
complex exponential e^{j*theta}, which IS the definition of Fourier — there
is no way around it.

==========================================================================
The Discrete Fourier Transform (DFT) — definition
==========================================================================

    Forward:   X[k] = SUM_{n=0..N-1}  x[n] * exp(-2*pi*j * k * n / N)

    Inverse:   x[n] = (1/N) * SUM_{k=0..N-1}  X[k] * exp(+2*pi*j * k * n / N)

Properties used in this project:

    - Linearity:        DFT(a*x + b*y) = a*DFT(x) + b*DFT(y)
    - Real signal:      x[n] real <=> X[N-k] = conj(X[k])   (conjugate symmetry)
    - Time shift:       x[n-m] <=> X[k] * exp(-2*pi*j * k * m / N)
    - Parseval:         sum |x[n]|^2 = (1/N) * sum |X[k]|^2

==========================================================================
Cooley-Tukey radix-2 (the trick that makes FFT O(N log N))
==========================================================================

Insight: split x into even-indexed and odd-indexed elements.

    X[k] = SUM_n x[n] * w^{kn},          w = exp(-2*pi*j/N)

         = SUM_{m} x[2m]   * w^{k*2m}
         + SUM_{m} x[2m+1] * w^{k*(2m+1)}

         = E[k]            + w^k * O[k]

where E and O are DFTs of half size. The trick: since w^{N/2} = -1,

    X[k + N/2] = E[k] - w^k * O[k]

So one N-point DFT becomes two (N/2)-point DFTs plus N/2 "butterfly"
combinations. Recurrence T(N) = 2*T(N/2) + O(N) solves to O(N log N).
"""

from __future__ import annotations

import cmath
import math
from typing import Sequence


# =============================================================================
# 1) Direct DFT  —  pure formula, O(N^2)
# =============================================================================

def dft_naive(x: Sequence[complex]) -> list[complex]:
    """Direct computation of X[k] = sum_n x[n] * exp(-2*pi*j*k*n/N).

    This is literally the definition transliterated into code. For each output
    bin k, we walk through the input and accumulate the rotating-phasor sum.

    Complexity: O(N^2). For N=256 this is 65 536 complex multiplications.
    For N=4096 it is 16 million — at which point you switch to FFT.

    Parameters
    ----------
    x : sequence of numbers (int, float, or complex)

    Returns
    -------
    list[complex] of length N
    """
    N = len(x)
    X: list[complex] = [0j] * N

    for k in range(N):
        accumulator = 0j
        for n in range(N):
            # angle = -2*pi*k*n/N  (radians, clockwise on the unit circle)
            angle = -2.0 * math.pi * k * n / N
            accumulator += x[n] * cmath.exp(1j * angle)
        X[k] = accumulator

    return X


def idft_naive(X: Sequence[complex]) -> list[complex]:
    """Direct computation of x[n] = (1/N) * sum_k X[k] * exp(+2*pi*j*k*n/N).

    Two differences from the forward DFT:
      1. Opposite sign in the exponential (+, not -).
      2. Division by N at the end (the "1/N" in front of the sum).
    """
    N = len(X)
    x: list[complex] = [0j] * N

    for n in range(N):
        accumulator = 0j
        for k in range(N):
            angle = +2.0 * math.pi * k * n / N
            accumulator += X[k] * cmath.exp(1j * angle)
        x[n] = accumulator / N

    return x


# =============================================================================
# 2) Recursive FFT (Cooley-Tukey radix-2)  —  O(N log N), divide and conquer
# =============================================================================

def fft_recursive(x: Sequence[complex]) -> list[complex]:
    """Radix-2 Cooley-Tukey FFT by recursion.

    N must be a power of 2 (1, 2, 4, 8, 16, ..., 256, ...).

    Algorithm:
        1. Base case: a 1-point DFT is the value itself.
        2. Split x into x_even (indices 0, 2, 4, ...) and x_odd (1, 3, 5, ...).
        3. Recurse on each half -> E and O of length N/2.
        4. Combine via butterfly:
              X[k]       = E[k] + w^k * O[k]
              X[k + N/2] = E[k] - w^k * O[k]
           where w = exp(-2*pi*j / N) is the principal Nth root of unity.

    Why it works: w^{N/2} = exp(-pi*j) = -1, so the second half of X uses
    the same products with a flipped sign.
    """
    x_list = [complex(v) for v in x]
    N = len(x_list)

    if N == 0:
        return []
    if N == 1:
        return x_list                              # 1-point DFT = identity
    if N & (N - 1) != 0:                           # bit trick: N is power of 2?
        raise ValueError(f"radix-2 FFT requires N power of 2, got N={N}.")

    # ---- Split ----
    even_part = fft_recursive(x_list[0::2])        # indices 0, 2, 4, ...
    odd_part = fft_recursive(x_list[1::2])         # indices 1, 3, 5, ...

    # ---- Butterfly combine ----
    half = N // 2
    X = [0j] * N
    for k in range(half):
        # Twiddle factor for this k.
        twiddle = cmath.exp(-2j * math.pi * k / N)
        product = twiddle * odd_part[k]
        X[k]        = even_part[k] + product
        X[k + half] = even_part[k] - product

    return X


def ifft_recursive(X: Sequence[complex]) -> list[complex]:
    """Inverse radix-2 FFT via the conjugate identity.

    Mathematical trick:
        IFFT(X) = conj( FFT( conj(X) ) ) / N

    Why this works:
        Forward: X[k] = sum_n x[n] * exp(-2*pi*j*k*n/N)
        Inverse: x[n] = (1/N) * sum_k X[k] * exp(+2*pi*j*k*n/N)

    If we conjugate X and feed it into the forward FFT, the +/- sign in the
    exponential gets absorbed by the conjugation, giving us the inverse
    formula (up to scaling by 1/N).

    This saves us from writing a second algorithm — same FFT code, just
    bracketed by `conj` on both sides and divided by N.
    """
    N = len(X)
    if N == 0:
        return []

    X_conj = [z.conjugate() for z in X]
    forward_of_conj = fft_recursive(X_conj)
    return [z.conjugate() / N for z in forward_of_conj]


# =============================================================================
# 3) Iterative FFT  —  bit-reversal + in-place butterflies
# =============================================================================
#
# Same Cooley-Tukey algorithm, but unrolled into loops. This is what numpy/
# scipy/MATLAB do under the hood. Faster than the recursive version because
# it avoids function-call overhead and works in-place (no list slicing).
# =============================================================================

def _bit_reverse_indices(n: int) -> list[int]:
    """Return [bit_reverse(i, log2(N)) for i in range(N)].

    Bit-reverse permutation is the standard FFT shuffle that maps the
    natural input order to the order required by the iterative butterflies.

    Example, N=8 (3 bits):
        i=0 (000) -> 000 = 0
        i=1 (001) -> 100 = 4
        i=2 (010) -> 010 = 2
        i=3 (011) -> 110 = 6
        i=4 (100) -> 001 = 1
        i=5 (101) -> 101 = 5
        i=6 (110) -> 011 = 3
        i=7 (111) -> 111 = 7
    """
    bits = int(math.log2(n))
    out = [0] * n
    for i in range(n):
        rev = 0
        v = i
        for _ in range(bits):
            rev = (rev << 1) | (v & 1)
            v >>= 1
        out[i] = rev
    return out


def fft_iterative(x: Sequence[complex]) -> list[complex]:
    """Iterative radix-2 FFT with bit-reversal. O(N log N), low constant factor.

    Stages:
        1. Permute x into bit-reversed order.
        2. For stage s in 1..log2(N):
              size = 2^s
              For each block of `size` consecutive samples:
                  For k in 0..size/2 - 1:
                      even = data[block + k]
                      odd  = data[block + k + size/2] * exp(-2*pi*j*k/size)
                      data[block + k]            = even + odd
                      data[block + k + size/2]   = even - odd

    Each stage doubles the size of the DFTs it produces, combining pairs of
    half-size DFTs into a full one via the butterfly. After log2(N) stages,
    the entire transform is done.
    """
    N = len(x)
    if N == 0:
        return []
    if N & (N - 1) != 0:
        raise ValueError(f"iterative radix-2 FFT requires N power of 2, got N={N}.")

    # ---- Step 1: bit-reversal permutation ----
    rev = _bit_reverse_indices(N)
    data: list[complex] = [complex(x[rev[i]]) for i in range(N)]

    # ---- Step 2: log2(N) stages of butterflies ----
    size = 2
    while size <= N:
        half = size // 2
        # All twiddles for this stage. w_step is the smallest rotation.
        w_step = cmath.exp(-2j * math.pi / size)

        for block_start in range(0, N, size):
            w = 1 + 0j
            for k in range(half):
                even_idx = block_start + k
                odd_idx = block_start + k + half

                even = data[even_idx]
                odd_w = data[odd_idx] * w

                data[even_idx] = even + odd_w
                data[odd_idx]  = even - odd_w

                w *= w_step
        size *= 2

    return data


def ifft_iterative(X: Sequence[complex]) -> list[complex]:
    """Iterative inverse FFT, via the conjugate identity."""
    N = len(X)
    if N == 0:
        return []
    X_conj = [z.conjugate() for z in X]
    forward = fft_iterative(X_conj)
    return [z.conjugate() / N for z in forward]


# =============================================================================
# 4) Verification — confirm our implementations agree with numpy.fft
# =============================================================================

def _max_complex_diff(a: Sequence[complex], b: Sequence[complex]) -> float:
    """Return the largest |a[i] - b[i]| element-wise."""
    return max(abs(complex(ai) - complex(bi)) for ai, bi in zip(a, b))


def verify_against_numpy(N: int = 256, tolerance: float = 1e-9) -> dict:
    """Run all four implementations against `numpy.fft` and report the largest
    numerical difference for each.

    A small (~1e-12) diff is expected due to floating-point rounding order.
    Anything larger than `tolerance` is a bug.

    Returns a dict mapping implementation name -> max-abs-diff against numpy.
    """
    import numpy as np

    # Random complex test input.
    rng = np.random.default_rng(42)
    x_np = rng.normal(size=N) + 1j * rng.normal(size=N)
    x_py = x_np.tolist()

    # Reference: numpy's FFT.
    X_ref = np.fft.fft(x_np)
    x_ref = np.fft.ifft(X_ref)

    results = {}

    # Forward — naive
    X_naive = dft_naive(x_py)
    results["dft_naive (forward)"] = _max_complex_diff(X_naive, X_ref.tolist())

    # Forward — recursive
    X_rec = fft_recursive(x_py)
    results["fft_recursive (forward)"] = _max_complex_diff(X_rec, X_ref.tolist())

    # Forward — iterative
    X_it = fft_iterative(x_py)
    results["fft_iterative (forward)"] = _max_complex_diff(X_it, X_ref.tolist())

    # Inverse — naive
    x_naive = idft_naive(X_ref.tolist())
    results["idft_naive (inverse)"] = _max_complex_diff(x_naive, x_ref.tolist())

    # Inverse — recursive
    x_rec = ifft_recursive(X_ref.tolist())
    results["ifft_recursive (inverse)"] = _max_complex_diff(x_rec, x_ref.tolist())

    # Inverse — iterative
    x_it = ifft_iterative(X_ref.tolist())
    results["ifft_iterative (inverse)"] = _max_complex_diff(x_it, x_ref.tolist())

    # Sanity: forward then inverse roundtrips.
    roundtrip_rec = ifft_recursive(fft_recursive(x_py))
    results["roundtrip (fft_recursive -> ifft_recursive)"] = (
        _max_complex_diff(roundtrip_rec, x_py)
    )

    roundtrip_it = ifft_iterative(fft_iterative(x_py))
    results["roundtrip (fft_iterative -> ifft_iterative)"] = (
        _max_complex_diff(roundtrip_it, x_py)
    )

    # Check tolerance and annotate.
    return {name: (diff, "PASS" if diff < tolerance else "FAIL")
            for name, diff in results.items()}


# =============================================================================
# 5) Self-test entry point  —  `python -m core.modulation.fft_from_scratch`
# =============================================================================

def _run_self_test() -> None:
    """Print a human-readable verification report."""
    print("=" * 72)
    print("FFT from-scratch — verification against numpy.fft (N=256, tol=1e-9)")
    print("=" * 72)
    results = verify_against_numpy(N=256)
    width = max(len(name) for name in results)
    for name, (diff, verdict) in results.items():
        print(f"  {name:<{width}}   diff={diff:.3e}   [{verdict}]")
    print("=" * 72)

    all_pass = all(v == "PASS" for _, v in results.values())
    print("OVERALL:", "ALL PASS" if all_pass else "SOMETHING FAILED")


if __name__ == "__main__":
    _run_self_test()

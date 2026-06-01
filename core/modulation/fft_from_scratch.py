"""Implementações da Transformada de Fourier — do zero, sem biblioteca de FFT.

PARA QUE SERVE (no mundo real, sem fórmulas):
    Uma onda no tempo (um som) é, na verdade, várias frequências tocando juntas. A
    Transformada de Fourier é a pergunta: "QUANTO de cada frequência existe nesse
    som, e com qual fase (adiantamento)?".

    O truque para responder: para testar UMA frequência, a gente "gira" o sinal
    contra uma seta de referência que roda naquela velocidade e SOMA tudo. Se o sinal
    realmente tem aquela frequência, os giros se alinham e a soma fica GRANDE; se não
    tem, os giros apontam para todo lado e se CANCELAM (soma perto de zero). Repetindo
    esse teste para todas as frequências, descobrimos a "receita" completa do som.

    - A seta de referência que roda é o "exponencial complexo" (`cmath.exp`). O ângulo
      define a velocidade do giro = a frequência sendo testada. Não tem como fugir
      dela: girar é a essência de Fourier.

    - O caminho INVERSO faz o oposto: dada a receita (quanto de cada frequência),
      remonta a onda no tempo somando todas as setas girando.

ESTE MÓDULO traz 4 versões, da mais simples (e lenta) à mais esperta (e rápida):

    1. dft_naive(x)     — testa cada frequência contra o sinal inteiro, uma por uma.
                          Matemática pura, sem esperteza. Lento: O(N^2).
    2. idft_naive(X)    — o caminho de volta, também direto.
    3. fft_recursive(x) — Cooley-Tukey: divide o sinal em pares/ímpares e reaproveita
                          contas repetidas. Bem mais rápido: O(N log N).
    4. fft_iterative(x) — o mesmo algoritmo, em forma de loops (é o que numpy/scipy
                          fazem por dentro, em C). É a versão que a pipeline usa.

A única dependência é `cmath` (biblioteca padrão), porque precisamos da seta que gira
(`cmath.exp`) — ela É a definição de Fourier.

A ESPERTEZA DO COOLEY-TUKEY (por que fica rápido):
    Separe o sinal em amostras de índice par e de índice ímpar. Acontece que a
    transformada do todo pode ser montada a partir das transformadas dessas duas
    metades, com uma combinação barata chamada "borboleta". Além disso, a segunda
    metade do resultado reaproveita as MESMAS contas da primeira, só trocando um
    sinal. Resolver duas metades + combinar é muito mais barato do que refazer tudo.
"""

from __future__ import annotations

# cmath: números complexos. Precisamos de cmath.exp = a "seta que gira".
import cmath
# math: pi e log2.
import math
# Tipagem: aceitamos qualquer sequência de números.
from typing import Sequence


# =============================================================================
# 1) DFT direta — fórmula pura, O(N^2)
# =============================================================================

def dft_naive(x: Sequence[complex]) -> list[complex]:
    """Calcula a receita de frequências testando uma frequência de cada vez.

    Para cada frequência k, percorre o sinal inteiro girando cada amostra contra a
    seta de referência de k e acumulando. É lento (O(N^2)): para N=256 são 65.536
    multiplicações complexas; para N=4096 já são 16 milhões (aí troca-se pela FFT).
    """
    # N = tamanho do sinal = número de frequências a testar.
    N = len(x)
    # X guardará a resposta: "quanto de cada frequência". Começa tudo em zero.
    X: list[complex] = [0j] * N

    # Para cada frequência k que queremos medir...
    for k in range(N):
        # ...começamos uma soma vazia (o acumulador de giros).
        accumulator = 0j
        # ...e percorremos cada amostra n do sinal.
        for n in range(N):
            # Ângulo de giro da seta de referência nesse instante. O sinal negativo
            # indica giro "no sentido horário"; a velocidade cresce com k e n.
            angle = -2.0 * math.pi * k * n / N
            # Gira a amostra contra a referência e soma. Se a frequência k existir no
            # sinal, esses termos se alinham e a soma cresce; senão, se cancelam.
            accumulator += x[n] * cmath.exp(1j * angle)
        # Terminada a varredura, o acumulador é "quanto de frequência k" havia.
        X[k] = accumulator

    # Devolve a receita completa (quanto + fase de cada frequência).
    return X


def idft_naive(X: Sequence[complex]) -> list[complex]:
    """Caminho de volta: a partir da receita, remonta a onda no tempo.

    Duas diferenças em relação à ida:
      1. A seta gira no sentido OPOSTO (ângulo com sinal +, não -).
      2. No fim divide-se por N (uma média), porque estamos somando a contribuição
         de todas as frequências para reconstruir cada instante.
    """
    # N = número de frequências = número de amostras a reconstruir.
    N = len(X)
    # x guardará a onda reconstruída no tempo. Começa em zero.
    x: list[complex] = [0j] * N

    # Para cada instante n da onda que queremos reconstruir...
    for n in range(N):
        # ...soma vazia.
        accumulator = 0j
        # ...percorremos cada frequência k da receita.
        for k in range(N):
            # Mesma seta giratória, mas no sentido contrário (sinal +).
            angle = +2.0 * math.pi * k * n / N
            # Soma a contribuição daquela frequência para este instante.
            accumulator += X[k] * cmath.exp(1j * angle)
        # Divide por N (a média) e guarda o valor reconstruído desse instante.
        x[n] = accumulator / N

    # Devolve a onda no tempo.
    return x


# =============================================================================
# 2) FFT recursiva (Cooley-Tukey radix-2) — O(N log N), dividir para conquistar
# =============================================================================

def fft_recursive(x: Sequence[complex]) -> list[complex]:
    """FFT pela receita do Cooley-Tukey, em forma recursiva.

    N precisa ser potência de 2 (1, 2, 4, ..., 256, ...). A ideia: quebrar o sinal
    em metades (pares e ímpares), resolver cada metade (que se quebra de novo, e de
    novo...), e juntar com a "borboleta".
    """
    # Garante que tudo é complexo (aceita int/float na entrada).
    x_list = [complex(v) for v in x]
    # Tamanho atual do pedaço.
    N = len(x_list)

    # Caso vazio: nada a fazer.
    if N == 0:
        return []
    # Caso base: um sinal de 1 amostra já é sua própria transformada.
    if N == 1:
        return x_list                              # 1-point DFT = identity
    # Truque de bit: "N & (N-1) == 0" só é verdade quando N é potência de 2.
    if N & (N - 1) != 0:                           # bit trick: N is power of 2?
        raise ValueError(f"radix-2 FFT requires N power of 2, got N={N}.")

    # Divide: resolve a FFT das amostras de índice par (0, 2, 4, ...)...
    even_part = fft_recursive(x_list[0::2])        # indices 0, 2, 4, ...
    # ...e das de índice ímpar (1, 3, 5, ...). (Cada chamada se subdivide sozinha.)
    odd_part = fft_recursive(x_list[1::2])         # indices 1, 3, 5, ...

    # Junta (a "borboleta"):
    # metade do tamanho — quantas combinações vamos fazer.
    half = N // 2
    # Espaço para o resultado completo.
    X = [0j] * N
    # Para cada posição k da primeira metade...
    for k in range(half):
        # "twiddle": a seta de ajuste de fase que casa par com ímpar nessa posição.
        twiddle = cmath.exp(-2j * math.pi * k / N)
        # Aplica o ajuste ao termo ímpar.
        product = twiddle * odd_part[k]
        # Primeira metade do resultado: par + ímpar-ajustado.
        X[k]        = even_part[k] + product
        # Segunda metade: par - ímpar-ajustado (reaproveita a MESMA conta, só o sinal
        # troca — é exatamente essa economia que torna a FFT rápida).
        X[k + half] = even_part[k] - product

    # Devolve a transformada completa.
    return X


def ifft_recursive(X: Sequence[complex]) -> list[complex]:
    """FFT inversa recursiva, usando um truque que reaproveita a FFT direta.

    Truque: para "rodar Fourier ao contrário", basta virar as setas (conjugar),
    passar pela MESMA FFT direta, virar de novo e dividir por N. Assim não
    precisamos escrever um segundo algoritmo.
    """
    # N = tamanho.
    N = len(X)
    # Caso vazio.
    if N == 0:
        return []

    # 1) Vira todas as setas (conjugado) — inverte o sentido de giro.
    X_conj = [z.conjugate() for z in X]
    # 2) Passa pela FFT direta normal.
    forward_of_conj = fft_recursive(X_conj)
    # 3) Vira de novo e divide por N (a média) -> resultado da inversa.
    return [z.conjugate() / N for z in forward_of_conj]


# =============================================================================
# 3) FFT iterativa — reordenação bit-reversa + borboletas no lugar
# =============================================================================
#
# Mesmo algoritmo do Cooley-Tukey, mas desenrolado em loops. É o que numpy/scipy
# fazem por baixo (em C). Mais rápido que a recursiva por não ter chamadas de função
# nem fatiar listas o tempo todo. É a versão usada pela pipeline OFDM.
# =============================================================================

def _bit_reverse_indices(n: int) -> list[int]:
    """Devolve a ordem "bit-reversa" dos índices 0..n-1.

    A versão iterativa precisa que a entrada seja reembaralhada nesta ordem especial
    (inverter os bits do número do índice) para as borboletas no lugar funcionarem.

    Exemplo, n=8 (3 bits): 1 (001) <-> 4 (100); 3 (011) <-> 6 (110); etc.
    """
    # Quantos bits tem cada índice (log2 de n).
    bits = int(math.log2(n))
    # Vetor de saída.
    out = [0] * n
    # Para cada índice i...
    for i in range(n):
        # 'rev' vai acumular o índice com os bits invertidos.
        rev = 0
        # cópia de i para ir consumindo bit a bit.
        v = i
        # Para cada bit...
        for _ in range(bits):
            # empurra 'rev' para a esquerda e coloca o bit menos significativo de v.
            rev = (rev << 1) | (v & 1)
            # descarta esse bit de v (anda para o próximo).
            v >>= 1
        # guarda o índice invertido na posição i.
        out[i] = rev
    # Devolve a tabela de reordenação.
    return out


def fft_iterative(x: Sequence[complex]) -> list[complex]:
    """FFT iterativa (Cooley-Tukey radix-2) com reordenação bit-reversa.

    Etapas: (1) reembaralha a entrada na ordem bit-reversa; (2) faz log2(N) estágios
    de borboletas, cada estágio juntando metades cada vez maiores. N potência de 2.
    """
    # Tamanho.
    N = len(x)
    # Caso vazio.
    if N == 0:
        return []
    # Exige potência de 2 (mesmo truque de bit de antes).
    if N & (N - 1) != 0:
        raise ValueError(f"iterative radix-2 FFT requires N power of 2, got N={N}.")

    # ---- Passo 1: permutação bit-reversa ----
    # monta a tabela de reordenação...
    rev = _bit_reverse_indices(N)
    # ...e copia a entrada já nessa nova ordem (garantindo que sejam complexos).
    data: list[complex] = [complex(x[rev[i]]) for i in range(N)]

    # ---- Passo 2: log2(N) estágios de borboletas ----
    # Começa juntando pares de 2, depois 4, 8, ... até N.
    size = 2
    while size <= N:
        # metade do bloco atual.
        half = size // 2
        # menor "passo" de giro deste estágio (a seta que avança a fase a cada k).
        w_step = cmath.exp(-2j * math.pi / size)

        # Percorre o sinal em blocos consecutivos de tamanho 'size'.
        for block_start in range(0, N, size):
            # 'w' é a seta de ajuste; começa sem girar (1+0j) e gira a cada k.
            w = 1 + 0j
            # Dentro do bloco, combina cada par "de cima" com seu par "de baixo".
            for k in range(half):
                # índice do elemento de cima (par).
                even_idx = block_start + k
                # índice do de baixo (ímpar).
                odd_idx = block_start + k + half

                # valor de cima.
                even = data[even_idx]
                # valor de baixo, com o ajuste de fase aplicado.
                odd_w = data[odd_idx] * w

                # borboleta: cima vira (cima + baixo-ajustado)...
                data[even_idx] = even + odd_w
                # ...e baixo vira (cima - baixo-ajustado), reaproveitando a conta.
                data[odd_idx]  = even - odd_w

                # avança a seta de ajuste para o próximo k.
                w *= w_step
        # próximo estágio: blocos com o dobro do tamanho.
        size *= 2

    # Devolve a transformada (calculada no lugar, em 'data').
    return data


def ifft_iterative(X: Sequence[complex]) -> list[complex]:
    """FFT inversa iterativa, pelo mesmo truque do conjugado."""
    # Tamanho.
    N = len(X)
    # Caso vazio.
    if N == 0:
        return []
    # 1) vira as setas, 2) passa pela FFT iterativa, 3) vira de novo e divide por N.
    X_conj = [z.conjugate() for z in X]
    forward = fft_iterative(X_conj)
    return [z.conjugate() / N for z in forward]


# =============================================================================
# 4) Verificação — confirma que nossas implementações batem com o numpy.fft
# =============================================================================

def _max_complex_diff(a: Sequence[complex], b: Sequence[complex]) -> float:
    """Maior diferença |a[i] - b[i]| elemento a elemento (o pior erro)."""
    # Compara par a par e devolve a maior distância encontrada.
    return max(abs(complex(ai) - complex(bi)) for ai, bi in zip(a, b))


def verify_against_numpy(N: int = 256, tolerance: float = 1e-9) -> dict:
    """Roda as 4 implementações contra o numpy.fft e mede o maior erro de cada uma.

    Uma diferença minúscula (~1e-12) é esperada por causa do arredondamento de ponto
    flutuante. Qualquer coisa acima da tolerância indica BUG. É o argumento do
    relatório: "implementei do zero e PROVEI que bate com a referência".
    """
    # numpy só é usado aqui como GABARITO de comparação (não na FFT em si).
    import numpy as np

    # Sinal complexo aleatório, mas reprodutível (semente fixa 42).
    rng = np.random.default_rng(42)
    x_np = rng.normal(size=N) + 1j * rng.normal(size=N)
    # Mesma entrada como lista de Python (para nossas funções).
    x_py = x_np.tolist()

    # Referência: a FFT e a IFFT do numpy.
    X_ref = np.fft.fft(x_np)
    x_ref = np.fft.ifft(X_ref)

    # Dicionário de resultados (nome -> maior erro).
    results = {}

    # Ida — versão ingênua: compara com a referência.
    X_naive = dft_naive(x_py)
    results["dft_naive (forward)"] = _max_complex_diff(X_naive, X_ref.tolist())

    # Ida — versão recursiva.
    X_rec = fft_recursive(x_py)
    results["fft_recursive (forward)"] = _max_complex_diff(X_rec, X_ref.tolist())

    # Ida — versão iterativa (a que a pipeline usa).
    X_it = fft_iterative(x_py)
    results["fft_iterative (forward)"] = _max_complex_diff(X_it, X_ref.tolist())

    # Volta — ingênua.
    x_naive = idft_naive(X_ref.tolist())
    results["idft_naive (inverse)"] = _max_complex_diff(x_naive, x_ref.tolist())

    # Volta — recursiva.
    x_rec = ifft_recursive(X_ref.tolist())
    results["ifft_recursive (inverse)"] = _max_complex_diff(x_rec, x_ref.tolist())

    # Volta — iterativa.
    x_it = ifft_iterative(X_ref.tolist())
    results["ifft_iterative (inverse)"] = _max_complex_diff(x_it, x_ref.tolist())

    # Sanidade: ida e depois volta tem que devolver o sinal original (recursiva).
    roundtrip_rec = ifft_recursive(fft_recursive(x_py))
    results["roundtrip (fft_recursive -> ifft_recursive)"] = (
        _max_complex_diff(roundtrip_rec, x_py)
    )

    # Sanidade: ida e volta (iterativa).
    roundtrip_it = ifft_iterative(fft_iterative(x_py))
    results["roundtrip (fft_iterative -> ifft_iterative)"] = (
        _max_complex_diff(roundtrip_it, x_py)
    )

    # Para cada resultado, anexa um veredito PASS/FAIL conforme a tolerância.
    return {name: (diff, "PASS" if diff < tolerance else "FAIL")
            for name, diff in results.items()}


# =============================================================================
# 5) Ponto de entrada do auto-teste — `python -m core.modulation.fft_from_scratch`
# =============================================================================

def _run_self_test() -> None:
    """Imprime um relatório legível da verificação contra o numpy."""
    # Cabeçalho.
    print("=" * 72)
    print("FFT from-scratch — verification against numpy.fft (N=256, tol=1e-9)")
    print("=" * 72)
    # Roda a verificação para N=256.
    results = verify_against_numpy(N=256)
    # Largura para alinhar os nomes na impressão.
    width = max(len(name) for name in results)
    # Imprime cada linha: nome, erro e veredito.
    for name, (diff, verdict) in results.items():
        print(f"  {name:<{width}}   diff={diff:.3e}   [{verdict}]")
    print("=" * 72)

    # Veredito geral: só passa se TODAS passarem.
    all_pass = all(v == "PASS" for _, v in results.values())
    print("OVERALL:", "ALL PASS" if all_pass else "SOMETHING FAILED")


# Se este arquivo for executado diretamente, roda o auto-teste.
if __name__ == "__main__":
    _run_self_test()

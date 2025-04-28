"""
高速フーリエ変換と数論変換 (FFT and NTT)
- 多項式乗算を効率的に行うためのアルゴリズム
- 主な実装:
  - FFT (高速フーリエ変換): 複素数演算を使用
  - NTT (数論変換): 整数のみで計算可能
  - 畳み込み (Convolution): 多項式乗算やベクトル畳み込み
- 計算量:
  - FFT/NTT: O(n log n)
  - 畳み込み: O(n log n)（素朴な実装では O(n²)）
- 応用:
  - 多項式乗算
  - 文字列マッチング
  - 巨大整数乗算
"""

import math
import cmath
from typing import List, Tuple, Union, Optional
import numpy as np  # 配列操作の効率化のため


def next_power_of_2(n: int) -> int:
    """
    nより大きい最小の2の累乗を返す
    
    Args:
        n: 基準となる数
        
    Returns:
        nより大きいか等しい最小の2の累乗
    """
    return 1 << (n - 1).bit_length()


def bit_reverse(n: int, bits: int) -> int:
    """
    n（bits桁の二進数）のビットを反転した値を返す
    
    Args:
        n: 反転する数値
        bits: ビット数
    
    Returns:
        ビット反転された数値
    """
    result = 0
    for i in range(bits):
        if n & (1 << i):
            result |= 1 << (bits - 1 - i)
    return result


def fft(a: List[complex], inverse: bool = False) -> List[complex]:
    """
    高速フーリエ変換（FFT）
    多項式を係数表現から点値表現へ変換（またはその逆）
    
    Args:
        a: 変換する配列（複素数）
        inverse: 逆変換（IFFT）を行う場合はTrue
    
    Returns:
        変換後の配列（複素数）
    """
    n = len(a)
    if n == 1:
        return a
    
    # 長さを2の累乗に調整
    n = next_power_of_2(n)
    a = a + [0] * (n - len(a))
    
    # ビットリバーサル置換
    log_n = (n - 1).bit_length()
    reversed_indices = [bit_reverse(i, log_n) for i in range(n)]
    
    # 置換後の配列を作成
    a = [a[reversed_indices[i]] for i in range(n)]
    
    # 基数2の蝶型演算
    direction = -1 if inverse else 1
    for s in range(1, log_n + 1):
        m = 1 << s  # 2^s
        omega_m = cmath.exp(direction * 2j * math.pi / m)
        
        for k in range(0, n, m):
            omega = complex(1, 0)
            for j in range(m // 2):
                t = a[k + j + m // 2] * omega
                u = a[k + j]
                a[k + j] = u + t
                a[k + j + m // 2] = u - t
                omega *= omega_m
    
    # 逆変換の場合は各要素をnで割る
    if inverse:
        a = [x / n for x in a]
    
    return a


def fft_numpy(a: List[complex], inverse: bool = False) -> List[complex]:
    """
    NumPyを使用した高速フーリエ変換（FFT）
    
    Args:
        a: 変換する配列（複素数）
        inverse: 逆変換（IFFT）を行う場合はTrue
    
    Returns:
        変換後の配列（複素数）
    """
    n = next_power_of_2(len(a))
    a_padded = np.array(a + [0] * (n - len(a)), dtype=complex)
    
    if inverse:
        return list(np.fft.ifft(a_padded) * n)  # IFFT
    else:
        return list(np.fft.fft(a_padded))  # FFT


def polynomial_multiply_fft(a: List[int], b: List[int]) -> List[int]:
    """
    FFTを使用した多項式乗算
    
    Args:
        a: 1つ目の多項式の係数（最小次数から最大次数の順）
        b: 2つ目の多項式の係数（最小次数から最大次数の順）
    
    Returns:
        積の多項式の係数
    """
    n = len(a) + len(b) - 1
    size = next_power_of_2(n)
    
    # 複素数に変換
    a_complex = [complex(x, 0) for x in a] + [complex(0, 0)] * (size - len(a))
    b_complex = [complex(x, 0) for x in b] + [complex(0, 0)] * (size - len(b))
    
    # FFTで変換
    a_fft = fft(a_complex)
    b_fft = fft(b_complex)
    
    # 点ごとの乗算
    c_fft = [a_fft[i] * b_fft[i] for i in range(size)]
    
    # IFFTで逆変換
    c = fft(c_fft, inverse=True)
    
    # 実数部を取り出し、四捨五入
    return [round(c[i].real) for i in range(n)]


def polynomial_multiply_fft_numpy(a: List[int], b: List[int]) -> List[int]:
    """
    NumPyのFFTを使用した多項式乗算（高速実装）
    
    Args:
        a: 1つ目の多項式の係数（最小次数から最大次数の順）
        b: 2つ目の多項式の係数（最小次数から最大次数の順）
    
    Returns:
        積の多項式の係数
    """
    n = len(a) + len(b) - 1
    size = next_power_of_2(n)
    
    # NumPyのFFTを使用
    a_fft = np.fft.fft(a, size)
    b_fft = np.fft.fft(b, size)
    c_fft = a_fft * b_fft
    c = np.fft.ifft(c_fft)
    
    # 実数部を取り出し、丸める
    return [round(x.real) for x in c[:n]]


def ntt(a: List[int], mod: int, root: int, inverse: bool = False) -> List[int]:
    """
    数論変換（NTT）
    
    Args:
        a: 変換する整数配列
        mod: 法（素数）
        root: 原始根
        inverse: 逆変換を行う場合はTrue
    
    Returns:
        変換後の整数配列
    """
    n = len(a)
    if n == 1:
        return a
    
    # 長さを2の累乗に調整
    n = next_power_of_2(n)
    a = a + [0] * (n - len(a))
    
    # ビットリバーサル置換
    log_n = (n - 1).bit_length()
    reversed_indices = [bit_reverse(i, log_n) for i in range(n)]
    
    # 置換後の配列を作成
    a = [a[reversed_indices[i]] for i in range(n)]
    
    # 原始根の逆元（逆変換用）
    if inverse:
        root = pow(root, mod - 2, mod)  # フェルマーの小定理による逆元
    
    # 基数2の蝶型演算
    for s in range(1, log_n + 1):
        m = 1 << s  # 2^s
        omega_m = pow(root, (mod - 1) // m, mod)
        
        for k in range(0, n, m):
            omega = 1
            for j in range(m // 2):
                t = (a[k + j + m // 2] * omega) % mod
                u = a[k + j]
                a[k + j] = (u + t) % mod
                a[k + j + m // 2] = (u - t) % mod
                if a[k + j + m // 2] < 0:
                    a[k + j + m // 2] += mod
                omega = (omega * omega_m) % mod
    
    # 逆変換の場合は各要素にnの逆元を掛ける
    if inverse:
        n_inv = pow(n, mod - 2, mod)  # nの逆元
        a = [(x * n_inv) % mod for x in a]
    
    return a


def polynomial_multiply_ntt(a: List[int], b: List[int], mod: int = 998244353) -> List[int]:
    """
    NTTを使用した多項式乗算（剰余環上）
    
    Args:
        a: 1つ目の多項式の係数（最小次数から最大次数の順）
        b: 2つ目の多項式の係数（最小次数から最大次数の順）
        mod: 法（デフォルトは998244353 = 119 * 2^23 + 1）
    
    Returns:
        積の多項式の係数（mod で割った余り）
    """
    n = len(a) + len(b) - 1
    size = next_power_of_2(n)
    
    # 係数を mod で割った余りに変換
    a_mod = [(x % mod) for x in a] + [0] * (size - len(a))
    b_mod = [(x % mod) for x in b] + [0] * (size - len(b))
    
    # 原始根（NTTの場合、通常は3）
    root = 3
    
    # NTTで変換
    a_ntt = ntt(a_mod, mod, root)
    b_ntt = ntt(b_mod, mod, root)
    
    # 点ごとの乗算
    c_ntt = [(a_ntt[i] * b_ntt[i]) % mod for i in range(size)]
    
    # 逆NTTで変換
    c = ntt(c_ntt, mod, root, inverse=True)
    
    return c[:n]


def convolve(a: List[int], b: List[int]) -> List[int]:
    """
    畳み込み演算（多項式乗算）
    
    Args:
        a: 1つ目の配列
        b: 2つ目の配列
    
    Returns:
        畳み込み結果の配列
    """
    return polynomial_multiply_fft_numpy(a, b)


def convolve_mod(a: List[int], b: List[int], mod: int) -> List[int]:
    """
    剰余環上での畳み込み演算（NTTを使用）
    
    Args:
        a: 1つ目の配列
        b: 2つ目の配列
        mod: 法
    
    Returns:
        畳み込み結果の配列（mod で割った余り）
    """
    if mod == 998244353 or mod == 167772161 or mod == 469762049:
        # NTTが使える法
        return polynomial_multiply_ntt(a, b, mod)
    else:
        # 一般的な法の場合は3つの素数で計算して中国剰余定理で復元
        mods = [998244353, 167772161, 469762049]
        results = []
        
        for m in mods:
            result = polynomial_multiply_ntt(a, b, m)
            results.append(result)
        
        # 中国剰余定理による復元
        n = len(results[0])
        final_result = [0] * n
        
        # 中国剰余定理の係数
        m1, m2, m3 = mods
        m1m2 = m1 * m2
        m = m1m2 * m3
        
        # 逆元
        inv_m1_m2 = pow(m1m2 % m3, m3 - 2, m3)
        inv_m1_m3 = pow((m1 * m3) % m2, m2 - 2, m2)
        inv_m2_m3 = pow((m2 * m3) % m1, m1 - 2, m1)
        
        for i in range(n):
            x1, x2, x3 = results[0][i], results[1][i], results[2][i]
            
            # ガーナーのアルゴリズムによる復元
            r1 = x1
            r2 = (x2 - r1) * pow(m1, m2 - 2, m2) % m2
            r3 = ((x3 - r1 - r2 * m1) % m3) * inv_m1_m2 % m3
            
            result = (r1 + r2 * m1 + r3 * m1m2) % m
            final_result[i] = result % mod
        
        return final_result


def multiply_large_integers(a: List[int], b: List[int]) -> List[int]:
    """
    巨大整数の乗算（FFTを使用）
    
    Args:
        a: 1つ目の整数の各桁（最下位桁から最上位桁の順）
        b: 2つ目の整数の各桁（最下位桁から最上位桁の順）
    
    Returns:
        積の各桁（最下位桁から最上位桁の順）
    """
    # 多項式乗算
    c = polynomial_multiply_fft_numpy(a, b)
    
    # 桁上がりの処理
    carry = 0
    result = []
    
    for digit in c:
        digit += carry
        result.append(digit % 10)
        carry = digit // 10
    
    # 残りの桁上がりを処理
    while carry > 0:
        result.append(carry % 10)
        carry //= 10
    
    # 先頭の0を削除
    while len(result) > 1 and result[-1] == 0:
        result.pop()
    
    return result


def fft_2d(matrix: List[List[complex]], inverse: bool = False) -> List[List[complex]]:
    """
    2次元高速フーリエ変換（2D FFT）
    
    Args:
        matrix: 変換する2次元配列（複素数）
        inverse: 逆変換を行う場合はTrue
    
    Returns:
        変換後の2次元配列（複素数）
    """
    rows = len(matrix)
    cols = len(matrix[0]) if rows > 0 else 0
    
    # 行方向のFFT
    result = [fft(row, inverse) for row in matrix]
    
    # 転置
    transposed = [[result[j][i] for j in range(rows)] for i in range(cols)]
    
    # 列方向のFFT（転置後は行方向）
    result = [fft(row, inverse) for row in transposed]
    
    # 再度転置して元の形式に戻す
    return [[result[j][i] for j in range(cols)] for i in range(rows)]


# 使用例
def example():
    # FFTによる多項式乗算
    a = [1, 2, 3]  # 1 + 2x + 3x^2
    b = [4, 5, 6]  # 4 + 5x + 6x^2
    
    print("多項式A:", a)
    print("多項式B:", b)
    
    # FFTによる乗算
    result = polynomial_multiply_fft(a, b)
    print("FFTによる多項式乗算の結果:", result)
    
    # NumPy FFTによる乗算
    result_numpy = polynomial_multiply_fft_numpy(a, b)
    print("NumPy FFTによる多項式乗算の結果:", result_numpy)
    
    # NTTによる乗算
    mod = 998244353
    result_ntt = polynomial_multiply_ntt(a, b, mod)
    print(f"NTTによる多項式乗算の結果 (mod {mod}):", result_ntt)
    
    # 大きな整数の乗算
    num1 = [5, 4, 3, 2, 1]  # 12345
    num2 = [8, 7, 6, 5, 4]  # 45678
    result_large = multiply_large_integers(num1, num2)
    print("大きな整数の乗算:", result_large)
    print("検証:", 12345 * 45678, "=", int(''.join(map(str, reversed(result_large)))))


if __name__ == "__main__":
    example()
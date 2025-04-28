"""
高速べき乗法（繰り返し二乗法）
- 効率的に累乗を計算するアルゴリズム
- 主な用途:
  - 大きな指数の累乗計算
  - モジュラべき乗（mod mでのa^b）
  - 行列累乗
- 特徴:
  - 分割統治法に基づくアルゴリズム
  - O(log n)の計算量で効率的
- 計算量:
  - O(log n)（nは指数）
"""

from typing import TypeVar, List, Union, Any, Callable
import copy

T = TypeVar('T')


def binary_exponentiation(base: int, exponent: int) -> int:
    """
    二進数展開による高速べき乗法（Integer版）
    
    Args:
        base: 底
        exponent: 指数（非負整数）
        
    Returns:
        int: base^exponent
    """
    if exponent < 0:
        raise ValueError("指数は非負整数である必要があります")
    
    if exponent == 0:
        return 1
    
    result = 1
    while exponent > 0:
        if exponent & 1:  # exponentが奇数の場合
            result *= base
        base *= base
        exponent >>= 1  # exponent = exponent // 2
    
    return result


def modular_exponentiation(base: int, exponent: int, mod: int) -> int:
    """
    モジュラべき乗法
    
    Args:
        base: 底
        exponent: 指数（非負整数）
        mod: 法（mod > 0）
        
    Returns:
        int: (base^exponent) % mod
    """
    if exponent < 0:
        raise ValueError("指数は非負整数である必要があります")
    if mod <= 0:
        raise ValueError("法は正の整数である必要があります")
    
    base %= mod  # まず底をmod未満にする
    
    if exponent == 0:
        return 1
    
    result = 1
    while exponent > 0:
        if exponent & 1:  # exponentが奇数の場合
            result = (result * base) % mod
        base = (base * base) % mod
        exponent >>= 1  # exponent = exponent // 2
    
    return result


def binary_exponentiation_float(base: float, exponent: int) -> float:
    """
    二進数展開による高速べき乗法（浮動小数点数版）
    
    Args:
        base: 底
        exponent: 指数（整数）
        
    Returns:
        float: base^exponent
    """
    if exponent == 0:
        return 1.0
    
    if exponent < 0:
        return 1.0 / binary_exponentiation_float(base, -exponent)
    
    result = 1.0
    while exponent > 0:
        if exponent & 1:  # exponentが奇数の場合
            result *= base
        base *= base
        exponent >>= 1  # exponent = exponent // 2
    
    return result


def matrix_multiply(A: List[List[int]], B: List[List[int]], mod: int = None) -> List[List[int]]:
    """
    行列の乗算
    
    Args:
        A: 行列A
        B: 行列B
        mod: 法（オプション）
        
    Returns:
        List[List[int]]: A * B
    """
    n = len(A)
    m = len(B[0])
    k = len(B)
    
    if len(A[0]) != k:
        raise ValueError("行列の次元が一致しません")
    
    C = [[0] * m for _ in range(n)]
    
    for i in range(n):
        for j in range(m):
            for l in range(k):
                C[i][j] += A[i][l] * B[l][j]
                if mod:
                    C[i][j] %= mod
    
    return C


def matrix_power(A: List[List[int]], exponent: int, mod: int = None) -> List[List[int]]:
    """
    行列累乗
    
    Args:
        A: 正方行列
        exponent: 指数（非負整数）
        mod: 法（オプション）
        
    Returns:
        List[List[int]]: A^exponent
    """
    if exponent < 0:
        raise ValueError("指数は非負整数である必要があります")
    
    n = len(A)
    if any(len(row) != n for row in A):
        raise ValueError("行列は正方行列である必要があります")
    
    # 単位行列
    result = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
    
    # 繰り返し二乗法
    base = [row[:] for row in A]  # Aのコピー
    while exponent > 0:
        if exponent & 1:  # exponentが奇数の場合
            result = matrix_multiply(result, base, mod)
        base = matrix_multiply(base, base, mod)
        exponent >>= 1
    
    return result


def fibonacci(n: int, mod: int = None) -> int:
    """
    行列累乗を用いたフィボナッチ数の高速計算
    
    Args:
        n: 項数（0-indexed）
        mod: 法（オプション）
        
    Returns:
        int: n番目のフィボナッチ数
    """
    if n <= 1:
        return n
    
    # フィボナッチ数列の遷移行列
    A = [[1, 1], [1, 0]]
    
    # A^(n-1)を計算
    result = matrix_power(A, n - 1, mod)
    
    # F_n = A^(n-1)[0][0]*F_1 + A^(n-1)[0][1]*F_0 = A^(n-1)[0][0]*1 + A^(n-1)[0][1]*0
    return result[0][0]


def generic_exponentiation(base: T, exponent: int, multiply: Callable[[T, T], T], identity: T) -> T:
    """
    ジェネリック高速べき乗法（任意の型に対して使える）
    
    Args:
        base: 底（任意の型）
        exponent: 指数（非負整数）
        multiply: 掛け算の関数
        identity: 単位元（e.g. 数値なら1、行列なら単位行列）
        
    Returns:
        T: base^exponent
    """
    if exponent < 0:
        raise ValueError("指数は非負整数である必要があります")
    
    if exponent == 0:
        return identity
    
    result = identity
    while exponent > 0:
        if exponent & 1:  # exponentが奇数の場合
            result = multiply(result, base)
        base = multiply(base, base)
        exponent >>= 1
    
    return result


# 最大公約数と最小公倍数（高速べき乗法と関連する基本的な数学関数）
def gcd(a: int, b: int) -> int:
    """
    ユークリッドの互除法による最大公約数の計算
    
    Args:
        a, b: 整数
        
    Returns:
        int: aとbの最大公約数
    """
    while b:
        a, b = b, a % b
    return a


def lcm(a: int, b: int) -> int:
    """
    最小公倍数の計算
    
    Args:
        a, b: 整数
        
    Returns:
        int: aとbの最小公倍数
    """
    return a * b // gcd(a, b) if a and b else 0


def extended_gcd(a: int, b: int) -> tuple:
    """
    拡張ユークリッドの互除法
    ax + by = gcd(a, b)となるx, yを求める
    
    Args:
        a, b: 整数
        
    Returns:
        tuple: (gcd, x, y)
    """
    if b == 0:
        return a, 1, 0
    
    gcd_val, x1, y1 = extended_gcd(b, a % b)
    x = y1
    y = x1 - (a // b) * y1
    
    return gcd_val, x, y


def mod_inverse(a: int, m: int) -> int:
    """
    モジュラ逆数の計算
    ax ≡ 1 (mod m)となるxを求める
    
    Args:
        a: 整数
        m: 法（素数）
        
    Returns:
        int: aのmod mにおける逆数
    """
    if gcd(a, m) != 1:
        raise ValueError(f"{a}と{m}は互いに素ではありません")
    
    # フェルマーの小定理を使う場合（mが素数の場合）
    if is_prime(m):
        return modular_exponentiation(a, m - 2, m)
    
    # 拡張ユークリッドの互除法を使う場合（一般的な場合）
    _, x, _ = extended_gcd(a, m)
    return (x % m + m) % m  # 正の値に正規化


def is_prime(n: int) -> bool:
    """
    素数判定（試し割り法）
    
    Args:
        n: 判定する数
        
    Returns:
        bool: nが素数ならTrue
    """
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    
    return True


# 使用例
def example():
    print("===== 高速べき乗法 =====")
    base = 2
    exponent = 10
    result = binary_exponentiation(base, exponent)
    print(f"{base}^{exponent} = {result}")
    
    print("\n===== モジュラべき乗法 =====")
    mod = 1000000007
    result = modular_exponentiation(base, exponent, mod)
    print(f"{base}^{exponent} % {mod} = {result}")
    
    print("\n===== 浮動小数点数の累乗 =====")
    base = 1.5
    exponent = 10
    result = binary_exponentiation_float(base, exponent)
    print(f"{base}^{exponent} = {result}")
    
    print("\n===== 行列累乗 =====")
    # 2x2行列
    A = [[1, 2], [3, 4]]
    exponent = 3
    result = matrix_power(A, exponent)
    print(f"A^{exponent} =")
    for row in result:
        print(row)
    
    print("\n===== 行列累乗によるフィボナッチ数の計算 =====")
    n = 10  # 10番目のフィボナッチ数（0-indexed）
    result = fibonacci(n)
    print(f"F_{n} = {result}")


if __name__ == "__main__":
    example()
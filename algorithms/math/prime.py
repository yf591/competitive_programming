"""
素数関連アルゴリズム（Prime Numbers Algorithms）
- 素数判定、素因数分解、エラトステネスの篩など
- 主な用途:
  - 素数の列挙
  - 素数判定
  - 素因数分解
  - 約数の列挙
- 計算量:
  - 素数判定（試し割り）: O(√n)
  - エラトステネスの篩: O(n log log n)
  - 素因数分解: O(√n)
"""

import math
from typing import List, Dict, Set, Tuple


def is_prime(n: int) -> bool:
    """
    試し割りによる素数判定
    
    Args:
        n: 判定する数
        
    Returns:
        bool: 素数であればTrue、そうでなければFalse
    """
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    
    # 6k±1の形の数のみを試し割り
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    
    return True


def sieve_of_eratosthenes(n: int) -> List[int]:
    """
    エラトステネスの篩によるn以下の素数の列挙
    
    Args:
        n: 上限値
        
    Returns:
        List[int]: n以下の素数のリスト
    """
    if n < 2:
        return []
    
    # 篩を初期化（最初はすべて素数候補）
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    
    # 篩をふるう
    for i in range(2, int(math.sqrt(n)) + 1):
        if sieve[i]:
            # iの倍数を篩から落とす（iは素数）
            for j in range(i * i, n + 1, i):
                sieve[j] = False
    
    # 篩に残った数（素数）をリストに格納
    primes = [i for i in range(n + 1) if sieve[i]]
    return primes


def segmented_sieve(n: int) -> List[int]:
    """
    区間篩（セグメント化されたエラトステネスの篩）
    大きな範囲の素数を省メモリで列挙
    
    Args:
        n: 上限値
        
    Returns:
        List[int]: n以下の素数のリスト
    """
    sqrt_n = int(math.sqrt(n))
    
    # 小さい範囲の素数を通常の篩で求める
    base_primes = sieve_of_eratosthenes(sqrt_n)
    
    # 結果に小さい範囲の素数を追加
    primes = base_primes.copy()
    
    # 区間サイズ（メモリ使用量とのトレードオフ）
    segment_size = max(sqrt_n, 1000)
    
    # 区間ごとに篩をかける
    for segment_start in range(sqrt_n + 1, n + 1, segment_size):
        segment_end = min(segment_start + segment_size - 1, n)
        segment = [True] * (segment_end - segment_start + 1)
        
        # 各基本素数について、セグメント内の倍数を篩い落とす
        for prime in base_primes:
            # セグメント内のprimeの最初の倍数を求める
            start_idx = (segment_start + prime - 1) // prime * prime
            if start_idx < segment_start:
                start_idx += prime
            
            # セグメント内のprimeの倍数を篩い落とす
            for j in range(start_idx, segment_end + 1, prime):
                segment[j - segment_start] = False
        
        # セグメント内の素数を結果に追加
        for i in range(segment_end - segment_start + 1):
            if segment[i]:
                primes.append(segment_start + i)
    
    return primes


def prime_factorization(n: int) -> List[int]:
    """
    試し割りによる素因数分解
    
    Args:
        n: 素因数分解する数
        
    Returns:
        List[int]: 素因数のリスト（重複を含む）
    """
    factors = []
    
    # 2で割り切れる限り割り続ける
    while n % 2 == 0:
        factors.append(2)
        n //= 2
    
    # 奇数の素因数を探す
    i = 3
    while i * i <= n:
        while n % i == 0:
            factors.append(i)
            n //= i
        i += 2
    
    # 残りが1より大きければ、それも素数
    if n > 1:
        factors.append(n)
    
    return factors


def prime_factorization_with_sieve(n: int, primes: List[int] = None) -> List[int]:
    """
    エラトステネスの篩を使用した素因数分解
    
    Args:
        n: 素因数分解する数
        primes: 事前に計算された素数のリスト（省略可）
        
    Returns:
        List[int]: 素因数のリスト（重複を含む）
    """
    if primes is None:
        # 事前に素数を計算（ルートnまで）
        primes = sieve_of_eratosthenes(int(math.sqrt(n)))
    
    factors = []
    
    # 与えられた素数リストで割り切れる限り割る
    for prime in primes:
        if prime * prime > n:
            break
        
        while n % prime == 0:
            factors.append(prime)
            n //= prime
    
    # 残りが1より大きければ、それも素数
    if n > 1:
        factors.append(n)
    
    return factors


def count_divisors(n: int) -> int:
    """
    正の約数の個数を計算
    素因数分解を利用: n = p1^a1 * p2^a2 * ... * pk^ak のとき、
    約数の個数は (a1+1) * (a2+1) * ... * (ak+1)
    
    Args:
        n: 約数を数える対象の数
        
    Returns:
        int: 約数の個数
    """
    if n == 0:
        return 0
    
    # 素因数分解（指数を含む）
    factorization = {}
    i = 2
    while i * i <= n:
        while n % i == 0:
            factorization[i] = factorization.get(i, 0) + 1
            n //= i
        i += 1
    
    if n > 1:
        factorization[n] = factorization.get(n, 0) + 1
    
    # 約数の個数を計算
    result = 1
    for exponent in factorization.values():
        result *= (exponent + 1)
    
    return result


def list_divisors(n: int) -> List[int]:
    """
    正の約数をすべて列挙
    
    Args:
        n: 約数を列挙する対象の数
        
    Returns:
        List[int]: 約数のリスト（昇順）
    """
    if n <= 0:
        return []
    
    divisors = []
    large_divisors = []
    
    # 約数を探す（ルートnまで）
    for i in range(1, int(math.sqrt(n)) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:  # 重複を避ける
                large_divisors.append(n // i)
    
    # 大きい約数を反転して元のリストに追加
    divisors.extend(reversed(large_divisors))
    
    return divisors


def gcd(a: int, b: int) -> int:
    """
    ユークリッドの互除法による最大公約数の計算
    
    Args:
        a: 1つ目の整数
        b: 2つ目の整数
        
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
        a: 1つ目の整数
        b: 2つ目の整数
        
    Returns:
        int: aとbの最小公倍数
    """
    return a * b // gcd(a, b) if a and b else 0


def extended_gcd(a: int, b: int) -> Tuple[int, int, int]:
    """
    拡張ユークリッドの互除法
    ax + by = gcd(a, b) となる整数 x, y を求める
    
    Args:
        a: 1つ目の整数
        b: 2つ目の整数
        
    Returns:
        Tuple[int, int, int]: (gcd(a, b), x, y)
    """
    if a == 0:
        return b, 0, 1
    
    gcd_val, x1, y1 = extended_gcd(b % a, a)
    x = y1 - (b // a) * x1
    y = x1
    
    return gcd_val, x, y


def modular_inverse(a: int, m: int) -> int:
    """
    モジュラ逆元を計算
    a * x ≡ 1 (mod m) となる x を求める
    
    Args:
        a: 逆元を求める数
        m: 法（素数である必要がある）
        
    Returns:
        int: aのmod mにおける逆元
    """
    g, x, y = extended_gcd(a, m)
    if g != 1:
        raise ValueError(f"モジュラ逆元が存在しません (gcd(a, m) = {g})")
    else:
        return (x % m + m) % m  # 正の値を返す


def linear_sieve(n: int) -> Tuple[List[int], List[int]]:
    """
    線形篩（Linear Sieve）による素数列挙と最小素因数の計算
    通常のエラトステネスの篩より効率的なアルゴリズム
    
    Args:
        n: 上限値
        
    Returns:
        Tuple[List[int], List[int]]: (素数のリスト, 各数の最小素因数)
    """
    lp = [0] * (n + 1)  # 最小素因数
    primes = []
    
    for i in range(2, n + 1):
        if lp[i] == 0:
            # iが素数の場合
            lp[i] = i
            primes.append(i)
        
        # iの倍数にiの素因数情報を伝播
        j = 0
        while j < len(primes) and primes[j] <= lp[i] and i * primes[j] <= n:
            lp[i * primes[j]] = primes[j]
            j += 1
    
    return primes, lp


def prime_factorization_linear_sieve(n: int, lp: List[int]) -> Dict[int, int]:
    """
    線形篩で求めた最小素因数を使って高速に素因数分解
    
    Args:
        n: 素因数分解する数
        lp: 最小素因数のリスト
        
    Returns:
        Dict[int, int]: {素因数: 指数} の辞書
    """
    factors = {}
    
    while n > 1:
        prime_factor = lp[n]
        factors[prime_factor] = factors.get(prime_factor, 0) + 1
        n //= prime_factor
    
    return factors


# エラトステネスの篩の変種：合成数の検出
def sieve_of_eratosthenes_composite(n: int) -> List[int]:
    """
    エラトステネスの篩で合成数のみを列挙
    
    Args:
        n: 上限値
        
    Returns:
        List[int]: n以下の合成数のリスト
    """
    if n < 4:  # 4未満には合成数はない
        return []
    
    # 篩を初期化（最初はすべて素数候補）
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    
    # 篩をふるう
    for i in range(2, int(math.sqrt(n)) + 1):
        if sieve[i]:
            for j in range(i * i, n + 1, i):
                sieve[j] = False
    
    # 篩から落とされた数（合成数）をリストに格納
    composites = [i for i in range(4, n + 1) if not sieve[i]]
    return composites


# 使用例
def example():
    n = 100  # 上限値
    
    print("===== 素数判定 =====")
    print(f"17は素数か？: {is_prime(17)}")
    print(f"35は素数か？: {is_prime(35)}")
    
    print("\n===== エラトステネスの篩 =====")
    primes = sieve_of_eratosthenes(n)
    print(f"{n}以下の素数: {primes}")
    print(f"{n}以下の素数の個数: {len(primes)}")
    
    print("\n===== 素因数分解 =====")
    num = 60
    factors = prime_factorization(num)
    print(f"{num}の素因数分解: {factors}")
    
    print("\n===== 約数列挙 =====")
    divisors = list_divisors(num)
    print(f"{num}の約数: {divisors}")
    print(f"{num}の約数の個数: {count_divisors(num)}")
    
    print("\n===== GCDとLCM =====")
    a, b = 24, 36
    print(f"gcd({a}, {b}) = {gcd(a, b)}")
    print(f"lcm({a}, {b}) = {lcm(a, b)}")
    
    print("\n===== 拡張ユークリッド =====")
    x, y = 42, 30
    g, s, t = extended_gcd(x, y)
    print(f"{x} * {s} + {y} * {t} = {g}")
    
    print("\n===== モジュラ逆元 =====")
    a, m = 3, 11  # mは素数
    inv = modular_inverse(a, m)
    print(f"{a}のmod {m}における逆元: {inv}")
    print(f"検証: {a} * {inv} mod {m} = {(a * inv) % m}")
    
    print("\n===== 線形篩 =====")
    primes, lp = linear_sieve(n)
    print(f"線形篩による{n}以下の素数の個数: {len(primes)}")
    
    print("\n===== 線形篩による素因数分解 =====")
    num = 42
    factors = prime_factorization_linear_sieve(num, lp)
    print(f"{num}の素因数分解: {factors}")


if __name__ == "__main__":
    example()
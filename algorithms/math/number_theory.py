"""
整数論アルゴリズム集 (Number Theory Algorithms)
- 整数論に関する様々なアルゴリズム群
- 主な実装:
  - 最大公約数(GCD)と最小公倍数(LCM)
  - 拡張ユークリッドの互除法
  - オイラーのφ関数（トーシェント関数）
  - モジュラ逆元
  - 素因数分解
  - 約数列挙
  - メビウス関数
  - 中国剰余定理 (CRT)
- 計算量:
  - GCD, LCM: O(log min(a, b))
  - 拡張ユークリッド: O(log min(a, b))
  - 素因数分解: O(√n)
  - 約数列挙: O(√n)
  - エラトステネスの篩: O(n log log n)
"""

from typing import List, Tuple, Dict, Set, Optional
import math


def gcd(a: int, b: int) -> int:
    """
    最大公約数を求める (Greatest Common Divisor)
    
    Args:
        a, b: 整数
    
    Returns:
        aとbの最大公約数
    """
    while b:
        a, b = b, a % b
    return abs(a)


def lcm(a: int, b: int) -> int:
    """
    最小公倍数を求める (Least Common Multiple)
    
    Args:
        a, b: 整数
    
    Returns:
        aとbの最小公倍数
    """
    if a == 0 and b == 0:
        return 0
    return abs(a * b) // gcd(a, b)


def extended_gcd(a: int, b: int) -> Tuple[int, int, int]:
    """
    拡張ユークリッドの互除法で ax + by = gcd(a, b) の整数解を求める
    
    Args:
        a, b: 整数
    
    Returns:
        (gcd(a, b), x, y): 最大公約数と解x, y
    """
    if b == 0:
        return a, 1, 0
    
    g, x, y = extended_gcd(b, a % b)
    return g, y, x - (a // b) * y


def mod_inverse(a: int, m: int) -> int:
    """
    モジュラ逆元 a^(-1) mod m を求める
    a と m が互いに素であることが必要
    
    Args:
        a: 逆元を求めたい数
        m: 法
    
    Returns:
        a^(-1) mod m
    
    Raises:
        ValueError: a と m が互いに素でない場合
    """
    g, x, y = extended_gcd(a, m)
    
    if g != 1:
        raise ValueError(f"{a}と{m}は互いに素ではありません。逆元が存在しません。")
    
    return x % m


def is_prime(n: int) -> bool:
    """
    素数判定（試し割り法）
    
    Args:
        n: 判定する整数
    
    Returns:
        nが素数ならTrue、そうでなければFalse
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


def miller_rabin(n: int, k: int = 40) -> bool:
    """
    ミラー・ラビン素数判定法
    確率的アルゴリズムだが、k回のテストで 1-(1/4)^k の確率で正しい
    2^64 以下の整数に対しては、[2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37] を
    テストすれば決定的に判定できる
    
    Args:
        n: 判定する整数
        k: テストの回数
    
    Returns:
        nが素数である確率が高ければTrue、そうでなければFalse
    """
    if n == 2 or n == 3:
        return True
    if n <= 1 or n % 2 == 0:
        return False
    
    # n - 1 = 2^r * d の形に書き換える
    r, d = 0, n - 1
    while d % 2 == 0:
        r += 1
        d //= 2
    
    # k回のテストを実行
    import random
    for _ in range(k):
        a = random.randint(2, n - 2)
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
            
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    
    return True


def sieve_of_eratosthenes(n: int) -> List[int]:
    """
    エラトステネスの篩で n 以下の素数を列挙
    
    Args:
        n: 上限
    
    Returns:
        n以下の素数のリスト
    """
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    
    return [i for i in range(n + 1) if is_prime[i]]


def segment_sieve(l: int, r: int) -> List[int]:
    """
    区間 [l, r] の素数を列挙（区間篩）
    
    Args:
        l: 下限
        r: 上限
    
    Returns:
        区間 [l, r] の素数のリスト
    """
    is_prime_small = [True] * (int((r)**0.5) + 1)
    is_prime_segment = [True] * (r - l + 1)
    
    # 小さい範囲の素数を列挙
    is_prime_small[0] = is_prime_small[1] = False
    for i in range(2, int((r)**0.5) + 1):
        if is_prime_small[i]:
            for j in range(i*i, int((r)**0.5) + 1, i):
                is_prime_small[j] = False
    
    # 区間 [l, r] の数を篩にかける
    for i in range(2, int((r)**0.5) + 1):
        if is_prime_small[i]:
            # l以上の最小のiの倍数から始める
            start = max(i * i, (l + i - 1) // i * i)
            for j in range(start, r + 1, i):
                if j >= l:
                    is_prime_segment[j - l] = False
    
    # l=1の場合は1を除外
    if l <= 1 and r >= 1:
        is_prime_segment[1 - l] = False
    
    return [i + l for i in range(r - l + 1) if is_prime_segment[i]]


def prime_factorize(n: int) -> Dict[int, int]:
    """
    素因数分解 O(√n)
    
    Args:
        n: 因数分解する整数
    
    Returns:
        素因数とその指数のディクショナリ {素因数: 指数}
    """
    factors = {}
    
    # 2で割り切れる場合
    while n % 2 == 0:
        factors[2] = factors.get(2, 0) + 1
        n //= 2
    
    # 3以上の奇数で試し割り
    i = 3
    while i * i <= n:
        while n % i == 0:
            factors[i] = factors.get(i, 0) + 1
            n //= i
        i += 2
    
    # 残った数が1より大きければ素数
    if n > 1:
        factors[n] = factors.get(n, 0) + 1
    
    return factors


def factorize(n: int) -> List[int]:
    """
    全ての約数を列挙
    
    Args:
        n: 約数を列挙したい整数
    
    Returns:
        nの全ての約数のリスト（ソート済み）
    """
    factors = []
    i = 1
    
    # √n までの約数を見つける
    while i * i <= n:
        if n % i == 0:
            factors.append(i)
            # i が n の平方根でなければ対応する約数も追加
            if i * i != n:
                factors.append(n // i)
        i += 1
    
    # ソートして返す
    return sorted(factors)


def euler_phi(n: int) -> int:
    """
    オイラーのφ関数（トーシェント関数）
    n以下でnと互いに素な自然数の個数を返す
    
    Args:
        n: 計算したい整数
    
    Returns:
        φ(n)
    """
    if n <= 0:
        return 0
    
    result = n
    
    # オイラー関数の公式: φ(n) = n * Π(1 - 1/p)
    # ここでpはnの素因数
    i = 2
    while i * i <= n:
        if n % i == 0:
            result -= result // i
            while n % i == 0:
                n //= i
        i += 1
    
    if n > 1:
        result -= result // n
    
    return result


def phi_list(n: int) -> List[int]:
    """
    1からnまでのオイラーのφ関数の値を計算
    
    Args:
        n: 上限
    
    Returns:
        [φ(1), φ(2), ..., φ(n)]
    """
    phi = [i for i in range(n + 1)]
    
    for i in range(2, n + 1):
        if phi[i] == i:  # iが素数なら
            for j in range(i, n + 1, i):
                phi[j] -= phi[j] // i
    
    return phi


def mobius(n: int) -> int:
    """
    メビウス関数を計算
    μ(n) = 
        1  (nが平方因子を持たず、素因数の個数が偶数)
        -1 (nが平方因子を持たず、素因数の個数が奇数)
        0  (nが平方因子を持つ)
    
    Args:
        n: 計算したい整数
    
    Returns:
        μ(n)
    """
    if n <= 0:
        return 0
    
    # 素因数分解して重複因子を調べる
    factors = prime_factorize(n)
    
    # 平方因子があれば0
    for count in factors.values():
        if count >= 2:
            return 0
    
    # 素因数の個数の偶奇で1か-1
    return 1 if len(factors) % 2 == 0 else -1


def mobius_list(n: int) -> List[int]:
    """
    1からnまでのメビウス関数の値を計算
    
    Args:
        n: 上限
    
    Returns:
        [μ(1), μ(2), ..., μ(n)]
    """
    mu = [0] * (n + 1)
    mu[1] = 1
    
    for i in range(1, n + 1):
        for j in range(i * 2, n + 1, i):
            mu[j] -= mu[i]
    
    return mu


def chinese_remainder_theorem(remainders: List[int], modulos: List[int]) -> Tuple[int, int]:
    """
    中国剰余定理 (Chinese Remainder Theorem)
    x ≡ r_i (mod m_i) を満たす x を求める
    
    Args:
        remainders: 余り r_i のリスト
        modulos: 法 m_i のリスト
    
    Returns:
        (x, M): x は解、M は全法の最小公倍数
        解がない場合は (0, -1)
    """
    if len(remainders) != len(modulos):
        raise ValueError("剰余と法のリストの長さが一致しません")
    
    x = 0
    M = 1
    
    for r, m in zip(remainders, modulos):
        g, p, q = extended_gcd(M, m)
        
        if (r - x) % g != 0:
            return 0, -1  # 解なし
        
        x = (x + (r - x) // g * p * M) % (M * m // g)
        M = M * m // g
    
    return x, M


def factorial_mod(n: int, mod: int) -> int:
    """
    n! mod p を計算
    
    Args:
        n: 階乗を計算したい数
        mod: 法
    
    Returns:
        n! mod p
    """
    result = 1
    for i in range(2, n + 1):
        result = (result * i) % mod
    return result


def binomial_coefficient_mod(n: int, k: int, mod: int) -> int:
    """
    二項係数 nCk mod p を計算
    ただし、p は素数であることを仮定
    
    Args:
        n, k: 二項係数 nCk のパラメータ
        mod: 法（素数）
    
    Returns:
        nCk mod p
    """
    if k < 0 or k > n:
        return 0
    
    if k == 0 or k == n:
        return 1
    
    k = min(k, n - k)
    
    numerator = 1  # 分子
    denominator = 1  # 分母
    
    for i in range(1, k + 1):
        numerator = (numerator * (n - (i - 1))) % mod
        denominator = (denominator * i) % mod
    
    # フェルマーの小定理を使って逆元を計算
    return (numerator * pow(denominator, mod - 2, mod)) % mod


def lucas_theorem(n: int, k: int, mod: int) -> int:
    """
    ルーカスの定理で大きな n に対する nCk mod p を計算
    p は素数である必要がある
    
    Args:
        n, k: 二項係数 nCk のパラメータ
        mod: 法（素数）
    
    Returns:
        nCk mod p
    """
    if k < 0 or k > n:
        return 0
    
    if k == 0 or k == n:
        return 1
    
    # nとkをp進展開
    n_digits = []
    k_digits = []
    
    while n > 0:
        n_digits.append(n % mod)
        n //= mod
    
    while k > 0:
        k_digits.append(k % mod)
        k //= mod
    
    # kがnより桁数が多い場合は0
    while len(k_digits) < len(n_digits):
        k_digits.append(0)
    
    # ルーカスの定理を適用
    result = 1
    for ni, ki in zip(n_digits, k_digits):
        if ki > ni:
            return 0
        # 各桁の二項係数を計算
        result = (result * binomial_coefficient_mod(ni, ki, mod)) % mod
    
    return result


def discrete_log(a: int, b: int, m: int) -> int:
    """
    離散対数問題: a^x ≡ b (mod m) となる最小のxを求める
    Baby-step Giant-step アルゴリズム O(√m)
    
    Args:
        a: 底
        b: 余り
        m: 法
    
    Returns:
        解 x、解がなければ -1
    """
    import math
    
    # aとmが互いに素でない場合の処理
    g = math.gcd(a, m)
    if g > 1:
        if b % g != 0:
            return -1  # 解なし
        
        # a^x ≡ b (mod m) を a/g * a'^(x-1) ≡ b/g (mod m/g) に変換
        return discrete_log(a // g, b // g * pow(a, m // g - 1, m // g), m // g) + 1
    
    # Baby-step Giant-step アルゴリズム
    n = int(math.sqrt(m)) + 1
    
    # Baby-step: a^j を計算して保存
    baby_steps = {}
    for j in range(n):
        baby_steps[pow(a, j, m)] = j
    
    # Giant-step: a^(-n)^i * b を計算して検索
    a_inv_n = pow(a, n * (m - 2), m)  # a^(-n) = a^(n·(m-2)) (mod m)
    
    for i in range(n):
        value = (b * pow(a_inv_n, i, m)) % m
        if value in baby_steps:
            return i * n + baby_steps[value]
    
    return -1  # 解なし


def primitive_root(p: int) -> int:
    """
    素数 p の原始根を求める
    
    Args:
        p: 素数
    
    Returns:
        p の最小の原始根、存在しなければ -1
    """
    if p == 2:
        return 1
    
    # p-1 の素因数を求める
    phi = p - 1
    factors = set(prime_factorize(phi).keys())
    
    # 原始根の候補を検証
    for g in range(2, p):
        is_primitive_root = True
        
        # g^(phi/q) ≠ 1 (mod p) をすべての素因数qで確認
        for q in factors:
            if pow(g, phi // q, p) == 1:
                is_primitive_root = False
                break
        
        if is_primitive_root:
            return g
    
    return -1  # 原始根が見つからない（通常は素数なら存在する）


def legendre_symbol(a: int, p: int) -> int:
    """
    ルジャンドル記号 (a/p) を計算
    p は奇素数
    
    Args:
        a: 整数
        p: 奇素数
    
    Returns:
        1  (a が p を法としたときの二次剰余)
        -1 (a が p を法としたときの非二次剰余)
        0  (a が p で割り切れる場合)
    """
    a %= p
    if a == 0:
        return 0
    
    if a == 1:
        return 1
    
    if a == 2:
        # 2の場合は p mod 8 で決まる
        if p % 8 == 1 or p % 8 == 7:
            return 1
        return -1
    
    if a % 2 == 0:
        # 2に関する二次剰余の法則
        return legendre_symbol(2, p) * legendre_symbol(a // 2, p)
    
    if a % 4 == 3 and p % 4 == 3:
        # 二次相互法則の特殊ケース
        return -legendre_symbol(p, a)
    
    # 二次相互法則の一般形
    return legendre_symbol(p % a, a)


def tonelli_shanks(n: int, p: int) -> Optional[int]:
    """
    トネリ・シャンクスのアルゴリズムで x^2 ≡ n (mod p) の解を求める
    p は奇素数
    
    Args:
        n: 整数
        p: 奇素数
    
    Returns:
        x（解が存在する場合）、解がなければ None
    """
    if legendre_symbol(n, p) != 1:
        return None  # 解なし
    
    if p % 4 == 3:
        # p ≡ 3 (mod 4) なら解は n^((p+1)/4) (mod p)
        return pow(n, (p + 1) // 4, p)
    
    # p-1 = Q * 2^S の形に分解
    Q, S = p - 1, 0
    while Q % 2 == 0:
        Q //= 2
        S += 1
    
    # 非二次剰余を見つける
    z = 2
    while legendre_symbol(z, p) != -1:
        z += 1
    
    # 初期値の設定
    M = S
    c = pow(z, Q, p)
    t = pow(n, Q, p)
    R = pow(n, (Q + 1) // 2, p)
    
    while t != 1:
        # t^(2^i) = 1 となる最小の i を見つける
        i = 0
        temp = t
        while temp != 1 and i < M:
            temp = (temp * temp) % p
            i += 1
        
        if i == 0:
            return R
        
        # b = c^(2^(M-i-1))
        b = pow(c, 1 << (M - i - 1), p)
        
        M = i
        c = (b * b) % p
        t = (t * c) % p
        R = (R * b) % p
    
    return R


# 使用例
def example():
    # GCDとLCM
    print("GCD(48, 18) =", gcd(48, 18))  # 6
    print("LCM(48, 18) =", lcm(48, 18))  # 144
    
    # 拡張ユークリッド
    a, b = 35, 15
    g, x, y = extended_gcd(a, b)
    print(f"{a}x + {b}y = {g} の解: x = {x}, y = {y}")
    print(f"検算: {a}*{x} + {b}*{y} = {a*x + b*y}")
    
    # モジュラ逆元
    a, m = 3, 11
    inv = mod_inverse(a, m)
    print(f"{a} * {inv} ≡ 1 (mod {m}):", (a * inv) % m == 1)
    
    # 素数判定
    n = 997
    print(f"{n} は素数か？:", is_prime(n))
    print(f"ミラー・ラビンでの判定: {miller_rabin(n)}")
    
    # 素数列挙
    limit = 20
    print(f"{limit}以下の素数:", sieve_of_eratosthenes(limit))
    
    # 区間篩
    l, r = 90, 100
    print(f"区間 [{l}, {r}] の素数:", segment_sieve(l, r))
    
    # 素因数分解
    n = 84
    print(f"{n}の素因数分解:", prime_factorize(n))
    
    # 約数列挙
    print(f"{n}の約数:", factorize(n))
    
    # オイラー関数
    n = 36
    print(f"φ({n}) =", euler_phi(n))
    
    # メビウス関数
    print(f"μ({n}) =", mobius(n))
    
    # 中国剰余定理
    remainders = [2, 3, 2]
    modulos = [3, 5, 7]
    x, M = chinese_remainder_theorem(remainders, modulos)
    print(f"x ≡ {remainders[0]} (mod {modulos[0]}), "
          f"x ≡ {remainders[1]} (mod {modulos[1]}), "
          f"x ≡ {remainders[2]} (mod {modulos[2]})")
    print(f"解: x ≡ {x} (mod {M})")


if __name__ == "__main__":
    example()
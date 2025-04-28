"""
モジュラ算術アルゴリズム集 (Modular Arithmetic Algorithms)
- 剰余演算に関するアルゴリズム群
- 主な実装:
  - モジュラ逆元（拡張ユークリッドアルゴリズム、フェルマーの小定理）
  - 中国剰余定理 (Chinese Remainder Theorem; CRT)
  - モジュラ累乗、除算、対数
  - モンゴメリリダクション
- 計算量:
  - モジュラ逆元: O(log mod)
  - 中国剰余定理: O(n log mod)（n個の合同式の場合）
  - モジュラ累乗: O(log exponent)
"""

from typing import List, Tuple, Optional
import math


def extended_gcd(a: int, b: int) -> Tuple[int, int, int]:
    """
    拡張ユークリッドアルゴリズム
    ax + by = gcd(a, b) となる (gcd, x, y) を求める
    
    Args:
        a: 第1の整数
        b: 第2の整数
    
    Returns:
        (gcd, x, y) の組
    """
    if a == 0:
        return (b, 0, 1)
    
    gcd, x1, y1 = extended_gcd(b % a, a)
    x = y1 - (b // a) * x1
    y = x1
    
    return (gcd, x, y)


def mod_inverse(a: int, m: int) -> int:
    """
    モジュラ逆元 a^(-1) mod m を拡張ユークリッドアルゴリズムで計算
    
    Args:
        a: 逆元を求める整数
        m: 法（素数である必要はない）
    
    Returns:
        aの逆元（mod m）
    
    Raises:
        ValueError: aとmが互いに素でない場合
    """
    g, x, y = extended_gcd(a, m)
    
    if g != 1:
        raise ValueError(f"{a}と{m}は互いに素ではないため、モジュラ逆元が存在しません。")
    else:
        # 負の値にならないようにmで割った余りを返す
        return (x % m + m) % m


def mod_inverse_fermat(a: int, p: int) -> int:
    """
    モジュラ逆元 a^(-1) mod p をフェルマーの小定理で計算
    
    Args:
        a: 逆元を求める整数
        p: 法（素数である必要がある）
    
    Returns:
        aの逆元（mod p）
    """
    return pow(a, p - 2, p)


def chinese_remainder_theorem(remainders: List[int], moduli: List[int]) -> Optional[Tuple[int, int]]:
    """
    中国剰余定理（Chinese Remainder Theorem; CRT）
    x ≡ r_i (mod m_i) を満たす x を計算
    
    Args:
        remainders: 余り r_i のリスト
        moduli: 法 m_i のリスト
    
    Returns:
        (x, lcm)のタプル。xは合同式を満たす最小の非負整数、lcmは全てのmoduliの最小公倍数
        解がない場合はNone
    """
    if len(remainders) != len(moduli):
        raise ValueError("余りと法のリストの長さが一致していません。")
    
    n = len(remainders)
    
    # ベースケース
    result = remainders[0]
    modulus = moduli[0]
    
    # 合同式をひとつずつ統合
    for i in range(1, n):
        # 現在の解と次の合同式の整合性をチェック
        gcd, u, v = extended_gcd(modulus, moduli[i])
        
        # 整合性がない場合
        if (remainders[i] - result) % gcd != 0:
            return None
        
        # 解の更新
        result = (result + (remainders[i] - result) % moduli[i] * u % moduli[i] * modulus // gcd) % (modulus * moduli[i] // gcd)
        modulus = modulus * moduli[i] // gcd
    
    # 結果は0以上modulus未満
    result = (result % modulus + modulus) % modulus
    
    return (result, modulus)


def mod_add(a: int, b: int, mod: int) -> int:
    """
    モジュラ加算: (a + b) mod m
    
    Args:
        a: 第1の数
        b: 第2の数
        mod: 法
    
    Returns:
        (a + b) mod m
    """
    return (a + b) % mod


def mod_sub(a: int, b: int, mod: int) -> int:
    """
    モジュラ減算: (a - b) mod m
    
    Args:
        a: 第1の数
        b: 第2の数
        mod: 法
    
    Returns:
        (a - b) mod m（常に非負の値）
    """
    return (a - b) % mod


def mod_mul(a: int, b: int, mod: int) -> int:
    """
    モジュラ乗算: (a * b) mod m
    
    Args:
        a: 第1の数
        b: 第2の数
        mod: 法
    
    Returns:
        (a * b) mod m
    """
    return (a * b) % mod


def mod_div(a: int, b: int, mod: int) -> int:
    """
    モジュラ除算: (a / b) mod m = a * b^(-1) mod m
    
    Args:
        a: 分子
        b: 分母
        mod: 法
    
    Returns:
        (a / b) mod m
    
    Raises:
        ValueError: bとmodが互いに素でない場合
    """
    return (a * mod_inverse(b, mod)) % mod


def mod_pow(base: int, exponent: int, mod: int) -> int:
    """
    モジュラ累乗: (base^exponent) mod m
    
    Args:
        base: 底
        exponent: 指数
        mod: 法
    
    Returns:
        (base^exponent) mod m
    """
    if mod == 1:
        return 0
    
    # 負の指数の場合は逆元を計算
    if exponent < 0:
        return mod_pow(mod_inverse(base, mod), -exponent, mod)
    
    return pow(base, exponent, mod)


def mod_sqrt(n: int, p: int) -> Optional[int]:
    """
    モジュラ平方根: x^2 ≡ n (mod p) の解を求める（pは奇素数）
    トネリ・シャンクスアルゴリズム
    
    Args:
        n: 平方剰余
        p: 法（奇素数）
    
    Returns:
        x^2 ≡ n (mod p) の解の一つ（解がある場合のみ）
        解がない場合はNone
    """
    if n == 0:
        return 0
    
    # ルジャンドル記号でnが平方剰余かチェック
    if pow(n, (p - 1) // 2, p) != 1:
        return None
    
    # p ≡ 3 (mod 4) の場合は簡単な公式がある
    if p % 4 == 3:
        return pow(n, (p + 1) // 4, p)
    
    # p ≡ 1 (mod 4) の場合はトネリ・シャンクスアルゴリズムを使用
    s = 0
    q = p - 1
    while q % 2 == 0:
        s += 1
        q //= 2
    
    # 非平方剰余 zを見つける
    z = 2
    while pow(z, (p - 1) // 2, p) == 1:
        z += 1
    
    m = s
    c = pow(z, q, p)
    t = pow(n, q, p)
    r = pow(n, (q + 1) // 2, p)
    
    while t != 1:
        # t^(2^i) ≡ 1 (mod p) となる最小のiを見つける
        i = 0
        temp = t
        while temp != 1:
            temp = (temp * temp) % p
            i += 1
            if i == m:  # これはありえない
                return None
        
        # b = c^(2^(m-i-1)) mod p
        b = pow(c, 1 << (m - i - 1), p)
        
        m = i
        c = (b * b) % p
        t = (t * c) % p
        r = (r * b) % p
    
    return r


def discrete_log(a: int, b: int, m: int) -> Optional[int]:
    """
    離散対数問題: a^x ≡ b (mod m) となるxを求める
    Baby-step Giant-step アルゴリズム
    
    Args:
        a: 底
        b: 値
        m: 法
    
    Returns:
        a^x ≡ b (mod m) を満たす最小の非負整数x
        解がない場合はNone
    """
    a %= m
    b %= m
    
    # mが1の場合は自明
    if m == 1:
        return 0
    
    # gcd(a, m) > 1 の場合の特別処理
    g = math.gcd(a, m)
    if g > 1:
        if b % g != 0:
            return None  # 解なし
        
        # a' = a/g, b' = b/g, m' = m/g として再帰的に解く
        m //= g
        b //= g
        e = 1
        
        # a^e ≡ b (mod m) を満たすeを求める
        for i in range(30):  # 最大30回の試行（必要に応じて調整）
            if b % g != 0:
                return None
            b //= g
            m //= g
            e += 1
            if math.gcd(a, m) == 1:
                break
        
        # 再帰呼び出し
        result = discrete_log(a % m, b % m, m)
        if result is None:
            return None
        return result + e
    
    # Baby-step Giant-step アルゴリズム
    n = int(math.sqrt(m)) + 1
    
    # Baby steps
    table = {}
    for i in range(n):
        value = pow(a, i, m)
        table[value] = i
    
    # Giant steps
    c = pow(a, n * (m - 2), m)  # a^(-n) mod m
    for i in range(n):
        value = (b * pow(c, i, m)) % m
        if value in table:
            return i * n + table[value]
    
    return None


def garner_algorithm(remainders: List[int], moduli: List[int]) -> int:
    """
    ガーナーのアルゴリズム（中国剰余定理の拡張）
    x ≡ r_i (mod m_i) となるような x を復元する
    中国剰余定理に比べて桁あふれを防ぎやすい
    
    Args:
        remainders: 余りのリスト [r_1, r_2, ..., r_n]
        moduli: 法のリスト [m_1, m_2, ..., m_n]
    
    Returns:
        全ての合同式を満たす値 x
    """
    n = len(remainders)
    
    # コンストラクタを計算
    coef = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i):
            coef[i][j] = mod_inverse(moduli[j], moduli[i])
            for k in range(j):
                coef[i][j] = (coef[i][j] * coef[i][k]) % moduli[i]
                coef[i][j] = (moduli[i] - coef[i][j]) if coef[i][j] > 0 else 0
    
    # 中国剰余定理を適用
    x = remainders[0]
    p = 1
    
    for i in range(1, n):
        p *= moduli[i - 1]
        
        # x_iを計算
        curr = remainders[i]
        for j in range(i):
            curr = (curr - x % moduli[i] * coef[i][j]) % moduli[i]
        curr = (curr * mod_inverse(p % moduli[i], moduli[i])) % moduli[i]
        
        x += p * curr
    
    return x


def montgomery_reduction_setup(mod: int) -> Tuple[int, int, int]:
    """
    モンゴメリリダクションのセットアップ
    
    Args:
        mod: 法
    
    Returns:
        (R, R_inverse, N_prime)のタプル
    """
    # Rはmodより大きい2の累乗
    bits = mod.bit_length()
    R = 1 << bits
    
    # R^(-1) mod mod
    R_inverse = mod_inverse(R % mod, mod)
    
    # R*R_inverse - mod*N_prime = 1 となるN_primeを計算
    _, _, N_prime = extended_gcd(mod, R)
    N_prime = (-N_prime) % R
    
    return (R, R_inverse, N_prime)


def to_montgomery(x: int, mod: int, R: int) -> int:
    """
    通常の整数をモンゴメリ表現に変換
    
    Args:
        x: 変換する整数
        mod: 法
        R: モンゴメリの定数（2^k形式）
    
    Returns:
        モンゴメリ表現でのx
    """
    return (x * R) % mod


def from_montgomery(x: int, mod: int, R_inverse: int) -> int:
    """
    モンゴメリ表現から通常の整数に変換
    
    Args:
        x: モンゴメリ表現の整数
        mod: 法
        R_inverse: モンゴメリ定数Rの逆元
    
    Returns:
        通常表現でのx
    """
    return (x * R_inverse) % mod


def montgomery_multiply(a: int, b: int, mod: int, R: int, N_prime: int) -> int:
    """
    モンゴメリ乗算：モンゴメリ表現での a * b
    
    Args:
        a: 第1の被乗数（モンゴメリ表現）
        b: 第2の被乗数（モンゴメリ表現）
        mod: 法
        R: モンゴメリの定数（2^k形式）
        N_prime: モンゴメリリダクションに必要な値
    
    Returns:
        モンゴメリ表現での a * b の結果
    """
    t = a * b
    m = (t * N_prime) % R
    t_reduced = (t + m * mod) // R
    
    if t_reduced >= mod:
        return t_reduced - mod
    else:
        return t_reduced


# 使用例
def example():
    # モジュラ逆元
    a, m = 3, 11
    print(f"{a}のmod {m}での逆元: {mod_inverse(a, m)}")
    
    a, p = 3, 11  # pは素数
    print(f"{a}のmod {p}での逆元（フェルマーの小定理）: {mod_inverse_fermat(a, p)}")
    
    # 中国剰余定理
    remainders = [3, 4, 5]
    moduli = [5, 7, 11]
    result = chinese_remainder_theorem(remainders, moduli)
    if result:
        x, lcm = result
        print(f"中国剰余定理の解: x ≡ {x} (mod {lcm})")
        print(f"チェック: x mod 5 = {x % 5}, x mod 7 = {x % 7}, x mod 11 = {x % 11}")
    else:
        print("解がありません。")
    
    # モジュラ算術
    a, b, mod = 10, 3, 7
    print(f"({a} + {b}) mod {mod} = {mod_add(a, b, mod)}")
    print(f"({a} - {b}) mod {mod} = {mod_sub(a, b, mod)}")
    print(f"({a} * {b}) mod {mod} = {mod_mul(a, b, mod)}")
    print(f"({a} / {b}) mod {mod} = {mod_div(a, b, mod)}")
    
    # モジュラ累乗
    base, exponent, mod = 2, 10, 1000000007
    print(f"{base}^{exponent} mod {mod} = {mod_pow(base, exponent, mod)}")
    
    # モジュラ平方根
    n, p = 5, 13
    sqrt_result = mod_sqrt(n, p)
    if sqrt_result is not None:
        print(f"{n}のmod {p}での平方根: {sqrt_result}")
        print(f"チェック: {sqrt_result}^2 mod {p} = {pow(sqrt_result, 2, p)}")
    else:
        print(f"{n}はmod {p}での平方剰余ではありません。")
    
    # 離散対数
    a, b, m = 2, 8, 11
    log_result = discrete_log(a, b, m)
    if log_result is not None:
        print(f"離散対数: {a}^x ≡ {b} (mod {m}) の解は x = {log_result}")
        print(f"チェック: {a}^{log_result} mod {m} = {pow(a, log_result, m)}")
    else:
        print(f"離散対数: {a}^x ≡ {b} (mod {m}) には解がありません。")
    
    # モンゴメリリダクション
    mod = 11
    R, R_inverse, N_prime = montgomery_reduction_setup(mod)
    print(f"モンゴメリリダクションの定数: R = {R}, R^(-1) = {R_inverse}, N' = {N_prime}")
    
    x = 5
    mont_x = to_montgomery(x, mod, R)
    print(f"{x}のモンゴメリ表現: {mont_x}")
    print(f"モンゴメリ表現からの復元: {from_montgomery(mont_x, mod, R_inverse)}")


if __name__ == "__main__":
    example()
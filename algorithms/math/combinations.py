"""
組み合わせ論アルゴリズム集 (Combinatorics Algorithms)
- 二項係数、組み合わせ、順列などの組み合わせ論に関する計算
- 主な実装:
  - 二項係数計算（愚直、パスカルの三角形、逆元）
  - 組み合わせクラス（前計算による高速化）
  - 順列、重複組み合わせ、重複順列
- 計算量:
  - 愚直法: O(min(k, n-k))
  - パスカルの三角形: O(n^2) 前計算、O(1) クエリ
  - mod逆元を使った方法: O(n) 前計算、O(1) クエリ
"""

from typing import List, Tuple, Dict, Optional
import math


def binomial_coefficient_naive(n: int, k: int) -> int:
    """
    二項係数 C(n,k) の計算（愚直な方法）
    
    Args:
        n: 要素の総数
        k: 選択する要素の数
    
    Returns:
        nCk の値
    """
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    
    # 計算量を減らすために k と n-k の小さい方を使う
    k = min(k, n - k)
    
    result = 1
    for i in range(k):
        result = result * (n - i) // (i + 1)
    
    return result


def pascal_triangle(n: int) -> List[List[int]]:
    """
    パスカルの三角形を生成（n行目まで）
    
    Args:
        n: 生成する行数（0からn-1行目まで）
    
    Returns:
        パスカルの三角形（2次元配列）
    """
    triangle = [[1]]
    
    for i in range(1, n):
        row = [1]
        for j in range(1, i):
            row.append(triangle[i-1][j-1] + triangle[i-1][j])
        row.append(1)
        triangle.append(row)
    
    return triangle


def binomial_coefficient_pascal(n: int, k: int, triangle: Optional[List[List[int]]] = None) -> int:
    """
    二項係数 C(n,k) の計算（パスカルの三角形を使用）
    
    Args:
        n: 要素の総数
        k: 選択する要素の数
        triangle: 事前に計算されたパスカルの三角形（オプション）
    
    Returns:
        nCk の値
    """
    if k < 0 or k > n:
        return 0
    
    if triangle is not None and n < len(triangle):
        return triangle[n][k]
    
    # 三角形が与えられていない、またはサイズが足りない場合は計算
    triangle = pascal_triangle(n + 1)
    return triangle[n][k]


class Combinatorics:
    """組み合わせ計算のためのクラス（mod演算対応、前計算による高速化）"""
    
    def __init__(self, n: int, mod: int = 10**9 + 7):
        """
        組み合わせ計算器を初期化（階乗と逆元を前計算）
        
        Args:
            n: 前計算する最大値
            mod: 剰余を取る値（デフォルト: 10^9 + 7）
        """
        self.mod = mod
        self.fact = [1] * (n + 1)  # 階乗: fact[i] = i!
        self.inv_fact = [1] * (n + 1)  # 逆階乗: inv_fact[i] = (i!)^(-1)
        
        # 階乗を計算
        for i in range(1, n + 1):
            self.fact[i] = (self.fact[i - 1] * i) % mod
        
        # 逆元を計算（フェルマーの小定理を使用）
        self.inv_fact[n] = pow(self.fact[n], mod - 2, mod)
        
        # 他の逆階乗を計算（大きい方から小さい方へ）
        for i in range(n, 0, -1):
            self.inv_fact[i - 1] = (self.inv_fact[i] * i) % mod
    
    def combination(self, n: int, k: int) -> int:
        """
        二項係数 C(n,k) を計算
        
        Args:
            n: 要素の総数
            k: 選択する要素の数
        
        Returns:
            nCk mod p の値
        """
        if k < 0 or k > n:
            return 0
        return (self.fact[n] * self.inv_fact[k] % self.mod * self.inv_fact[n - k] % self.mod)
    
    def permutation(self, n: int, k: int) -> int:
        """
        順列 P(n,k) を計算
        
        Args:
            n: 要素の総数
            k: 並べる要素の数
        
        Returns:
            nPk mod p の値
        """
        if k < 0 or k > n:
            return 0
        return (self.fact[n] * self.inv_fact[n - k] % self.mod)
    
    def homogeneous_combination(self, n: int, k: int) -> int:
        """
        重複組み合わせ H(n,k) = C(n+k-1,k) を計算
        
        Args:
            n: 要素の種類数
            k: 選択する要素の数（重複可）
        
        Returns:
            nHk mod p の値
        """
        return self.combination(n + k - 1, k)
    
    def stirling_number_second(self, n: int, k: int) -> int:
        """
        第2種スターリング数 S(n,k) を計算
        n個の区別できる物をk個の区別できないグループに分ける方法の数
        
        Args:
            n: 物の数
            k: グループ数
        
        Returns:
            S(n,k) mod p の値
        """
        if n == 0 and k == 0:
            return 1
        if n == 0 or k == 0:
            return 0
        
        result = 0
        for i in range(k + 1):
            coef = self.combination(k, i)
            term = (pow(k - i, n, self.mod) * coef) % self.mod
            if i % 2 == 1:
                result = (result - term) % self.mod
            else:
                result = (result + term) % self.mod
        
        result = (result * self.inv_fact[k]) % self.mod
        return result
    
    def catalan(self, n: int) -> int:
        """
        カタラン数 Cat(n) を計算
        
        Args:
            n: インデックス
        
        Returns:
            Cat(n) mod p の値
        """
        return (self.combination(2 * n, n) * pow(n + 1, -1, self.mod)) % self.mod


def multinomial_coefficient(counts: List[int]) -> int:
    """
    多項係数 (n; k1, k2, ..., km) の計算
    
    Args:
        counts: 各要素の出現回数のリスト
    
    Returns:
        多項係数の値
    """
    n = sum(counts)
    result = 1
    
    for k in counts:
        result *= math.comb(n, k)
        n -= k
    
    return result


def multinomial_coefficient_mod(counts: List[int], mod: int) -> int:
    """
    多項係数 (n; k1, k2, ..., km) のmod計算
    
    Args:
        counts: 各要素の出現回数のリスト
        mod: 剰余を取る値
    
    Returns:
        多項係数 mod p の値
    """
    n = sum(counts)
    combi = Combinatorics(n, mod)
    
    result = combi.fact[n]
    for k in counts:
        result = (result * combi.inv_fact[k]) % mod
    
    return result


def next_permutation(arr: List[int]) -> bool:
    """
    与えられた配列の次の辞書順の順列を生成（in-place）
    
    Args:
        arr: 順列を表す配列
    
    Returns:
        次の順列が存在する場合はTrue、存在しない（最後の順列）場合はFalse
    """
    n = len(arr)
    # 後ろから逆転ポイント（arr[i] < arr[i+1]の位置）を探す
    i = n - 2
    while i >= 0 and arr[i] >= arr[i + 1]:
        i -= 1
    
    if i < 0:
        # 既に最後の順列の場合は元の配列を逆順にして最初の順列に戻す
        arr.reverse()
        return False
    
    # arr[i]より大きい値の中で最小のものを後方から探す
    j = n - 1
    while arr[j] <= arr[i]:
        j -= 1
    
    # 交換して後半を逆順にする
    arr[i], arr[j] = arr[j], arr[i]
    arr[i+1:] = reversed(arr[i+1:])
    return True


def prev_permutation(arr: List[int]) -> bool:
    """
    与えられた配列の前の辞書順の順列を生成（in-place）
    
    Args:
        arr: 順列を表す配列
    
    Returns:
        前の順列が存在する場合はTrue、存在しない（最初の順列）場合はFalse
    """
    n = len(arr)
    # 後ろから逆転ポイント（arr[i] > arr[i+1]の位置）を探す
    i = n - 2
    while i >= 0 and arr[i] <= arr[i + 1]:
        i -= 1
    
    if i < 0:
        # 既に最初の順列の場合は元の配列を逆順にして最後の順列に戻す
        arr.reverse()
        return False
    
    # arr[i]より小さい値の中で最大のものを後方から探す
    j = n - 1
    while arr[j] >= arr[i]:
        j -= 1
    
    # 交換して後半を逆順にする
    arr[i], arr[j] = arr[j], arr[i]
    arr[i+1:] = reversed(arr[i+1:])
    return True


def all_permutations(elements: List) -> List[List]:
    """
    与えられた要素のすべての順列を生成
    
    Args:
        elements: 順列を生成する要素のリスト
    
    Returns:
        すべての順列のリスト
    """
    from itertools import permutations
    return list(permutations(elements))


def all_combinations(elements: List, r: int) -> List[List]:
    """
    与えられた要素からr個選ぶすべての組み合わせを生成
    
    Args:
        elements: 組み合わせを生成する要素のリスト
        r: 選択する要素の数
    
    Returns:
        すべての組み合わせのリスト
    """
    from itertools import combinations
    return list(combinations(elements, r))


# 使用例
def example():
    # 二項係数の計算（愚直な方法）
    n, k = 10, 3
    print(f"C({n},{k}) = {binomial_coefficient_naive(n, k)}")
    
    # パスカルの三角形を使った二項係数計算
    triangle = pascal_triangle(11)  # 0行目〜10行目まで
    print(f"パスカルの三角形でのC({n},{k}) = {binomial_coefficient_pascal(n, k, triangle)}")
    
    # 前計算を使った組み合わせ計算
    comb = Combinatorics(100)  # 100までの値を前計算
    print(f"組み合わせクラスでのC({n},{k}) = {comb.combination(n, k)}")
    
    # 順列
    print(f"P({n},{k}) = {comb.permutation(n, k)}")
    
    # 重複組み合わせ
    print(f"H({n},{k}) = {comb.homogeneous_combination(n, k)}")
    
    # カタラン数
    n = 4
    print(f"Cat({n}) = {comb.catalan(n)}")
    
    # 第2種スターリング数
    n, k = 6, 3
    print(f"S({n},{k}) = {comb.stirling_number_second(n, k)}")
    
    # 多項係数
    counts = [2, 3, 1]
    print(f"Multinomial({counts}) = {multinomial_coefficient(counts)}")
    
    # 次の順列
    arr = [1, 2, 3]
    print(f"元の順列: {arr}")
    next_permutation(arr)
    print(f"次の順列: {arr}")
    
    # すべての順列
    elements = [1, 2, 3]
    perms = all_permutations(elements)
    print(f"{elements}のすべての順列: {perms}")
    
    # すべての組み合わせ
    elements = [1, 2, 3, 4]
    r = 2
    combs = all_combinations(elements, r)
    print(f"{elements}から{r}個選ぶすべての組み合わせ: {combs}")


if __name__ == "__main__":
    example()
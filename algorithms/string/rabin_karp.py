"""
ラビン-カープ法（Rabin-Karp Algorithm）
- ハッシュ関数を利用した文字列検索アルゴリズム
- 主な用途:
  - テキスト内でのパターン検索
  - 複数のパターン検索
  - 文字列のプラグマリズム検出
- 特徴:
  - ローリングハッシュを使用
  - 高速な実装が可能
  - 複数パターンの検索に適している
- 計算量:
  - 平均ケース: O(n + m)（nはテキスト長、mはパターン長）
  - 最悪ケース: O(n * m)（ハッシュ衝突が多い場合）
"""

from typing import List


def rabin_karp(text: str, pattern: str, q: int = 101) -> List[int]:
    """
    ラビン-カープ法による文字列検索
    
    Args:
        text: 検索対象のテキスト
        pattern: 検索するパターン
        q: モジュロ演算の素数（ハッシュの衝突を減らす）
        
    Returns:
        List[int]: パターンが見つかった開始位置のリスト
    """
    n, m = len(text), len(pattern)
    if m == 0 or m > n:
        return []
    
    d = 256  # 文字セットのサイズ（ASCIIの場合）
    h = pow(d, m - 1, q)  # d^(m-1) % q
    p, t = 0, 0  # パターンとテキストウィンドウのハッシュ値
    
    # パターンと最初のウィンドウのハッシュを計算
    for i in range(m):
        p = (d * p + ord(pattern[i])) % q
        t = (d * t + ord(text[i])) % q
    
    result = []
    
    # テキスト全体をスライディングウィンドウで検索
    for i in range(n - m + 1):
        # ハッシュ値が一致した場合、文字列を比較して確認
        if p == t:
            # 実際に文字列を比較して確認（ハッシュ衝突対策）
            if text[i:i+m] == pattern:
                result.append(i)
        
        # 次のウィンドウのハッシュ値を計算（ローリングハッシュ）
        if i < n - m:
            # 古い文字を削除し、新しい文字を追加
            t = (d * (t - ord(text[i]) * h) + ord(text[i + m])) % q
            if t < 0:
                t += q  # 負の値になる場合に修正
    
    return result


def rabin_karp_multiple(text: str, patterns: List[str], q: int = 101) -> dict:
    """
    複数パターンを一度に検索するラビン-カープ法
    
    Args:
        text: 検索対象のテキスト
        patterns: 検索するパターンのリスト
        q: モジュロ演算の素数
        
    Returns:
        dict: パターンごとの出現位置のリスト
    """
    if not patterns:
        return {}
    
    # すべてのパターンの長さが同じと仮定
    # 異なる長さの場合は、長さごとにグループ化して処理する必要がある
    m = len(patterns[0])
    n = len(text)
    
    if n < m:
        return {pattern: [] for pattern in patterns}
    
    d = 256  # 文字セットのサイズ
    h = pow(d, m - 1, q)  # d^(m-1) % q
    
    # 各パターンのハッシュ値を計算
    pattern_hashes = {}
    for pattern in patterns:
        if len(pattern) != m:
            continue  # 長さが異なるパターンはスキップ
        
        p_hash = 0
        for char in pattern:
            p_hash = (d * p_hash + ord(char)) % q
        pattern_hashes[p_hash] = pattern
    
    result = {pattern: [] for pattern in patterns}
    
    # 最初のウィンドウのハッシュを計算
    t = 0
    for i in range(m):
        t = (d * t + ord(text[i])) % q
    
    # テキスト全体をスキャン
    for i in range(n - m + 1):
        # ハッシュ値が一致するパターンがあるか確認
        if t in pattern_hashes:
            pattern = pattern_hashes[t]
            # 実際に文字列を比較して確認
            if text[i:i+m] == pattern:
                result[pattern].append(i)
        
        # 次のウィンドウのハッシュ値を計算
        if i < n - m:
            t = (d * (t - ord(text[i]) * h) + ord(text[i + m])) % q
            if t < 0:
                t += q
    
    return result


class RollingHash:
    """
    複数のハッシュ値を使用したローリングハッシュ実装
    衝突確率を下げるために複数のモジュロを使用
    """
    
    def __init__(self, text: str, base: int = 256):
        """
        ローリングハッシュを初期化
        
        Args:
            text: 対象テキスト
            base: ハッシュのベース（通常は文字セットのサイズ）
        """
        self.text = text
        self.base = base
        self.n = len(text)
        
        # 2つの異なる大きな素数を使用
        self.mod1 = 10**9 + 7
        self.mod2 = 10**9 + 9
        
        # ハッシュ値の累積配列
        self.hash1 = [0] * (self.n + 1)
        self.hash2 = [0] * (self.n + 1)
        
        # baseの累乗値の累積配列
        self.pow1 = [1] * (self.n + 1)
        self.pow2 = [1] * (self.n + 1)
        
        # ハッシュ値と累乗値を計算
        for i in range(self.n):
            self.hash1[i+1] = (self.hash1[i] * self.base + ord(text[i])) % self.mod1
            self.hash2[i+1] = (self.hash2[i] * self.base + ord(text[i])) % self.mod2
            
            self.pow1[i+1] = (self.pow1[i] * self.base) % self.mod1
            self.pow2[i+1] = (self.pow2[i] * self.base) % self.mod2
    
    def get_hash(self, left: int, right: int) -> tuple:
        """
        テキストの部分文字列[left, right)のハッシュ値を取得
        
        Args:
            left: 開始位置（含む）
            right: 終了位置（含まない）
            
        Returns:
            tuple: 2つのハッシュ値のタプル
        """
        hash1 = (self.hash1[right] - self.hash1[left] * self.pow1[right - left]) % self.mod1
        if hash1 < 0:
            hash1 += self.mod1
            
        hash2 = (self.hash2[right] - self.hash2[left] * self.pow2[right - left]) % self.mod2
        if hash2 < 0:
            hash2 += self.mod2
            
        return (hash1, hash2)
    
    def find_pattern(self, pattern: str) -> List[int]:
        """
        テキスト内でパターンを検索
        
        Args:
            pattern: 検索するパターン
            
        Returns:
            List[int]: パターンが見つかった開始位置のリスト
        """
        m = len(pattern)
        if m == 0 or m > self.n:
            return []
        
        # パターンのハッシュ値を計算
        p_hash1, p_hash2 = 0, 0
        for i in range(m):
            p_hash1 = (p_hash1 * self.base + ord(pattern[i])) % self.mod1
            p_hash2 = (p_hash2 * self.base + ord(pattern[i])) % self.mod2
        
        result = []
        
        # テキスト内で検索
        for i in range(self.n - m + 1):
            hash1, hash2 = self.get_hash(i, i + m)
            if hash1 == p_hash1 and hash2 == p_hash2:
                # ダブルハッシュで一致した場合でも、念のため実際に比較
                if self.text[i:i+m] == pattern:
                    result.append(i)
        
        return result
    
    def longest_common_substring(self, other_text: str) -> tuple:
        """
        2つの文字列間で最長共通部分文字列を見つける
        
        Args:
            other_text: 比較する別のテキスト
            
        Returns:
            tuple: (最長共通部分文字列, 開始位置1, 開始位置2)
        """
        other_hash = RollingHash(other_text)
        
        # 二分探索で最長の共通部分文字列の長さを見つける
        def check(length):
            # 1つ目のテキストの全ての長さlengthの部分文字列のハッシュをセットに入れる
            hash_set = set()
            for i in range(self.n - length + 1):
                hash_set.add(self.get_hash(i, i + length))
            
            # 2つ目のテキストの部分文字列との一致を確認
            for i in range(len(other_text) - length + 1):
                curr_hash = other_hash.get_hash(i, i + length)
                if curr_hash in hash_set:
                    # ハッシュが一致したら実際に文字列比較
                    for j in range(self.n - length + 1):
                        if self.get_hash(j, j + length) == curr_hash and self.text[j:j+length] == other_text[i:i+length]:
                            return (j, i)  # 開始位置を返す
            return (-1, -1)
        
        left, right = 0, min(self.n, len(other_text)) + 1
        start1, start2 = -1, -1
        
        while left < right:
            mid = (left + right) // 2
            pos1, pos2 = check(mid)
            
            if pos1 != -1:  # 長さmidの共通部分文字列が存在する
                left = mid + 1
                start1, start2 = pos1, pos2
            else:
                right = mid
        
        length = left - 1
        if length == 0:
            return ("", -1, -1)
        
        return (self.text[start1:start1+length], start1, start2)


# 使用例
def example():
    print("===== 基本的なラビン-カープ法 =====")
    text = "ABABABABCABABAB"
    pattern = "ABABC"
    
    occurrences = rabin_karp(text, pattern)
    if occurrences:
        print(f"パターン '{pattern}' は位置 {occurrences} で見つかりました")
    else:
        print(f"パターン '{pattern}' は見つかりませんでした")
    
    print("\n===== 複数パターンの検索 =====")
    patterns = ["AB", "ABABC", "BAB"]
    result = rabin_karp_multiple(text, patterns)
    
    for pattern, positions in result.items():
        if positions:
            print(f"パターン '{pattern}' は位置 {positions} で見つかりました")
        else:
            print(f"パターン '{pattern}' は見つかりませんでした")
    
    print("\n===== ローリングハッシュの応用 =====")
    text1 = "ABABABABCABABAB"
    text2 = "XYZABABCMNOPQ"
    
    rolling_hash = RollingHash(text1)
    occurrences = rolling_hash.find_pattern("ABABC")
    print(f"パターン 'ABABC' は位置 {occurrences} で見つかりました")
    
    lcs, pos1, pos2 = rolling_hash.longest_common_substring(text2)
    print(f"最長共通部分文字列: '{lcs}'")
    print(f"text1の位置: {pos1}, text2の位置: {pos2}")


if __name__ == "__main__":
    example()
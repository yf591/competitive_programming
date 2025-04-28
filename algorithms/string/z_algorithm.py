"""
Z アルゴリズム（Z Algorithm）
- 文字列の各位置で、元の文字列との最長共通接頭辞の長さを線形時間で計算するアルゴリズム
- 主な用途:
  - 文字列の中でのパターン検索
  - 文字列の周期性の検出
  - 最長共通接頭辞の計算
- 特徴:
  - 前処理の結果を再利用するテクニックで高速化
  - 線形時間での実装が可能
- 計算量:
  - O(n)（nは文字列長）
"""

from typing import List


def z_function(s: str) -> List[int]:
    """
    文字列のZアルゴリズムを実行し、Z配列を返す
    
    Args:
        s: 入力文字列
        
    Returns:
        Z配列: Z[i]は、sとs[i:]の最長共通接頭辞の長さ
    """
    n = len(s)
    z = [0] * n
    
    # 最初の要素は特別扱い（全体の文字列長）
    z[0] = n
    
    # Z配列を計算
    l, r = 0, 0  # [l, r]は最右の共通接頭辞区間
    for i in range(1, n):
        # 現在のiが[l,r]区間内にある場合、以前の計算結果を再利用
        if i <= r:
            # min(すでに計算した値, 区間の右端までの距離)
            z[i] = min(r - i + 1, z[i - l])
        
        # 共通接頭辞の長さを直接計算して拡張
        while i + z[i] < n and s[z[i]] == s[i + z[i]]:
            z[i] += 1
        
        # 最右の共通接頭辞区間を更新
        if i + z[i] - 1 > r:
            l, r = i, i + z[i] - 1
    
    return z


def string_search_z(text: str, pattern: str) -> List[int]:
    """
    Zアルゴリズムを使用してパターンの出現位置を見つける
    
    Args:
        text: 検索対象のテキスト
        pattern: 検索するパターン
        
    Returns:
        List[int]: パターンが見つかった開始位置のリスト
    """
    if not pattern:
        return list(range(len(text) + 1))
    
    if not text:
        return []
    
    # パターン + 特殊文字 + テキストを結合
    # 特殊文字はパターンとテキスト中に存在しないものを使用
    concatenated = pattern + '$' + text
    z_values = z_function(concatenated)
    
    # パターンの長さと一致するZ値の位置を記録
    pattern_len = len(pattern)
    result = []
    
    for i in range(pattern_len + 1, len(concatenated)):
        if z_values[i] == pattern_len:
            # パターン開始位置に変換（テキスト内での位置）
            result.append(i - pattern_len - 1)
    
    return result


def longest_palindrome_substring(s: str) -> str:
    """
    Zアルゴリズムを使用して最長回文部分文字列を見つける
    （Manacherのアルゴリズムほど効率的ではないが、Zアルゴリズムの応用例として）
    
    Args:
        s: 入力文字列
        
    Returns:
        最長回文部分文字列
    """
    if not s:
        return ""
    
    n = len(s)
    best_len = 0
    best_center = 0
    
    # すべての可能な中心について検証
    for center in range(n):
        # 奇数長の回文
        left = center
        right = center
        while left >= 0 and right < n and s[left] == s[right]:
            curr_len = right - left + 1
            if curr_len > best_len:
                best_len = curr_len
                best_center = center
            left -= 1
            right += 1
        
        # 偶数長の回文
        left = center
        right = center + 1
        while left >= 0 and right < n and s[left] == s[right]:
            curr_len = right - left + 1
            if curr_len > best_len:
                best_len = curr_len
                best_center = center + 0.5  # 小数点で偶数長を示す
            left -= 1
            right += 1
    
    # 最長回文の開始と終了位置を計算
    if best_center == int(best_center):  # 奇数長
        center = int(best_center)
        radius = best_len // 2
        return s[center - radius:center + radius + 1]
    else:  # 偶数長
        center = int(best_center)
        radius = best_len // 2
        return s[center - radius + 1:center + radius + 1]


def string_periods(s: str) -> List[int]:
    """
    Zアルゴリズムを使用して文字列の全ての周期を見つける
    
    Args:
        s: 入力文字列
        
    Returns:
        List[int]: 文字列の全ての周期のリスト
    """
    n = len(s)
    z = z_function(s)
    periods = []
    
    # 周期の定義: p is a period of s if s[0:n-p] == s[p:n]
    for p in range(1, n):
        if z[p] + p >= n:  # 文字列の残りの部分が一致
            periods.append(p)
    
    return periods


def longest_common_prefix(strings: List[str]) -> str:
    """
    複数の文字列の最長共通接頭辞を見つける
    
    Args:
        strings: 文字列のリスト
        
    Returns:
        最長共通接頭辞
    """
    if not strings:
        return ""
    
    if len(strings) == 1:
        return strings[0]
    
    # 1つめの文字列と他の文字列の共通接頭辞長を計算
    min_prefix_len = float('inf')
    
    for i in range(1, len(strings)):
        # 1つめの文字列 + '$' + 比較する文字列
        concatenated = strings[0] + '$' + strings[i]
        z = z_function(concatenated)
        
        # 2つめの文字列の開始位置
        start_pos = len(strings[0]) + 1
        prefix_len = 0
        
        # 共通接頭辞長は、比較する文字列の最初の文字のZ値
        if start_pos < len(concatenated):
            prefix_len = z[start_pos]
        
        min_prefix_len = min(min_prefix_len, prefix_len)
    
    return strings[0][:min_prefix_len]


# 使用例
def example():
    print("===== Z アルゴリズム =====")
    s = "ababab"
    z = z_function(s)
    print(f"文字列: {s}")
    print(f"Z配列: {z}")
    
    print("\n===== 文字列検索 =====")
    text = "ababcababab"
    pattern = "abab"
    
    occurrences = string_search_z(text, pattern)
    if occurrences:
        print(f"パターン '{pattern}' は位置 {occurrences} で見つかりました")
    else:
        print(f"パターン '{pattern}' は見つかりませんでした")
    
    print("\n===== 回文検出 =====")
    palindrome_text = "abacaba"
    longest_palindrome = longest_palindrome_substring(palindrome_text)
    print(f"文字列 '{palindrome_text}' の最長回文部分文字列: '{longest_palindrome}'")
    
    print("\n===== 文字列の周期 =====")
    periodic = "abababab"
    periods = string_periods(periodic)
    print(f"文字列 '{periodic}' の周期: {periods}")
    
    print("\n===== 最長共通接頭辞 =====")
    string_list = ["abcdef", "abcxyz", "abcd123"]
    lcp = longest_common_prefix(string_list)
    print(f"文字列 {string_list} の最長共通接頭辞: '{lcp}'")


if __name__ == "__main__":
    example()
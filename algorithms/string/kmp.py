"""
KMP法（Knuth-Morris-Pratt Algorithm）
- 文字列検索のための効率的なアルゴリズム
- テキスト内でパターンが出現する全ての位置を線形時間で探索できる
- 部分一致テーブルを利用して無駄な比較を省略

特徴:
- 時間計算量: O(N + M) （Nはテキストの長さ、Mはパターンの長さ）
- 空間計算量: O(M) （パターンの長さ分の部分一致テーブル）

応用例:
- 文字列内の部分文字列検索
- 周期的なパターンの検出
- 最長の接頭辞かつ接尾辞の発見
"""

from typing import List


def compute_lps(pattern: str) -> List[int]:
    """
    KMP法で使用する部分一致テーブル（LPS: Longest Prefix Suffix）を計算する
    
    Args:
        pattern: 検索パターン
        
    Returns:
        List[int]: 各位置での最長の接頭辞かつ接尾辞の長さ
    """
    m = len(pattern)
    lps = [0] * m  # 部分一致テーブル
    
    # 最初の位置は常に0
    length = 0  # 前の位置までの最長の接頭辞かつ接尾辞の長さ
    i = 1
    
    while i < m:
        if pattern[i] == pattern[length]:
            # 文字が一致する場合、長さを増やして記録
            length += 1
            lps[i] = length
            i += 1
        else:
            if length != 0:
                # 不一致の場合、前の部分一致結果を使って長さを調整
                length = lps[length - 1]
            else:
                # 長さが0の場合、この位置のlpsは0
                lps[i] = 0
                i += 1
    
    return lps


def kmp_search(text: str, pattern: str) -> List[int]:
    """
    KMP法によるパターン検索を行う
    
    Args:
        text: 検索対象のテキスト
        pattern: 検索するパターン
        
    Returns:
        List[int]: テキスト内でパターンが出現する全ての開始位置（0-indexed）
    """
    if not pattern:
        return []  # パターンが空の場合
    
    n, m = len(text), len(pattern)
    if m > n:
        return []  # パターンがテキストより長い場合
    
    # 部分一致テーブルの計算
    lps = compute_lps(pattern)
    
    result = []  # パターンが見つかった位置を格納
    
    i = 0  # テキストのインデックス
    j = 0  # パターンのインデックス
    
    while i < n:
        # 文字が一致する場合
        if pattern[j] == text[i]:
            i += 1
            j += 1
        
        # パターン全体が一致した場合
        if j == m:
            result.append(i - j)  # 開始位置を記録
            j = lps[j - 1]  # 次の検索開始位置を設定
        
        # 文字が不一致かつインデックスがまだテキスト内の場合
        elif i < n and pattern[j] != text[i]:
            if j != 0:
                j = lps[j - 1]  # 部分一致テーブルを使って次の比較位置を設定
            else:
                i += 1  # パターンの先頭からの比較をやり直し
    
    return result


def z_function(s: str) -> List[int]:
    """
    Z関数（各位置iから始まる最長の接頭辞の長さを計算）
    
    Args:
        s: 文字列
        
    Returns:
        List[int]: Z配列（Z[i]はS[0...i-1]とS[i...n-1]の最長共通接頭辞）
    """
    n = len(s)
    z = [0] * n
    z[0] = n  # Z[0]は文字列の長さに等しい（全体が共通）
    
    l, r = 0, 0  # 現在見ているZ-boxの左端と右端
    for i in range(1, n):
        if i <= r:  # 既に計算したZ-boxの範囲内
            z[i] = min(r - i + 1, z[i - l])
        
        # Z[i]を直接計算または延長
        while i + z[i] < n and s[z[i]] == s[i + z[i]]:
            z[i] += 1
        
        # Z-boxの更新
        if i + z[i] - 1 > r:
            l, r = i, i + z[i] - 1
    
    return z


# 使用例
def example():
    # KMP法による文字列検索
    text = "ABABDABACDABABCABAB"
    pattern = "ABABCABAB"
    
    print(f"テキスト: {text}")
    print(f"パターン: {pattern}")
    
    positions = kmp_search(text, pattern)
    if positions:
        print(f"パターンが見つかった位置（0-indexed）: {positions}")
        # 位置を可視化
        for pos in positions:
            print(" " * pos + pattern)
            print(text)
            print("-" * len(text))
    else:
        print("パターンは見つかりませんでした")
    
    # 部分一致テーブル
    lps = compute_lps(pattern)
    print(f"\nパターン {pattern} の部分一致テーブル:")
    for i, val in enumerate(pattern):
        print(f"{val:3}", end="")
    print()
    for i, val in enumerate(lps):
        print(f"{val:3}", end="")
    print()
    
    # Z関数
    s = "abacaba"
    z = z_function(s)
    print(f"\n文字列 {s} のZ関数:")
    print(" ".join(map(str, z)))


if __name__ == "__main__":
    example()
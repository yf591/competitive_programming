"""
マナカーのアルゴリズム（Manacher's Algorithm）
- 線形時間で文字列の全ての回文部分文字列を見つけるアルゴリズム
- 主な用途:
  - 最長回文部分文字列の検索
  - 各位置を中心とする最大回文の長さの計算
- 特徴:
  - 前計算の結果を再利用するテクニックで高速化
  - 奇数・偶数長の回文を同時に扱える
- 計算量:
  - O(n)（nは文字列長）
"""

from typing import List, Tuple


def manacher(s: str) -> List[int]:
    """
    マナカーのアルゴリズムを実行し、各位置での最大回文半径を計算
    
    Args:
        s: 入力文字列
        
    Returns:
        List[int]: 半径配列（p[i]はi位置を中心とする回文の半径）
    """
    # 文字間に特殊文字を挿入して、奇数長の文字列に変換
    t = '#' + '#'.join(s) + '#'
    n = len(t)
    
    # p[i]はt[i]を中心とする回文の半径（自身を含む）
    p = [0] * n
    
    c = 0  # 現在の回文の中心
    r = 0  # 現在の回文の右端（c + p[c] - 1）
    
    for i in range(n):
        # iがrより小さい場合、対応する位置の値を利用
        if i < r:
            mirror = 2 * c - i  # iのcに対する対称位置
            p[i] = min(r - i, p[mirror])
        
        # 中心iの周りを拡張して、最大回文を見つける
        left, right = i - (p[i] + 1), i + (p[i] + 1)
        while left >= 0 and right < n and t[left] == t[right]:
            p[i] += 1
            left -= 1
            right += 1
        
        # 現在の回文が右端を超える場合、中心と右端を更新
        if i + p[i] > r:
            c = i
            r = i + p[i]
    
    return p


def longest_palindrome_substring(s: str) -> str:
    """
    マナカーのアルゴリズムを使用して最長回文部分文字列を見つける
    
    Args:
        s: 入力文字列
        
    Returns:
        str: 最長回文部分文字列
    """
    if not s:
        return ""
    
    # 回文半径の計算
    p = manacher(s)
    
    # 変換後の文字列
    t = '#' + '#'.join(s) + '#'
    
    # 最大の回文半径とその中心を見つける
    max_len = max(p)
    center = p.index(max_len)
    
    # 元の文字列における開始位置を計算
    # center - max_len は回文の左端、2で割ると元の文字列での位置
    start = (center - max_len) // 2
    
    # 元の文字列における回文の長さを計算
    # 元の文字列での回文の長さは、変換後の長さ/2（切り上げ）
    length = max_len
    
    return s[start:start + length]


def all_palindrome_substrings(s: str) -> List[str]:
    """
    マナカーのアルゴリズムを使用して全ての回文部分文字列を見つける
    
    Args:
        s: 入力文字列
        
    Returns:
        List[str]: すべての回文部分文字列のリスト（重複なし）
    """
    if not s:
        return []
    
    # 回文半径の計算
    p = manacher(s)
    
    # 変換後の文字列
    t = '#' + '#'.join(s) + '#'
    n = len(t)
    
    # 回文部分文字列を格納するセット（重複を避けるため）
    palindromes = set()
    
    for i in range(n):
        # 各中心位置について、可能なすべての回文を生成
        for j in range(1, p[i] + 1):
            # 変換後の文字列での回文の範囲: [i-j+1, i+j-1]
            # 元の文字列での開始位置と長さを計算
            start = (i - j) // 2
            
            # 偶数位置なら長さj、奇数位置ならj+1（変換後の文字列では）
            length = j
            
            # 元の文字列の範囲を計算
            palindrome = s[start:start + length]
            palindromes.add(palindrome)
    
    # セットをリストに変換して返す（長さでソート）
    return sorted(list(palindromes), key=len)


def count_palindrome_substrings(s: str) -> int:
    """
    マナカーのアルゴリズムを使用して回文部分文字列の数を数える
    
    Args:
        s: 入力文字列
        
    Returns:
        int: 回文部分文字列の総数（重複を含む）
    """
    if not s:
        return 0
    
    # 回文半径の計算
    p = manacher(s)
    
    # 変換後の文字列の長さ
    n = len('#' + '#'.join(s) + '#')
    
    # 回文の総数をカウント
    count = 0
    for i in range(n):
        # p[i]は回文の半径なので、可能な回文の数はp[i]
        # ただし、空文字列は除外するため、開始長さは1から
        count += (p[i] + 1) // 2
    
    return count


def longest_palindrome_subsequence(s: str) -> str:
    """
    最長回文部分列（subsequence）を見つける
    注: マナカーのアルゴリズムは部分文字列（substring）には効果的ですが、
        部分列（subsequence）には直接適用できないため、動的計画法を使用
    
    Args:
        s: 入力文字列
        
    Returns:
        str: 最長回文部分列
    """
    if not s:
        return ""
    
    n = len(s)
    # dp[i][j] = sの部分列s[i:j+1]の最長回文部分列の長さ
    dp = [[0] * n for _ in range(n)]
    
    # 全ての1文字は回文
    for i in range(n):
        dp[i][i] = 1
    
    # 部分列の再構築のための配列
    path = [[0] * n for _ in range(n)]
    
    # 長さ2以上の部分列について
    for cl in range(2, n + 1):  # 部分列の長さ
        for i in range(n - cl + 1):  # 開始位置
            j = i + cl - 1  # 終了位置
            
            if s[i] == s[j] and cl == 2:
                dp[i][j] = 2
                path[i][j] = 1  # 両端の文字を取る
            elif s[i] == s[j]:
                dp[i][j] = dp[i + 1][j - 1] + 2
                path[i][j] = 1  # 両端の文字を取る
            else:
                if dp[i + 1][j] > dp[i][j - 1]:
                    dp[i][j] = dp[i + 1][j]
                    path[i][j] = 2  # 左側の文字をスキップ
                else:
                    dp[i][j] = dp[i][j - 1]
                    path[i][j] = 3  # 右側の文字をスキップ
    
    # 最長回文部分列の再構築
    result = []
    reconstruct_palindrome(path, s, 0, n - 1, result)
    
    return ''.join(result)


def reconstruct_palindrome(path: List[List[int]], s: str, i: int, j: int, result: List[str]):
    """最長回文部分列を再構築するヘルパー関数"""
    if i > j:
        return
    
    if i == j:
        result.append(s[i])
    else:
        if path[i][j] == 1:  # 両端の文字を取る
            result.append(s[i])
            # 内側の部分列があれば再帰
            if i + 1 <= j - 1:
                reconstruct_palindrome(path, s, i + 1, j - 1, result)
            # 中央の1文字だけの場合は追加しない（すでに左端を追加済み）
            elif i + 1 == j:
                pass
            else:
                result.append(s[j])
        elif path[i][j] == 2:  # 左側の文字をスキップ
            reconstruct_palindrome(path, s, i + 1, j, result)
        else:  # 右側の文字をスキップ
            reconstruct_palindrome(path, s, i, j - 1, result)


def palindrome_pairs(words: List[str]) -> List[Tuple[int, int]]:
    """
    リスト内の単語のペアで、連結するとパリンドロームになるものを見つける
    
    Args:
        words: 単語のリスト
        
    Returns:
        List[Tuple[int, int]]: パリンドロームを形成する単語のインデックスペア
    """
    result = []
    word_dict = {word: i for i, word in enumerate(words)}
    
    for i, word in enumerate(words):
        for j in range(len(word) + 1):
            # word[:j] + word[j:]という分割を考える
            
            # 前半が回文かチェック
            prefix = word[:j]
            if is_palindrome(prefix):
                # 後半の反転が辞書にあるか確認
                reversed_suffix = word[j:][::-1]
                if reversed_suffix in word_dict and word_dict[reversed_suffix] != i:
                    # (reversed_suffix + word)はパリンドローム
                    result.append((word_dict[reversed_suffix], i))
            
            # 後半が回文でj>0（空文字でない）場合
            suffix = word[j:]
            if j > 0 and is_palindrome(suffix):
                # 前半の反転が辞書にあるか
                reversed_prefix = word[:j][::-1]
                if reversed_prefix in word_dict and word_dict[reversed_prefix] != i:
                    # (word + reversed_prefix)はパリンドローム
                    result.append((i, word_dict[reversed_prefix]))
    
    return result


def is_palindrome(s: str) -> bool:
    """文字列が回文かどうかをチェック"""
    return s == s[::-1]


# 使用例
def example():
    print("===== マナカーのアルゴリズム =====")
    s = "babadada"
    p = manacher(s)
    
    # 変換後の文字列を表示
    t = '#' + '#'.join(s) + '#'
    print(f"元の文字列: {s}")
    print(f"変換後の文字列: {t}")
    print(f"回文の半径: {p}")
    
    print("\n===== 最長回文部分文字列 =====")
    longest = longest_palindrome_substring(s)
    print(f"最長回文部分文字列: {longest}")
    
    print("\n===== すべての回文部分文字列 =====")
    all_palindromes = all_palindrome_substrings("abac")
    print(f"回文部分文字列の数: {len(all_palindromes)}")
    print(f"回文部分文字列: {all_palindromes}")
    
    print("\n===== 回文部分文字列の数 =====")
    count = count_palindrome_substrings("aaa")
    print(f"回文部分文字列の数（重複含む）: {count}")
    
    print("\n===== 最長回文部分列 =====")
    s = "ACGTGTCAAAATCG"
    lps = longest_palindrome_subsequence(s)
    print(f"文字列: {s}")
    print(f"最長回文部分列: {lps}")
    
    print("\n===== 回文ペア =====")
    words = ["abcd", "dcba", "lls", "s", "sssll"]
    pairs = palindrome_pairs(words)
    print(f"回文を形成するペアのインデックス: {pairs}")
    for i, j in pairs:
        print(f"'{words[i]}' + '{words[j]}' = '{words[i] + words[j]}'")


if __name__ == "__main__":
    example()
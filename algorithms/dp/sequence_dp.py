"""
配列・文字列のDP（Sequence DP）
- 配列や文字列に対する動的計画法のアルゴリズム
- 含まれるアルゴリズム:
  - 最長増加部分列（LIS）
  - 最長共通部分列（LCS）
  - 編集距離（レーベンシュタイン距離）
- 計算量:
  - LIS: O(n²) または O(n log n)（二分探索使用時）
  - LCS: O(nm)
  - 編集距離: O(nm)
"""

from typing import List, Tuple, Optional
import bisect


def longest_increasing_subsequence(arr: List[int]) -> int:
    """
    最長増加部分列（LIS）の長さを求める（O(n²)のDP解法）
    
    Args:
        arr: 対象の配列
        
    Returns:
        最長増加部分列の長さ
    """
    if not arr:
        return 0
        
    n = len(arr)
    dp = [1] * n  # dp[i]はarr[i]で終わる最長増加部分列の長さ
    
    for i in range(1, n):
        for j in range(i):
            if arr[i] > arr[j]:
                dp[i] = max(dp[i], dp[j] + 1)
                
    return max(dp)


def longest_increasing_subsequence_fast(arr: List[int]) -> int:
    """
    最長増加部分列（LIS）の長さを求める（O(n log n)の二分探索解法）
    
    Args:
        arr: 対象の配列
        
    Returns:
        最長増加部分列の長さ
    """
    if not arr:
        return 0
    
    # tailsは、長さiの増加部分列の末尾として考えられる最小値
    tails = []
    
    for x in arr:
        # 二分探索: xがどのtailsの位置に入るか
        idx = bisect.bisect_left(tails, x)
        
        if idx == len(tails):
            tails.append(x)  # 新しい長さの部分列を発見
        else:
            tails[idx] = x  # より小さな末尾値で更新
    
    return len(tails)


def longest_increasing_subsequence_with_sequence(arr: List[int]) -> Tuple[int, List[int]]:
    """
    最長増加部分列（LIS）の長さと実際の部分列を求める
    
    Args:
        arr: 対象の配列
        
    Returns:
        LISの長さと、実際のLISを表す配列のタプル
    """
    if not arr:
        return 0, []
        
    n = len(arr)
    dp = [1] * n  # dp[i]はarr[i]で終わる最長増加部分列の長さ
    prev = [-1] * n  # 各位置のLISにおける前の要素のインデックス
    
    for i in range(1, n):
        for j in range(i):
            if arr[i] > arr[j] and dp[i] < dp[j] + 1:
                dp[i] = dp[j] + 1
                prev[i] = j
    
    # 最長部分列の末尾を見つける
    max_length = max(dp)
    end_idx = dp.index(max_length)
    
    # 部分列を復元
    sequence = []
    while end_idx != -1:
        sequence.append(arr[end_idx])
        end_idx = prev[end_idx]
    
    return max_length, sequence[::-1]  # 逆順に


def longest_common_subsequence(s1: str, s2: str) -> int:
    """
    2つの文字列の最長共通部分列（LCS）の長さを求める
    
    Args:
        s1: 1つ目の文字列
        s2: 2つ目の文字列
        
    Returns:
        最長共通部分列の長さ
    """
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
                
    return dp[m][n]


def longest_common_subsequence_with_sequence(s1: str, s2: str) -> Tuple[int, str]:
    """
    2つの文字列の最長共通部分列（LCS）の長さと実際の部分列を求める
    
    Args:
        s1: 1つ目の文字列
        s2: 2つ目の文字列
        
    Returns:
        LCSの長さと、実際のLCSを表す文字列のタプル
    """
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # DPテーブルを構築
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    
    # LCSを復元
    i, j = m, n
    lcs = []
    
    while i > 0 and j > 0:
        if s1[i - 1] == s2[j - 1]:
            lcs.append(s1[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] > dp[i][j - 1]:
            i -= 1
        else:
            j -= 1
            
    return dp[m][n], ''.join(reversed(lcs))


def longest_common_substring(s1: str, s2: str) -> Tuple[int, str]:
    """
    2つの文字列の最長共通部分文字列の長さと実際の部分文字列を求める
    
    Args:
        s1: 1つ目の文字列
        s2: 2つ目の文字列
        
    Returns:
        最長共通部分文字列の長さと、実際の部分文字列を表す文字列のタプル
    """
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    max_length = 0
    end_pos = 0
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                if dp[i][j] > max_length:
                    max_length = dp[i][j]
                    end_pos = i
            else:
                dp[i][j] = 0
                
    return max_length, s1[end_pos - max_length:end_pos]


def edit_distance(s1: str, s2: str) -> int:
    """
    2つの文字列間の編集距離（レーベンシュタイン距離）を計算
    
    Args:
        s1: 1つ目の文字列
        s2: 2つ目の文字列
        
    Returns:
        編集距離（挿入、削除、置換の最小回数）
    """
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # 基底ケース: 空文字列への変換
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],      # 削除
                    dp[i][j - 1],      # 挿入
                    dp[i - 1][j - 1]   # 置換
                )
                
    return dp[m][n]


def edit_distance_with_operations(s1: str, s2: str) -> Tuple[int, List[Tuple[str, str, int, Optional[str]]]]:
    """
    2つの文字列間の編集距離と実際の編集操作を計算
    
    Args:
        s1: 1つ目の文字列
        s2: 2つ目の文字列
        
    Returns:
        編集距離と、編集操作のリストのタプル
        各編集操作は(操作タイプ, 位置, インデックス, [置換時の文字]) の形式
    """
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # 基底ケース
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    # DPテーブルを構築
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],
                    dp[i][j - 1],
                    dp[i - 1][j - 1]
                )
    
    # 編集操作を復元
    i, j = m, n
    operations = []
    
    while i > 0 or j > 0:
        if i > 0 and j > 0 and s1[i - 1] == s2[j - 1]:
            # 文字が一致する場合は操作なし
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            # 置換操作
            operations.append(("substitute", "s1", i - 1, s2[j - 1]))
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            # 削除操作
            operations.append(("delete", "s1", i - 1, None))
            i -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            # 挿入操作
            operations.append(("insert", "s1", i, s2[j - 1]))
            j -= 1
    
    return dp[m][n], operations[::-1]  # 操作を正順に


# 使用例
def example():
    # 最長増加部分列（LIS）
    arr = [10, 9, 2, 5, 3, 7, 101, 18]
    lis_length = longest_increasing_subsequence(arr)
    lis_length_fast = longest_increasing_subsequence_fast(arr)
    lis_length_with_seq, lis_seq = longest_increasing_subsequence_with_sequence(arr)
    
    print("配列:", arr)
    print(f"最長増加部分列の長さ (DP解法): {lis_length}")
    print(f"最長増加部分列の長さ (二分探索解法): {lis_length_fast}")
    print(f"最長増加部分列: {lis_seq}")
    
    # 最長共通部分列（LCS）
    s1 = "abcde"
    s2 = "ace"
    lcs_length = longest_common_subsequence(s1, s2)
    lcs_length_with_seq, lcs_seq = longest_common_subsequence_with_sequence(s1, s2)
    
    print(f"\n文字列1: {s1}")
    print(f"文字列2: {s2}")
    print(f"最長共通部分列の長さ: {lcs_length}")
    print(f"最長共通部分列: {lcs_seq}")
    
    # 最長共通部分文字列
    s1 = "abcdefghi"
    s2 = "xyzabcdef"
    substr_length, substr = longest_common_substring(s1, s2)
    
    print(f"\n文字列1: {s1}")
    print(f"文字列2: {s2}")
    print(f"最長共通部分文字列の長さ: {substr_length}")
    print(f"最長共通部分文字列: {substr}")
    
    # 編集距離
    s1 = "intention"
    s2 = "execution"
    ed = edit_distance(s1, s2)
    ed_with_ops, operations = edit_distance_with_operations(s1, s2)
    
    print(f"\n文字列1: {s1}")
    print(f"文字列2: {s2}")
    print(f"編集距離: {ed}")
    print("編集操作:")
    for op_type, target, idx, char in operations:
        if op_type == "substitute":
            print(f"  置換: {s1[idx]} -> {char} at position {idx}")
        elif op_type == "delete":
            print(f"  削除: {s1[idx]} at position {idx}")
        elif op_type == "insert":
            print(f"  挿入: {char} at position {idx}")


if __name__ == "__main__":
    example()
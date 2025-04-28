"""
動的計画法（Dynamic Programming）の基本アルゴリズム
- 部分問題の解を再利用して計算量を削減するアルゴリズム設計手法
- 主な用途:
  - 最適化問題の解決
  - カウント問題
  - マルチステージ決定問題
- 特徴:
  - 重複計算を避けることで時間計算量を改善
  - ボトムアップ（表埋め）またはトップダウン（メモ化再帰）で実装可能
- 代表的な問題:
  - ナップサック問題
  - 最長共通部分列（LCS）
  - 編集距離（レーベンシュタイン距離）
"""

from typing import List, Dict, Tuple, Set, Optional
import numpy as np


def knapsack_01(weights: List[int], values: List[int], capacity: int) -> int:
    """
    0-1ナップサック問題（ボトムアップDP）
    
    Args:
        weights: 各アイテムの重さ
        values: 各アイテムの価値
        capacity: ナップサックの容量
        
    Returns:
        int: 最大価値
    """
    n = len(weights)
    # dp[i][w] = 最初のi個のアイテムから選んで重さの合計がw以下になる最大価値
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            # i番目のアイテム（0-indexedで i-1）を選ばない場合
            dp[i][w] = dp[i - 1][w]
            
            # i番目のアイテムを選ぶ場合（容量が足りる場合のみ）
            if w >= weights[i - 1]:
                dp[i][w] = max(dp[i][w], dp[i - 1][w - weights[i - 1]] + values[i - 1])
    
    return dp[n][capacity]


def knapsack_01_optimized(weights: List[int], values: List[int], capacity: int) -> int:
    """
    0-1ナップサック問題（メモリ最適化版）
    
    Args:
        weights: 各アイテムの重さ
        values: 各アイテムの価値
        capacity: ナップサックの容量
        
    Returns:
        int: 最大価値
    """
    n = len(weights)
    # 1次元DPテーブルを使用（容量のみを考慮）
    dp = [0] * (capacity + 1)
    
    for i in range(n):
        # 重要: 容量を大きい方から順に埋める（値の上書きを防ぐ）
        for w in range(capacity, weights[i] - 1, -1):
            dp[w] = max(dp[w], dp[w - weights[i]] + values[i])
    
    return dp[capacity]


def knapsack_01_with_solution(weights: List[int], values: List[int], capacity: int) -> Tuple[int, List[int]]:
    """
    0-1ナップサック問題（最適解と選んだアイテムのインデックスを返す）
    
    Args:
        weights: 各アイテムの重さ
        values: 各アイテムの価値
        capacity: ナップサックの容量
        
    Returns:
        Tuple[int, List[int]]: (最大価値, 選んだアイテムのインデックスリスト)
    """
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            # i番目のアイテム（0-indexedで i-1）を選ばない場合
            dp[i][w] = dp[i - 1][w]
            
            # i番目のアイテムを選ぶ場合（容量が足りる場合のみ）
            if w >= weights[i - 1] and dp[i - 1][w - weights[i - 1]] + values[i - 1] > dp[i][w]:
                dp[i][w] = dp[i - 1][w - weights[i - 1]] + values[i - 1]
    
    # 最適解を復元
    selected_items = []
    w = capacity
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i - 1][w]:  # このアイテムを選んだ
            selected_items.append(i - 1)  # 0-indexedに変換
            w -= weights[i - 1]
    
    # リストを反転（最初に選んだアイテムから順に）
    selected_items.reverse()
    
    return dp[n][capacity], selected_items


def unbounded_knapsack(weights: List[int], values: List[int], capacity: int) -> int:
    """
    無制限ナップサック問題（各アイテムを何個でも選べる）
    
    Args:
        weights: 各アイテムの重さ
        values: 各アイテムの価値
        capacity: ナップサックの容量
        
    Returns:
        int: 最大価値
    """
    # dp[w] = 重さの合計がwになる最大価値
    dp = [0] * (capacity + 1)
    
    for w in range(1, capacity + 1):
        for i in range(len(weights)):
            if weights[i] <= w:
                dp[w] = max(dp[w], dp[w - weights[i]] + values[i])
    
    return dp[capacity]


def knapsack_with_groups(
    weights: List[List[int]], 
    values: List[List[int]], 
    capacity: int
) -> int:
    """
    グループ付きナップサック問題
    （各グループから最大1つのアイテムを選ぶ）
    
    Args:
        weights: グループごとの重さリスト
        values: グループごとの価値リスト
        capacity: ナップサックの容量
        
    Returns:
        int: 最大価値
    """
    n_groups = len(weights)
    # dp[g][w] = 最初のgグループから選んで重さがw以下になる最大価値
    dp = [[0] * (capacity + 1) for _ in range(n_groups + 1)]
    
    for g in range(1, n_groups + 1):
        group_weights = weights[g - 1]
        group_values = values[g - 1]
        
        for w in range(capacity + 1):
            # このグループからアイテムを選ばない場合
            dp[g][w] = dp[g - 1][w]
            
            # このグループの各アイテムを試す
            for i in range(len(group_weights)):
                if w >= group_weights[i]:
                    dp[g][w] = max(dp[g][w], dp[g - 1][w - group_weights[i]] + group_values[i])
    
    return dp[n_groups][capacity]


def lcs(s1: str, s2: str) -> str:
    """
    最長共通部分列（Longest Common Subsequence）
    
    Args:
        s1: 1つ目の文字列
        s2: 2つ目の文字列
        
    Returns:
        str: 最長共通部分列
    """
    m, n = len(s1), len(s2)
    # dp[i][j] = s1[:i]とs2[:j]の最長共通部分列の長さ
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    
    # 最長共通部分列を再構築
    lcs_string = []
    i, j = m, n
    while i > 0 and j > 0:
        if s1[i - 1] == s2[j - 1]:
            lcs_string.append(s1[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] > dp[i][j - 1]:
            i -= 1
        else:
            j -= 1
    
    return ''.join(reversed(lcs_string))


def lcs_length(s1: str, s2: str) -> int:
    """
    最長共通部分列の長さを求める（メモリ最適化版）
    
    Args:
        s1: 1つ目の文字列
        s2: 2つ目の文字列
        
    Returns:
        int: 最長共通部分列の長さ
    """
    m, n = len(s1), len(s2)
    # 2行だけ使う最適化版
    curr = [0] * (n + 1)
    prev = [0] * (n + 1)
    
    for i in range(1, m + 1):
        # 行を交換
        prev, curr = curr, prev
        
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
    
    return curr[n]


def edit_distance(s1: str, s2: str) -> int:
    """
    編集距離（レーベンシュタイン距離）
    
    Args:
        s1: 1つ目の文字列
        s2: 2つ目の文字列
        
    Returns:
        int: s1をs2に変換するための最小操作回数
             （挿入、削除、置換）
    """
    m, n = len(s1), len(s2)
    # dp[i][j] = s1[:i]からs2[:j]への編集距離
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # 境界条件：空文字列への変換コスト
    for i in range(m + 1):
        dp[i][0] = i  # 文字を削除するコスト
    
    for j in range(n + 1):
        dp[0][j] = j  # 文字を挿入するコスト
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]  # 変更不要
            else:
                # min(置換, 削除, 挿入)
                dp[i][j] = 1 + min(
                    dp[i - 1][j - 1],  # 置換
                    dp[i - 1][j],      # 削除
                    dp[i][j - 1]       # 挿入
                )
    
    return dp[m][n]


def edit_distance_with_operations(s1: str, s2: str) -> Tuple[int, List[str]]:
    """
    編集距離と具体的な操作列を求める
    
    Args:
        s1: 1つ目の文字列
        s2: 2つ目の文字列
        
    Returns:
        Tuple[int, List[str]]: (編集距離, 操作リスト)
    """
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # 境界条件
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    # DP表を埋める
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1])
    
    # 操作を再構築
    operations = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and s1[i - 1] == s2[j - 1]:
            # 文字が同じ（操作不要）
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            # 置換
            operations.append(f"Replace {s1[i-1]} with {s2[j-1]}")
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            # 削除
            operations.append(f"Delete {s1[i-1]}")
            i -= 1
        else:  # j > 0 and dp[i][j] == dp[i][j-1] + 1
            # 挿入
            operations.append(f"Insert {s2[j-1]}")
            j -= 1
    
    # 操作を逆順に
    operations.reverse()
    return dp[m][n], operations


def longest_increasing_subsequence(arr: List[int]) -> int:
    """
    最長増加部分列（Longest Increasing Subsequence）
    
    Args:
        arr: 整数配列
        
    Returns:
        int: 最長増加部分列の長さ
    """
    if not arr:
        return 0
    
    n = len(arr)
    # dp[i] = arr[i]で終わる最長増加部分列の長さ
    dp = [1] * n
    
    for i in range(1, n):
        for j in range(i):
            if arr[i] > arr[j]:
                dp[i] = max(dp[i], dp[j] + 1)
    
    return max(dp)


def longest_increasing_subsequence_optimized(arr: List[int]) -> int:
    """
    最長増加部分列の長さを二分探索で求める（O(n log n)）
    
    Args:
        arr: 整数配列
        
    Returns:
        int: 最長増加部分列の長さ
    """
    if not arr:
        return 0
    
    # tail[i] = 長さi+1の増加部分列における最小の末尾値
    tail = []
    
    for x in arr:
        # xを挿入できる位置を二分探索
        idx = bisect_left(tail, x)
        
        if idx == len(tail):
            # 新しい長さの増加部分列ができる
            tail.append(x)
        else:
            # 既存の増加部分列の末尾を更新
            tail[idx] = x
    
    return len(tail)


def bisect_left(arr: List[int], x: int) -> int:
    """
    二分探索でxを挿入できる最左位置を見つける
    """
    left, right = 0, len(arr)
    while left < right:
        mid = (left + right) // 2
        if arr[mid] < x:
            left = mid + 1
        else:
            right = mid
    return left


def longest_common_substring(s1: str, s2: str) -> str:
    """
    最長共通部分文字列（連続する部分文字列）
    
    Args:
        s1: 1つ目の文字列
        s2: 2つ目の文字列
        
    Returns:
        str: 最長共通部分文字列
    """
    m, n = len(s1), len(s2)
    # dp[i][j] = s1[i-1]とs2[j-1]で終わる最長共通部分文字列の長さ
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
    
    # 最長共通部分文字列を再構築
    start_pos = end_pos - max_length
    return s1[start_pos:end_pos]


def subset_sum(nums: List[int], target: int) -> bool:
    """
    部分和問題：指定した和になる部分集合が存在するか判定
    
    Args:
        nums: 整数配列
        target: 目標の和
        
    Returns:
        bool: 部分集合が存在すればTrue、なければFalse
    """
    # dp[i][j] = 最初のi個の整数から選んで和をjにできるかどうか
    n = len(nums)
    dp = [[False] * (target + 1) for _ in range(n + 1)]
    
    # 和が0になる部分集合は、「何も選ばない」という選択があるのでTrue
    for i in range(n + 1):
        dp[i][0] = True
    
    for i in range(1, n + 1):
        for j in range(1, target + 1):
            # i番目の要素を選ばない場合
            dp[i][j] = dp[i - 1][j]
            
            # i番目の要素を選ぶ場合（選べる場合のみ）
            if j >= nums[i - 1]:
                dp[i][j] |= dp[i - 1][j - nums[i - 1]]
    
    return dp[n][target]


def subset_sum_optimized(nums: List[int], target: int) -> bool:
    """
    部分和問題（メモリ最適化版）
    
    Args:
        nums: 整数配列
        target: 目標の和
        
    Returns:
        bool: 部分集合が存在すればTrue、なければFalse
    """
    # 1次元DPテーブル
    dp = [False] * (target + 1)
    dp[0] = True
    
    for num in nums:
        # 目標から逆順に埋めて上書きを防ぐ
        for j in range(target, num - 1, -1):
            dp[j] |= dp[j - num]
    
    return dp[target]


def subset_sum_with_solution(nums: List[int], target: int) -> Tuple[bool, List[int]]:
    """
    部分和問題と解の復元
    
    Args:
        nums: 整数配列
        target: 目標の和
        
    Returns:
        Tuple[bool, List[int]]: (可能かどうか, 選んだ要素のインデックスリスト)
    """
    n = len(nums)
    dp = [[False] * (target + 1) for _ in range(n + 1)]
    
    # 和が0になる部分集合は「何も選ばない」
    for i in range(n + 1):
        dp[i][0] = True
    
    for i in range(1, n + 1):
        for j in range(1, target + 1):
            # i番目の要素を選ばない場合
            dp[i][j] = dp[i - 1][j]
            
            # i番目の要素を選ぶ場合（選べる場合のみ）
            if j >= nums[i - 1]:
                dp[i][j] |= dp[i - 1][j - nums[i - 1]]
    
    # 解が存在しない場合
    if not dp[n][target]:
        return False, []
    
    # 解を再構築
    selected = []
    j = target
    for i in range(n, 0, -1):
        # この要素を選んだ場合
        if j >= nums[i - 1] and dp[i - 1][j - nums[i - 1]]:
            selected.append(i - 1)  # 0-indexedに変換
            j -= nums[i - 1]
    
    selected.reverse()  # 順序を元に戻す
    return True, selected


def coin_change(coins: List[int], amount: int) -> int:
    """
    コイン問題：指定した金額を作るのに必要な最小のコイン数
    
    Args:
        coins: 使用可能なコインの種類
        amount: 目標金額
        
    Returns:
        int: 最小のコイン数（不可能な場合は-1）
    """
    # dp[i] = i円を作るのに必要な最小のコイン数
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    
    for coin in coins:
        for i in range(coin, amount + 1):
            dp[i] = min(dp[i], dp[i - coin] + 1)
    
    return dp[amount] if dp[amount] != float('inf') else -1


def coin_combinations(coins: List[int], amount: int) -> int:
    """
    コイン問題：指定した金額を作る組み合わせの総数
    
    Args:
        coins: 使用可能なコインの種類
        amount: 目標金額
        
    Returns:
        int: 組み合わせの総数
    """
    # dp[i] = i円を作る組み合わせの総数
    dp = [0] * (amount + 1)
    dp[0] = 1
    
    for coin in coins:
        for i in range(coin, amount + 1):
            dp[i] += dp[i - coin]
    
    return dp[amount]


def matrix_chain_multiplication(dimensions: List[int]) -> int:
    """
    行列連鎖積の最適な計算順序
    
    Args:
        dimensions: 行列のサイズ（連続する行列AとBがあるとき、
                    Aは dimensions[0]×dimensions[1]、
                    Bは dimensions[1]×dimensions[2]、という形式）
        
    Returns:
        int: 最小の乗算回数
    """
    n = len(dimensions) - 1  # 行列の数
    
    # dp[i][j] = i番目からj番目までの行列を掛け合わせる最小コスト
    dp = [[0] * (n + 1) for _ in range(n + 1)]
    
    # 鎖の長さを徐々に増やしていく
    for chain_len in range(2, n + 1):
        for i in range(1, n - chain_len + 2):
            j = i + chain_len - 1
            dp[i][j] = float('inf')
            
            # i番目からj番目の行列の間に分割点kを置く
            for k in range(i, j):
                cost = dp[i][k] + dp[k + 1][j] + dimensions[i - 1] * dimensions[k] * dimensions[j]
                dp[i][j] = min(dp[i][j], cost)
    
    return dp[1][n]


def rod_cutting(prices: List[int], n: int) -> int:
    """
    棒切り問題：長さnの棒を切断して得られる最大の利益
    
    Args:
        prices: 各長さの価格リスト（prices[i]は長さi+1の価格）
        n: 棒の長さ
        
    Returns:
        int: 最大利益
    """
    # dp[i] = 長さiの棒から得られる最大利益
    dp = [0] * (n + 1)
    
    for i in range(1, n + 1):
        for j in range(1, i + 1):
            # 長さjの棒を切り取り、残りを再利用
            if j <= len(prices):
                dp[i] = max(dp[i], dp[i - j] + prices[j - 1])
    
    return dp[n]


def longest_palindromic_subsequence(s: str) -> int:
    """
    最長回文部分列（Longest Palindromic Subsequence）
    
    Args:
        s: 文字列
        
    Returns:
        int: 最長回文部分列の長さ
    """
    n = len(s)
    # dp[i][j] = s[i:j+1]の最長回文部分列の長さ
    dp = [[0] * n for _ in range(n)]
    
    # 長さ1の部分列は常に回文
    for i in range(n):
        dp[i][i] = 1
    
    # 長さを徐々に増やす
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            
            if s[i] == s[j]:
                dp[i][j] = dp[i + 1][j - 1] + 2
            else:
                dp[i][j] = max(dp[i + 1][j], dp[i][j - 1])
    
    return dp[0][n - 1]


# 使用例
def example():
    print("===== 0-1ナップサック問題 =====")
    weights = [2, 3, 4, 5]
    values = [3, 4, 5, 6]
    capacity = 8
    
    max_value = knapsack_01(weights, values, capacity)
    print(f"最大価値: {max_value}")
    
    max_value_optimized = knapsack_01_optimized(weights, values, capacity)
    print(f"最適化版の最大価値: {max_value_optimized}")
    
    max_value, selected = knapsack_01_with_solution(weights, values, capacity)
    print(f"選んだアイテム (0-indexed): {selected}")
    
    print("\n===== 無制限ナップサック問題 =====")
    max_value_unbounded = unbounded_knapsack(weights, values, capacity)
    print(f"最大価値（無制限）: {max_value_unbounded}")
    
    print("\n===== 最長共通部分列（LCS）=====")
    s1 = "ABCBDAB"
    s2 = "BDCABA"
    lcs_result = lcs(s1, s2)
    print(f"LCS: {lcs_result}")
    print(f"LCS長: {lcs_length(s1, s2)}")
    
    print("\n===== 編集距離 =====")
    s1 = "kitten"
    s2 = "sitting"
    dist = edit_distance(s1, s2)
    print(f"編集距離: {dist}")
    
    dist, ops = edit_distance_with_operations(s1, s2)
    print(f"操作列: {ops}")
    
    print("\n===== 最長増加部分列（LIS）=====")
    arr = [10, 22, 9, 33, 21, 50, 41, 60]
    lis_length = longest_increasing_subsequence(arr)
    print(f"LIS長: {lis_length}")
    
    lis_length_optimized = longest_increasing_subsequence_optimized(arr)
    print(f"LIS長（最適化版）: {lis_length_optimized}")
    
    print("\n===== 最長共通部分文字列 =====")
    s1 = "ABABC"
    s2 = "BABCA"
    lcs_substring = longest_common_substring(s1, s2)
    print(f"最長共通部分文字列: {lcs_substring}")
    
    print("\n===== 部分和問題 =====")
    nums = [3, 34, 4, 12, 5, 2]
    target = 9
    
    exists = subset_sum(nums, target)
    print(f"和が{target}になる部分集合は存在する？: {exists}")
    
    exists_optimized = subset_sum_optimized(nums, target)
    print(f"（最適化版）和が{target}になる部分集合は存在する？: {exists_optimized}")
    
    exists, subset = subset_sum_with_solution(nums, target)
    print(f"選んだ要素のインデックス: {subset}")
    
    print("\n===== コイン問題 =====")
    coins = [1, 2, 5]
    amount = 11
    
    min_coins = coin_change(coins, amount)
    print(f"{amount}円を作るのに必要な最小のコイン数: {min_coins}")
    
    combinations = coin_combinations(coins, amount)
    print(f"{amount}円を作る組み合わせの総数: {combinations}")


if __name__ == "__main__":
    example()
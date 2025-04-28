"""
ナップサック問題（Knapsack Problem）
- 制約のある組み合わせ最適化問題の代表例
- 容量制限のあるナップサックに、価値の総和を最大化するようにアイテムを選ぶ
- バリエーション:
  - 0/1 ナップサック問題: 各アイテムを選ぶか選ばないか
  - 分数ナップサック問題: アイテムを分割して選べる
  - 複数ナップサック問題: 複数のナップサックに最適に割り当てる
  - 多次元ナップサック問題: 複数の制約を持つ
- 計算量:
  - 0/1 ナップサック問題: O(nW) (n: アイテム数, W: ナップサックの容量)
  - 分数ナップサック問題: O(n log n) (ソートにかかる時間)
"""

from typing import List, Tuple, Dict


def knapsack_01(weights: List[int], values: List[int], capacity: int) -> int:
    """
    0/1ナップサック問題を解く標準的なDP解法
    
    各アイテムを「取る」か「取らない」かの二択となる
    
    Args:
        weights: 各アイテムの重量
        values: 各アイテムの価値
        capacity: ナップサックの容量
        
    Returns:
        最大価値
    """
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            if weights[i - 1] <= w:
                # アイテムiを選ぶ場合と選ばない場合の最大値を選択
                dp[i][w] = max(values[i - 1] + dp[i - 1][w - weights[i - 1]], dp[i - 1][w])
            else:
                # アイテムiは選べないので、前のアイテムまでの最大値を引き継ぐ
                dp[i][w] = dp[i - 1][w]
                
    return dp[n][capacity]


def knapsack_01_with_items(weights: List[int], values: List[int], capacity: int) -> Tuple[int, List[int]]:
    """
    0/1ナップサック問題を解き、選択したアイテムのインデックスも返す
    
    Args:
        weights: 各アイテムの重量
        values: 各アイテムの価値
        capacity: ナップサックの容量
        
    Returns:
        (最大価値, 選択したアイテムのインデックスのリスト)
    """
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    
    # DP計算
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            if weights[i - 1] <= w:
                # アイテムiを選ぶ場合と選ばない場合の最大値を選択
                dp[i][w] = max(values[i - 1] + dp[i - 1][w - weights[i - 1]], dp[i - 1][w])
            else:
                # アイテムiは選べないので、前のアイテムまでの最大値を引き継ぐ
                dp[i][w] = dp[i - 1][w]
    
    # 選択したアイテムを復元
    selected_items = []
    w = capacity
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i - 1][w]:
            # アイテムiを選択したなら
            selected_items.append(i - 1)  # 0-indexedに変換
            w -= weights[i - 1]
    
    selected_items.reverse()  # 小さい順に並べ替え
    return dp[n][capacity], selected_items


def knapsack_01_optimized(weights: List[int], values: List[int], capacity: int) -> int:
    """
    0/1ナップサック問題の空間最適化版（1次元DP）
    
    Args:
        weights: 各アイテムの重量
        values: 各アイテムの価値
        capacity: ナップサックの容量
        
    Returns:
        最大価値
    """
    n = len(weights)
    dp = [0] * (capacity + 1)
    
    for i in range(n):
        # 容量の大きい方から更新していく（重複利用を防ぐため）
        for w in range(capacity, weights[i] - 1, -1):
            dp[w] = max(dp[w], values[i] + dp[w - weights[i]])
                
    return dp[capacity]


def fractional_knapsack(weights: List[int], values: List[int], capacity: int) -> float:
    """
    分数ナップサック問題を解く（貪欲法）
    
    各アイテムを分割して部分的に選ぶことができる
    
    Args:
        weights: 各アイテムの重量
        values: 各アイテムの価値
        capacity: ナップサックの容量
        
    Returns:
        最大価値（浮動小数点）
    """
    # 単位重量あたりの価値を計算してソート
    items = [(values[i] / weights[i], weights[i], values[i]) for i in range(len(weights))]
    items.sort(reverse=True)  # 単位価値の高い順に並べる
    
    total_value = 0
    remaining_capacity = capacity
    
    for unit_value, weight, value in items:
        if remaining_capacity >= weight:
            # アイテム全体を選択
            total_value += value
            remaining_capacity -= weight
        else:
            # アイテムの一部だけ選択
            total_value += unit_value * remaining_capacity
            break
            
    return total_value


def knapsack_unlimited(weights: List[int], values: List[int], capacity: int) -> int:
    """
    無制限ナップサック問題を解く（同じアイテムを複数回選択可能）
    
    Args:
        weights: 各アイテムの重量
        values: 各アイテムの価値
        capacity: ナップサックの容量
        
    Returns:
        最大価値
    """
    dp = [0] * (capacity + 1)
    
    for w in range(1, capacity + 1):
        for i in range(len(weights)):
            if weights[i] <= w:
                dp[w] = max(dp[w], dp[w - weights[i]] + values[i])
                
    return dp[capacity]


def knapsack_bounded(weights: List[int], values: List[int], counts: List[int], capacity: int) -> int:
    """
    制限付きナップサック問題（各アイテムには選択可能な最大数がある）
    
    Args:
        weights: 各アイテムの重量
        values: 各アイテムの価値
        counts: 各アイテムの最大数
        capacity: ナップサックの容量
        
    Returns:
        最大価値
    """
    n = len(weights)
    dp = [0] * (capacity + 1)
    
    for i in range(n):
        # 各アイテムごとの処理
        for j in range(1, counts[i] + 1):
            # 1からcounts[i]個まで使う場合をすべて試す
            weight = weights[i] * j
            value = values[i] * j
            
            # 容量の大きい方から更新
            for w in range(capacity, weight - 1, -1):
                dp[w] = max(dp[w], dp[w - weight] + value)
                
    return dp[capacity]


def multi_dimensional_knapsack(weights_list: List[List[int]], values: List[int], capacities: List[int]) -> int:
    """
    多次元ナップサック問題を解く（複数の制約条件を持つ）
    
    Args:
        weights_list: 各アイテムの各次元における重量 [次元][アイテム]
        values: 各アイテムの価値
        capacities: 各次元のナップサックの容量
        
    Returns:
        最大価値
    """
    n = len(values)
    dimensions = len(capacities)
    
    # 動的計画法テーブル
    dp = {}
    dp[(0, ) * dimensions] = 0
    
    # 部分問題を解く
    for i in range(n):
        new_dp = dp.copy()
        item_weights = [weights_list[d][i] for d in range(dimensions)]
        
        for state, value in dp.items():
            # アイテムを選ぶ場合
            new_state = tuple(min(state[d] + item_weights[d], capacities[d]) for d in range(dimensions))
            if all(new_state[d] <= capacities[d] for d in range(dimensions)):
                new_dp[new_state] = max(new_dp.get(new_state, 0), value + values[i])
        
        dp = new_dp
    
    # 最大価値を見つける
    max_value = 0
    for value in dp.values():
        max_value = max(max_value, value)
    
    return max_value


# 使用例
def example():
    # 0/1 ナップサック問題
    weights = [2, 3, 4, 5]
    values = [3, 4, 5, 6]
    capacity = 8
    
    max_value = knapsack_01(weights, values, capacity)
    print(f"0/1 ナップサック問題の最大価値: {max_value}")
    
    max_value, selected_items = knapsack_01_with_items(weights, values, capacity)
    print(f"選択したアイテム: {selected_items}")
    
    # 空間最適化版
    max_value = knapsack_01_optimized(weights, values, capacity)
    print(f"空間最適化版の最大価値: {max_value}")
    
    # 分数ナップサック問題
    max_value = fractional_knapsack(weights, values, capacity)
    print(f"分数ナップサック問題の最大価値: {max_value}")
    
    # 無制限ナップサック問題
    max_value = knapsack_unlimited(weights, values, capacity)
    print(f"無制限ナップサック問題の最大価値: {max_value}")
    
    # 制限付きナップサック問題
    counts = [1, 2, 3, 2]  # 各アイテムの最大数
    max_value = knapsack_bounded(weights, values, counts, capacity)
    print(f"制限付きナップサック問題の最大価値: {max_value}")
    
    # 多次元ナップサック問題
    weights_dim1 = [2, 3, 4, 5]
    weights_dim2 = [3, 1, 2, 4]
    weights_list = [weights_dim1, weights_dim2]
    capacities = [8, 7]
    max_value = multi_dimensional_knapsack(weights_list, values, capacities)
    print(f"多次元ナップサック問題の最大価値: {max_value}")


if __name__ == "__main__":
    example()
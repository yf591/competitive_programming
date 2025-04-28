"""
二分探索アルゴリズム（Binary Search Algorithm）
- ソート済みの配列から効率的に要素を検索するアルゴリズム
- 主な用途:
  - ソート済み配列での要素検索
  - 答えの境界を求める問題（二分探索法）
  - 最適化問題の解決
- 特徴:
  - 各ステップで問題サイズを半分に減らす
  - ログ時間で解を見つけられる
- 計算量:
  - O(log n)（nは配列の要素数）
"""

from typing import List, TypeVar, Callable, Optional, Any, Union

T = TypeVar('T')


def binary_search(arr: List[T], target: T) -> int:
    """
    基本的な二分探索アルゴリズム
    
    Args:
        arr: ソート済みの配列
        target: 検索対象の要素
        
    Returns:
        int: 要素が見つかった場合はそのインデックス、見つからない場合は-1
    """
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = left + (right - left) // 2
        
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1


def lower_bound(arr: List[T], target: T) -> int:
    """
    target以上の最初の要素の位置を返す
    
    Args:
        arr: ソート済みの配列
        target: 検索対象の値
        
    Returns:
        int: target以上の最初の要素のインデックス
             該当する要素がない場合はlen(arr)
    """
    left, right = 0, len(arr)
    
    while left < right:
        mid = left + (right - left) // 2
        
        if arr[mid] < target:
            left = mid + 1
        else:
            right = mid
    
    return left


def upper_bound(arr: List[T], target: T) -> int:
    """
    targetより大きい最初の要素の位置を返す
    
    Args:
        arr: ソート済みの配列
        target: 検索対象の値
        
    Returns:
        int: targetより大きい最初の要素のインデックス
             該当する要素がない場合はlen(arr)
    """
    left, right = 0, len(arr)
    
    while left < right:
        mid = left + (right - left) // 2
        
        if arr[mid] <= target:
            left = mid + 1
        else:
            right = mid
    
    return left


def binary_search_first_occurrence(arr: List[T], target: T) -> int:
    """
    targetの最初の出現位置を返す
    
    Args:
        arr: ソート済みの配列
        target: 検索対象の値
        
    Returns:
        int: targetの最初の出現位置
             該当する要素がない場合は-1
    """
    pos = lower_bound(arr, target)
    if pos < len(arr) and arr[pos] == target:
        return pos
    return -1


def binary_search_last_occurrence(arr: List[T], target: T) -> int:
    """
    targetの最後の出現位置を返す
    
    Args:
        arr: ソート済みの配列
        target: 検索対象の値
        
    Returns:
        int: targetの最後の出現位置
             該当する要素がない場合は-1
    """
    pos = upper_bound(arr, target)
    if pos > 0 and arr[pos - 1] == target:
        return pos - 1
    return -1


def binary_search_predicate(left: int, right: int, predicate: Callable[[int], bool]) -> int:
    """
    述語関数に基づく二分探索
    predicateがTrueとなる最小の値を求める
    
    Args:
        left: 探索範囲の左端
        right: 探索範囲の右端
        predicate: 判定関数 f(x) -> bool
        
    Returns:
        int: predicateがTrueとなる最小の値
             該当する値がない場合はright+1を返す
    """
    while left < right:
        mid = left + (right - left) // 2
        
        if predicate(mid):
            right = mid
        else:
            left = mid + 1
    
    # 解が存在するか確認
    if left <= right and predicate(left):
        return left
    
    return right + 1


def binary_search_float(
    left: float, right: float, 
    predicate: Callable[[float], bool], 
    epsilon: float = 1e-9, 
    max_iterations: int = 100
) -> float:
    """
    浮動小数点数の範囲で二分探索を行う
    
    Args:
        left: 探索範囲の左端
        right: 探索範囲の右端
        predicate: 判定関数 f(x) -> bool
        epsilon: 許容誤差
        max_iterations: 最大繰り返し回数
        
    Returns:
        float: predicateがTrueとなる最小の値
               該当する値がない場合はrightを返す
    """
    iterations = 0
    
    while right - left > epsilon and iterations < max_iterations:
        mid = left + (right - left) / 2
        
        if predicate(mid):
            right = mid
        else:
            left = mid
        
        iterations += 1
    
    return left


def ternary_search_max(
    left: float, right: float, 
    function: Callable[[float], float], 
    epsilon: float = 1e-9
) -> float:
    """
    三分探索で凸関数の最大値を求める
    
    Args:
        left: 探索範囲の左端
        right: 探索範囲の右端
        function: 評価関数 f(x) -> float
        epsilon: 許容誤差
        
    Returns:
        float: 関数の最大値を取る点のx座標
    """
    while right - left > epsilon:
        mid1 = left + (right - left) / 3
        mid2 = right - (right - left) / 3
        
        if function(mid1) < function(mid2):
            left = mid1
        else:
            right = mid2
    
    return (left + right) / 2


def ternary_search_min(
    left: float, right: float, 
    function: Callable[[float], float], 
    epsilon: float = 1e-9
) -> float:
    """
    三分探索で凸関数の最小値を求める
    
    Args:
        left: 探索範囲の左端
        right: 探索範囲の右端
        function: 評価関数 f(x) -> float
        epsilon: 許容誤差
        
    Returns:
        float: 関数の最小値を取る点のx座標
    """
    while right - left > epsilon:
        mid1 = left + (right - left) / 3
        mid2 = right - (right - left) / 3
        
        if function(mid1) > function(mid2):
            left = mid1
        else:
            right = mid2
    
    return (left + right) / 2


def binary_search_matrix(matrix: List[List[T]], target: T) -> tuple:
    """
    行と列がソートされた2次元配列での二分探索
    
    Args:
        matrix: ソートされた2次元配列（各行と各列がソート済み）
        target: 検索対象の値
        
    Returns:
        tuple: (row, col) 要素が見つかった場合はその位置、見つからない場合は(-1, -1)
    """
    if not matrix or not matrix[0]:
        return (-1, -1)
    
    rows, cols = len(matrix), len(matrix[0])
    row, col = 0, cols - 1
    
    while row < rows and col >= 0:
        if matrix[row][col] == target:
            return (row, col)
        elif matrix[row][col] > target:
            col -= 1
        else:
            row += 1
    
    return (-1, -1)


def exponential_search(arr: List[T], target: T) -> int:
    """
    指数探索：大きな配列や無限配列での効率的な二分探索
    
    Args:
        arr: ソート済みの配列
        target: 検索対象の値
        
    Returns:
        int: 要素が見つかった場合はそのインデックス、見つからない場合は-1
    """
    if len(arr) == 0:
        return -1
    
    if arr[0] == target:
        return 0
    
    # 指数的に増加する範囲を探す
    i = 1
    while i < len(arr) and arr[i] <= target:
        i *= 2
    
    # 見つかった範囲で二分探索
    return binary_search_range(arr, target, i // 2, min(i, len(arr) - 1))


def binary_search_range(arr: List[T], target: T, left: int, right: int) -> int:
    """
    指定した範囲内での二分探索
    
    Args:
        arr: ソート済みの配列
        target: 検索対象の値
        left: 探索範囲の左端
        right: 探索範囲の右端
        
    Returns:
        int: 要素が見つかった場合はそのインデックス、見つからない場合は-1
    """
    while left <= right:
        mid = left + (right - left) // 2
        
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1


def interpolation_search(arr: List[int], target: int) -> int:
    """
    補間探索：均等に分布した数値配列で効率的な検索
    
    Args:
        arr: ソート済みの数値配列
        target: 検索対象の値
        
    Returns:
        int: 要素が見つかった場合はそのインデックス、見つからない場合は-1
    """
    left, right = 0, len(arr) - 1
    
    while left <= right and arr[left] <= target <= arr[right]:
        if left == right:
            if arr[left] == target:
                return left
            return -1
        
        # 補間公式でmidを計算
        pos = left + int(((right - left) * (target - arr[left])) / (arr[right] - arr[left]))
        
        if arr[pos] == target:
            return pos
        elif arr[pos] < target:
            left = pos + 1
        else:
            right = pos - 1
    
    return -1


def count_occurrences(arr: List[T], target: T) -> int:
    """
    ソート済み配列内の要素の出現回数をカウント
    
    Args:
        arr: ソート済みの配列
        target: カウント対象の値
        
    Returns:
        int: targetの出現回数
    """
    first = lower_bound(arr, target)
    last = upper_bound(arr, target)
    
    return last - first


def find_closest(arr: List[T], target: T) -> Optional[T]:
    """
    ソート済み配列内でtargetに最も近い値を見つける
    
    Args:
        arr: ソート済みの配列
        target: 対象の値
        
    Returns:
        Optional[T]: targetに最も近い値、配列が空の場合はNone
    """
    if not arr:
        return None
    
    pos = lower_bound(arr, target)
    
    if pos == 0:
        return arr[0]
    if pos == len(arr):
        return arr[-1]
    
    # targetの前後の要素で、より近い方を選ぶ
    if abs(arr[pos] - target) < abs(arr[pos - 1] - target):
        return arr[pos]
    else:
        return arr[pos - 1]


# 使用例
def example():
    print("===== 基本的な二分探索 =====")
    arr = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
    target = 7
    idx = binary_search(arr, target)
    print(f"{target}の位置: {idx}")
    
    print("\n===== 境界の二分探索 =====")
    arr = [1, 3, 5, 5, 5, 7, 9, 11]
    target = 5
    
    first = lower_bound(arr, target)
    last = upper_bound(arr, target)
    print(f"{target}以上の最初の位置: {first}")
    print(f"{target}より大きい最初の位置: {last}")
    print(f"{target}の出現回数: {last - first}")
    
    first_occur = binary_search_first_occurrence(arr, target)
    last_occur = binary_search_last_occurrence(arr, target)
    print(f"{target}の最初の出現位置: {first_occur}")
    print(f"{target}の最後の出現位置: {last_occur}")
    
    print("\n===== 述語二分探索 =====")
    # xの2乗がターゲット以上になる最小のxを求める
    target = 30
    predicate = lambda x: x * x >= target
    result = binary_search_predicate(0, 10, predicate)
    print(f"x^2 >= {target}を満たす最小のx: {result}")
    
    print("\n===== 浮動小数点二分探索 =====")
    # √2を二分探索で計算
    predicate = lambda x: x * x >= 2
    sqrt2 = binary_search_float(0, 2, predicate)
    print(f"√2 ≈ {sqrt2}")
    
    print("\n===== 三分探索 =====")
    # f(x) = -(x-2)^2 + 4 の最大値
    func = lambda x: -(x - 2) ** 2 + 4
    max_x = ternary_search_max(0, 4, func)
    print(f"f(x) = -(x-2)^2 + 4 の最大値を取るx ≈ {max_x}")
    print(f"最大値 ≈ {func(max_x)}")


if __name__ == "__main__":
    example()
"""
座標圧縮（Coordinate Compression）
- 大きな範囲の値を0からの連続した整数に変換するテクニック
- 主な用途:
  - 値の範囲が大きい場合のメモリ効率化
  - インデックスとしての使用（配列やフェニック木など）
  - 離散化によるアルゴリズムの簡略化
- 計算量:
  - 前処理: O(n log n)（ソートに依存）
  - クエリ: O(log n)（二分探索）または O(1)（辞書使用時）
"""

from typing import List, Dict, Tuple, Union, TypeVar, Generic, Any
import bisect

T = TypeVar('T', int, float, str)


def coordinate_compression(arr: List[T]) -> Tuple[List[int], Dict[int, T]]:
    """
    座標圧縮を行う基本的な実装
    
    Args:
        arr: 元の配列
        
    Returns:
        圧縮後の配列と、圧縮された値から元の値へのマッピング
    """
    # 重複を除去して昇順にソート
    sorted_arr = sorted(set(arr))
    
    # 元の値から圧縮された値へのマッピング
    compress_map = {val: i for i, val in enumerate(sorted_arr)}
    
    # 値を圧縮
    compressed = [compress_map[x] for x in arr]
    
    # 圧縮された値から元の値へのマッピング
    reverse_map = {i: val for i, val in enumerate(sorted_arr)}
    
    return compressed, reverse_map


def compress_2d_points(points: List[Tuple[int, int]]) -> Tuple[List[Tuple[int, int]], Dict[int, int], Dict[int, int]]:
    """
    2次元平面上の点の座標圧縮
    
    Args:
        points: (x, y)座標のタプルのリスト
        
    Returns:
        圧縮後の点のリスト、x座標のマッピング、y座標のマッピング
    """
    # x座標とy座標を分離
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    
    # x座標とy座標を別々に圧縮
    x_sorted = sorted(set(x_coords))
    y_sorted = sorted(set(y_coords))
    
    x_compress = {val: i for i, val in enumerate(x_sorted)}
    y_compress = {val: i for i, val in enumerate(y_sorted)}
    
    # 点の座標を圧縮
    compressed_points = [(x_compress[p[0]], y_compress[p[1]]) for p in points]
    
    # 元の値へのマッピング
    x_reverse = {i: val for i, val in enumerate(x_sorted)}
    y_reverse = {i: val for i, val in enumerate(y_sorted)}
    
    return compressed_points, x_reverse, y_reverse


class CoordinateCompressor(Generic[T]):
    """
    座標圧縮を行うためのクラス
    値の追加と圧縮の操作を分離可能
    """
    
    def __init__(self):
        """座標圧縮器を初期化"""
        self.values = set()
        self._sorted_values = None
        self._compress_map = None
        self._reverse_map = None
        self._is_compressed = False
    
    def add(self, value: T) -> None:
        """値を追加"""
        self.values.add(value)
        self._is_compressed = False
    
    def add_all(self, values: List[T]) -> None:
        """複数の値を一度に追加"""
        self.values.update(values)
        self._is_compressed = False
    
    def compress(self) -> None:
        """圧縮処理を実行"""
        self._sorted_values = sorted(self.values)
        self._compress_map = {val: i for i, val in enumerate(self._sorted_values)}
        self._reverse_map = {i: val for i, val in enumerate(self._sorted_values)}
        self._is_compressed = True
    
    def get_compressed(self, value: T) -> int:
        """
        値を圧縮した結果を取得
        
        Args:
            value: 圧縮する値
            
        Returns:
            圧縮後の値（インデックス）
            
        Raises:
            ValueError: 値が見つからないか、まだ圧縮処理が実行されていない場合
        """
        if not self._is_compressed:
            raise ValueError("圧縮処理が実行されていません。compress()を先に呼び出してください。")
        
        if value in self._compress_map:
            return self._compress_map[value]
        
        # 値が見つからない場合、近い値のインデックスを返すオプション
        # （この場合は値の挿入位置を返す）
        idx = bisect.bisect_left(self._sorted_values, value)
        raise ValueError(f"値 {value} は圧縮対象に含まれていません。挿入位置: {idx}")
    
    def get_original(self, compressed: int) -> T:
        """
        圧縮された値から元の値を取得
        
        Args:
            compressed: 圧縮後の値（インデックス）
            
        Returns:
            元の値
            
        Raises:
            ValueError: インデックスが範囲外か、まだ圧縮処理が実行されていない場合
        """
        if not self._is_compressed:
            raise ValueError("圧縮処理が実行されていません。compress()を先に呼び出してください。")
        
        if compressed in self._reverse_map:
            return self._reverse_map[compressed]
        
        raise ValueError(f"インデックス {compressed} は範囲外です。有効範囲: 0-{len(self._sorted_values)-1}")
    
    def get_compressed_all(self, values: List[T]) -> List[int]:
        """複数の値を一度に圧縮"""
        return [self.get_compressed(val) for val in values]
    
    def get_original_all(self, compressed_values: List[int]) -> List[T]:
        """複数の圧縮値から元の値を取得"""
        return [self.get_original(val) for val in compressed_values]
    
    def size(self) -> int:
        """圧縮後の値の範囲（異なる値の数）"""
        if not self._is_compressed:
            self.compress()
        return len(self._sorted_values)
    
    def get_all_original_values(self) -> List[T]:
        """ソートされた元の値のリストを取得"""
        if not self._is_compressed:
            self.compress()
        return self._sorted_values.copy()


def range_queries_with_compression(ranges: List[Tuple[int, int]], queries: List[Tuple[str, int, int]]) -> List[int]:
    """
    座標圧縮を使用した範囲クエリの処理例
    
    例: 数直線上の範囲を追加し、ある点がいくつの範囲に含まれるかを計算
    
    Args:
        ranges: (start, end)形式の範囲のリスト
        queries: (操作タイプ, 値1, 値2)形式のクエリのリスト
                "add": 範囲を追加、"count": 点が含まれる範囲の数を計算
        
    Returns:
        各"count"クエリの結果のリスト
    """
    # 座標圧縮に使用する値を収集
    all_values = []
    for start, end in ranges:
        all_values.append(start)
        all_values.append(end)
    
    for query_type, val1, val2 in queries:
        if query_type == "count":
            all_values.append(val1)
    
    # 座標圧縮
    compressor = CoordinateCompressor()
    compressor.add_all(all_values)
    compressor.compress()
    
    # 圧縮後の座標で範囲を表現
    compressed_ranges = []
    for start, end in ranges:
        compressed_start = compressor.get_compressed(start)
        compressed_end = compressor.get_compressed(end)
        compressed_ranges.append((compressed_start, compressed_end))
    
    max_coord = compressor.size()
    
    # いもす法用の配列
    imos = [0] * (max_coord + 1)
    
    # 範囲の開始点で+1、終了点の次で-1
    for start, end in compressed_ranges:
        imos[start] += 1
        imos[end + 1] -= 1
    
    # 累積和を計算
    for i in range(1, len(imos)):
        imos[i] += imos[i - 1]
    
    # クエリ処理
    results = []
    for query_type, val1, val2 in queries:
        if query_type == "count":
            compressed_point = compressor.get_compressed(val1)
            results.append(imos[compressed_point])
    
    return results


# 使用例
def example():
    # 基本的な座標圧縮
    arr = [100, 5, 1000, 10, 5, 100, 10000]
    compressed, reverse_map = coordinate_compression(arr)
    
    print("元の配列:", arr)
    print("圧縮後の配列:", compressed)
    print("元の値へのマッピング:", reverse_map)
    
    # 2次元座標の圧縮
    points = [(100, 200), (50, 30), (100, 30), (200, 100)]
    compressed_points, x_map, y_map = compress_2d_points(points)
    
    print("\n元の点:", points)
    print("圧縮後の点:", compressed_points)
    print("X座標マッピング:", x_map)
    print("Y座標マッピング:", y_map)
    
    # CoordinateCompressorクラスの使用例
    compressor = CoordinateCompressor()
    compressor.add_all([100, 5, 1000, 10, 5])
    compressor.compress()
    
    print("\nCompressorを使った圧縮:")
    for val in [5, 10, 100, 1000]:
        print(f"{val} -> {compressor.get_compressed(val)}")
    
    # 実用的な例: 範囲と点のクエリ
    ranges = [(10, 30), (20, 40), (30, 50)]
    queries = [
        ("count", 25, 0),  # 25は何個の範囲に含まれる？
        ("count", 10, 0),  # 10は？
        ("count", 40, 0)   # 40は？
    ]
    
    results = range_queries_with_compression(ranges, queries)
    print("\n範囲クエリ結果:")
    for i, (query_type, val, _) in enumerate(queries):
        if query_type == "count":
            print(f"点 {val} は {results[i]} 個の範囲に含まれる")


if __name__ == "__main__":
    example()
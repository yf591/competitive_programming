"""
セグメント木（Segment Tree）
- 配列に対する区間クエリと一点更新を効率的に処理するデータ構造
- 主な操作:
  - 区間に対するクエリ処理（区間の最小値、最大値、合計など）
  - 配列の一点更新
- 特徴:
  - 完全二分木で実装
  - 区間クエリ、一点更新ともにO(log N)で実行可能
- 用途:
  - 区間の和、最小値、最大値、GCDなどの計算
  - 区間更新が必要な場合は遅延セグメント木を使用
"""

from typing import List, Callable, TypeVar, Generic, Optional, Union
import operator
from math import ceil, log2

T = TypeVar('T')

class SegmentTree(Generic[T]):
    """
    セグメント木の汎用実装
    
    Attributes:
        n: 元の配列のサイズ（2のべき乗に調整）
        tree: セグメント木を表すリスト
        identity: 演算の単位元
        operation: 区間クエリで使用する二項演算
    """
    
    def __init__(self, arr: List[T], operation: Callable[[T, T], T], identity: T):
        """
        セグメント木の初期化
        
        Args:
            arr: 元の配列
            operation: 区間クエリで使用する二項演算（例: operator.add, min, max）
            identity: 演算の単位元（例: 0（和の場合）、float('inf')（最小値の場合））
        """
        self.n = 1
        while self.n < len(arr):
            self.n *= 2
            
        self.identity = identity
        self.operation = operation
        
        # セグメント木の初期化（サイズは2n-1）
        self.tree = [identity] * (2 * self.n - 1)
        
        # 葉ノードに値を設定
        for i, val in enumerate(arr):
            self.tree[i + self.n - 1] = val
            
        # 親ノードを構築
        for i in range(self.n - 2, -1, -1):
            self.tree[i] = self.operation(self.tree[2 * i + 1], self.tree[2 * i + 2])
    
    def update(self, idx: int, val: T):
        """
        インデックスidxの値をvalに更新
        
        Args:
            idx: 更新するインデックス
            val: 新しい値
        """
        # 葉ノードの位置
        idx = idx + self.n - 1
        self.tree[idx] = val
        
        # 親ノードを更新
        while idx > 0:
            idx = (idx - 1) // 2
            self.tree[idx] = self.operation(self.tree[2 * idx + 1], self.tree[2 * idx + 2])
    
    def query(self, left: int, right: int) -> T:
        """
        区間[left, right)に対するクエリを処理
        
        Args:
            left: 区間の左端（含む）
            right: 区間の右端（含まない）
            
        Returns:
            区間に対するクエリ結果
        """
        return self._query_recursive(left, right, 0, 0, self.n)
    
    def _query_recursive(self, left: int, right: int, node: int, node_left: int, node_right: int) -> T:
        """
        再帰的にクエリを処理
        
        Args:
            left, right: クエリの区間[left, right)
            node: 現在のノードのインデックス
            node_left, node_right: ノードが担当する区間[node_left, node_right)
        """
        # クエリ区間外
        if right <= node_left or left >= node_right:
            return self.identity
            
        # ノード区間がクエリ区間に完全に含まれる
        if left <= node_left and node_right <= right:
            return self.tree[node]
            
        # 部分的に重複する場合、再帰的に探索
        mid = (node_left + node_right) // 2
        left_result = self._query_recursive(left, right, 2 * node + 1, node_left, mid)
        right_result = self._query_recursive(left, right, 2 * node + 2, mid, node_right)
        
        return self.operation(left_result, right_result)
        
    def __getitem__(self, idx: int) -> T:
        """
        配列のようにインデックスでアクセス可能にする
        
        Args:
            idx: アクセスするインデックス
            
        Returns:
            インデックスidxの値
        """
        if idx < 0 or idx >= self.n:
            raise IndexError("Index out of range")
        return self.tree[idx + self.n - 1]
    
    def __setitem__(self, idx: int, val: T):
        """
        配列のように値を更新可能にする
        
        Args:
            idx: 更新するインデックス
            val: 新しい値
        """
        self.update(idx, val)


class RangeMinimumQuery(SegmentTree[int]):
    """
    Range Minimum Query (RMQ) のセグメント木
    区間の最小値クエリを処理する
    """
    
    def __init__(self, arr: List[int]):
        super().__init__(arr, min, float('inf'))


class RangeSumQuery(SegmentTree[int]):
    """
    Range Sum Query (RSQ) のセグメント木
    区間の合計クエリを処理する
    """
    
    def __init__(self, arr: List[int]):
        super().__init__(arr, operator.add, 0)


class RangeMaximumQuery(SegmentTree[int]):
    """
    Range Maximum Query のセグメント木
    区間の最大値クエリを処理する
    """
    
    def __init__(self, arr: List[int]):
        super().__init__(arr, max, float('-inf'))


class RangeGCDQuery(SegmentTree[int]):
    """
    Range GCD Query のセグメント木
    区間のGCDを計算する
    """
    
    def __init__(self, arr: List[int]):
        import math
        super().__init__(arr, math.gcd, 0)
        
        # GCDの場合、単位元を初期値に依存させる必要がある場合がある
        if arr:
            idx = self.n - 1
            self.tree[idx] = arr[0]
            for i in range(1, min(len(arr), self.n)):
                idx += 1
                self.tree[idx] = arr[i]
            
            for i in range(self.n - 2, -1, -1):
                self.tree[i] = math.gcd(self.tree[2 * i + 1], self.tree[2 * i + 2])


# 使用例
def example():
    arr = [5, 3, 7, 9, 1, 4, 6, 2]
    
    print("===== Range Minimum Query =====")
    rmq = RangeMinimumQuery(arr)
    print(f"元の配列: {arr}")
    print(f"区間[1, 5)の最小値: {rmq.query(1, 5)}")  # 1
    rmq.update(2, 0)  # インデックス2の値を0に更新
    print(f"更新後の区間[1, 5)の最小値: {rmq.query(1, 5)}")  # 0
    
    print("\n===== Range Sum Query =====")
    rsq = RangeSumQuery(arr)
    print(f"元の配列: {arr}")
    print(f"区間[1, 5)の合計: {rsq.query(1, 5)}")  # 3 + 7 + 9 + 1 = 20
    rsq.update(2, 0)  # インデックス2の値を0に更新
    print(f"更新後の区間[1, 5)の合計: {rsq.query(1, 5)}")  # 3 + 0 + 9 + 1 = 13
    
    print("\n===== Range Maximum Query =====")
    rmax = RangeMaximumQuery(arr)
    print(f"元の配列: {arr}")
    print(f"区間[1, 5)の最大値: {rmax.query(1, 5)}")  # 9
    rmax.update(3, 10)  # インデックス3の値を10に更新
    print(f"更新後の区間[1, 5)の最大値: {rmax.query(1, 5)}")  # 10
    
    print("\n===== 汎用セグメント木 =====")
    # XORのセグメント木
    xor_segtree = SegmentTree(arr, operator.xor, 0)
    print(f"元の配列: {arr}")
    print(f"区間[1, 5)のXOR: {xor_segtree.query(1, 5)}")
    xor_segtree.update(2, 10)
    print(f"更新後の区間[1, 5)のXOR: {xor_segtree.query(1, 5)}")


if __name__ == "__main__":
    example()
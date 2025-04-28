"""
フェニック木（Binary Indexed Tree / BIT）
- 配列の部分和と一点更新を効率的に処理するデータ構造
- セグメント木より実装が単純で、より少ないメモリ使用量
- 主な操作:
  - 配列の先頭からインデックスiまでの和（累積和）
  - 配列の要素の更新
- 特徴:
  - 累積和、一点更新ともにO(log N)で実行可能
  - メモリ効率が良い（元の配列と同じサイズのみ必要）
- 用途:
  - 動的な累積和の計算
  - 区間の和（区間[l,r]の和 = sum(0,r) - sum(0,l-1)）
  - 転倒数の計算
"""

from typing import List, Optional


class BinaryIndexedTree:
    """
    Binary Indexed Tree (BIT) / Fenwick Tree
    
    Attributes:
        n: 配列のサイズ
        tree: BITを表すリスト（1-indexedで使用）
    """
    
    def __init__(self, n: int):
        """
        サイズnのBITを初期化
        
        Args:
            n: 配列のサイズ
        """
        self.n = n
        self.tree = [0] * (n + 1)  # 1-indexedで使用
    
    def __init__(self, arr: Optional[List[int]] = None):
        """
        配列からBITを初期化、または空のBITを作成
        
        Args:
            arr: 初期化する配列（省略可能）
        """
        if arr:
            self.n = len(arr)
            self.tree = [0] * (self.n + 1)  # 1-indexedで使用
            
            # 配列を使って初期化
            for i, val in enumerate(arr):
                self.add(i, val)
        else:
            self.n = 0
            self.tree = []
    
    def add(self, idx: int, val: int):
        """
        インデックスidxの要素にvalを加算
        
        Args:
            idx: 更新するインデックス（0-indexed）
            val: 加算する値
        """
        idx += 1  # 1-indexedに変換
        while idx <= self.n:
            self.tree[idx] += val
            idx += idx & -idx  # 最下位ビットを加算
    
    def sum(self, idx: int) -> int:
        """
        区間[0, idx]の和を計算（0-indexed）
        
        Args:
            idx: 右端のインデックス（含む）
            
        Returns:
            区間[0, idx]の和
        """
        if idx < 0:
            return 0
            
        idx = min(idx + 1, self.n)  # 1-indexedに変換、範囲を確認
        result = 0
        while idx > 0:
            result += self.tree[idx]
            idx -= idx & -idx  # 最下位ビットを減算
        return result
    
    def range_sum(self, left: int, right: int) -> int:
        """
        区間[left, right]の和を計算（0-indexed）
        
        Args:
            left: 左端のインデックス（含む）
            right: 右端のインデックス（含む）
            
        Returns:
            区間[left, right]の和
        """
        return self.sum(right) - self.sum(left - 1)
    
    def __getitem__(self, idx: int) -> int:
        """
        インデックスidxの要素を取得（point queryと同じ）
        
        Args:
            idx: 取得するインデックス
            
        Returns:
            インデックスidxの値
        """
        return self.range_sum(idx, idx)
    
    def __setitem__(self, idx: int, val: int):
        """
        インデックスidxの要素をvalに設定
        
        Args:
            idx: 設定するインデックス
            val: 新しい値
        """
        current = self[idx]
        self.add(idx, val - current)
    
    def get_array(self) -> List[int]:
        """
        BITから元の配列を復元
        
        Returns:
            元の配列
        """
        result = [0] * self.n
        for i in range(self.n):
            result[i] = self.range_sum(i, i)
        return result


class RangeAddPointQuery:
    """
    区間加算・一点取得のBIT
    通常のBITを2つ使って、区間加算をO(log N)で実現
    """
    
    def __init__(self, n: int):
        """
        サイズnのBITを初期化
        
        Args:
            n: 配列のサイズ
        """
        self.n = n
        self.bit1 = BinaryIndexedTree(n)  # 値を保存
        self.bit2 = BinaryIndexedTree(n)  # i*a_iを保存
    
    def add(self, left: int, right: int, val: int):
        """
        区間[left, right]に値valを加算
        
        Args:
            left: 左端のインデックス（含む）
            right: 右端のインデックス（含む）
            val: 加算する値
        """
        self.bit1.add(left, val)
        self.bit1.add(right + 1, -val)
        self.bit2.add(left, val * left)
        self.bit2.add(right + 1, -val * (right + 1))
    
    def sum(self, idx: int) -> int:
        """
        インデックスidxまでの和を計算（区間加算を考慮）
        
        Args:
            idx: インデックス（0-indexed）
            
        Returns:
            インデックスidxの値
        """
        return self.bit1.sum(idx) * (idx + 1) - self.bit2.sum(idx)
    
    def __getitem__(self, idx: int) -> int:
        """
        インデックスidxの要素を取得
        
        Args:
            idx: 取得するインデックス
            
        Returns:
            インデックスidxの値
        """
        return self.sum(idx) - self.sum(idx - 1)


# 使用例
def example():
    # 基本的なBITの使用例
    print("===== Binary Indexed Tree (BIT) =====")
    arr = [3, 1, 4, 1, 5, 9, 2, 6]
    bit = BinaryIndexedTree(arr)
    
    print(f"元の配列: {arr}")
    print(f"BIT: {bit.tree[1:]}")
    print(f"区間[0, 3]の和: {bit.range_sum(0, 3)}")  # 3+1+4+1 = 9
    print(f"区間[2, 5]の和: {bit.range_sum(2, 5)}")  # 4+1+5+9 = 19
    
    # 要素の更新
    bit.add(3, 5)  # インデックス3の値を+5する
    print(f"インデックス3の値を+5した後:")
    print(f"区間[0, 3]の和: {bit.range_sum(0, 3)}")  # 3+1+4+(1+5) = 14
    print(f"区間[2, 5]の和: {bit.range_sum(2, 5)}")  # 4+(1+5)+5+9 = 24
    
    # 元の配列の復元
    print(f"BITから復元した配列: {bit.get_array()}")
    
    # 配列の要素を直接設定
    bit[2] = 10  # インデックス2の値を10に設定
    print(f"インデックス2の値を10に設定した後:")
    print(f"区間[0, 3]の和: {bit.range_sum(0, 3)}")  # 3+1+10+(1+5) = 20
    
    # 区間加算・一点取得のBIT
    print("\n===== 区間加算・一点取得のBIT =====")
    n = 8
    rabit = RangeAddPointQuery(n)
    
    # 初期値は全て0
    print(f"初期状態での値:")
    print([rabit[i] for i in range(n)])  # [0, 0, 0, 0, 0, 0, 0, 0]
    
    # 区間[1, 3]に5を加算
    rabit.add(1, 3, 5)
    print(f"区間[1, 3]に5を加算した後:")
    print([rabit[i] for i in range(n)])  # [0, 5, 5, 5, 0, 0, 0, 0]
    
    # 区間[2, 5]に3を加算
    rabit.add(2, 5, 3)
    print(f"区間[2, 5]に3を加算した後:")
    print([rabit[i] for i in range(n)])  # [0, 5, 8, 8, 3, 3, 0, 0]
    
    # インデックス3の値を確認
    print(f"インデックス3の値: {rabit[3]}")  # 5 + 3 = 8


if __name__ == "__main__":
    example()
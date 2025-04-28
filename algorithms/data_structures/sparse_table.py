"""
Sparse Table - 静的な区間クエリ（最小値、最大値、GCD）の高速計算のためのデータ構造
- 構築時間: O(n log n)
- クエリ時間: O(1)
- メモリ: O(n log n)
"""

import math

class SparseTable:
    def __init__(self, arr, func=min):
        """
        Sparse Table を構築する
        
        Parameters:
        arr (list): 元の配列
        func (function): 区間に適用する関数（デフォルトはmin）
        
        Note: funcはモノイドである必要があります（min, max, gcd など）
        """
        self.n = len(arr)
        self.func = func
        self.log_table = [0] * (self.n + 1)
        for i in range(2, self.n + 1):
            self.log_table[i] = self.log_table[i // 2] + 1
        
        self.k = self.log_table[self.n] + 1 if self.n > 0 else 0
        self.table = [[0] * self.n for _ in range(self.k)]
        
        # スパーステーブルを構築
        for i in range(self.n):
            self.table[0][i] = arr[i]
        
        for i in range(1, self.k):
            j = 0
            while j + (1 << i) <= self.n:
                self.table[i][j] = self.func(
                    self.table[i-1][j], 
                    self.table[i-1][j + (1 << (i-1))]
                )
                j += 1
    
    def query(self, left, right):
        """
        区間[left, right)のクエリを処理する
        
        Parameters:
        left (int): 区間の左端（含む）
        right (int): 区間の右端（含まない）
        
        Returns:
        クエリの結果
        """
        if left >= right:
            raise ValueError("左端は右端より小さくなければなりません")
            
        # 区間の長さの2の対数を計算
        length = right - left
        k = self.log_table[length]
        
        # 区間の両端から2^k個の要素を取り、関数を適用する
        return self.func(
            self.table[k][left],
            self.table[k][right - (1 << k)]
        )


# 使用例
if __name__ == "__main__":
    # 最小値のクエリ
    arr = [1, 3, 4, 8, 6, 1, 4, 2]
    min_table = SparseTable(arr, min)
    print(min_table.query(0, 4))  # 区間[0, 4)の最小値: 1
    print(min_table.query(2, 7))  # 区間[2, 7)の最小値: 1
    
    # 最大値のクエリ
    max_table = SparseTable(arr, max)
    print(max_table.query(0, 4))  # 区間[0, 4)の最大値: 8
    print(max_table.query(2, 7))  # 区間[2, 7)の最大値: 8
    
    # GCDのクエリ
    import math
    gcd_table = SparseTable(arr, math.gcd)
    print(gcd_table.query(0, 4))  # 区間[0, 4)のGCD: 1
    print(gcd_table.query(2, 7))  # 区間[2, 7)のGCD: 1
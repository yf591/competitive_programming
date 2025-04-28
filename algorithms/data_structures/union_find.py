"""
UnionFind（素集合データ構造）
- 要素のグループ化を効率的に管理するデータ構造
- 2つの主要な操作：
  1. unite(x, y): 要素xと要素yを含む集合を併合
  2. find(x): 要素xが属する集合の代表元を返す
- その他の操作：
  - same(x, y): 要素xと要素yが同じ集合に属するか判定
  - size(x): 要素xが属する集合のサイズを取得
- 計算量：
  - ほぼO(α(n))（αはアッカーマン関数の逆関数でほぼ定数）
"""

class UnionFind:
    def __init__(self, n):
        """n要素で初期化"""
        self.n = n
        self.parent = [-1] * n  # 親要素のインデックス（負の場合は自身が根で、絶対値がサイズ）
    
    def find(self, x):
        """要素xが属するグループの根を返す"""
        if self.parent[x] < 0:
            return x
        else:
            # 経路圧縮: 検索経路上の全ての要素の親を根に直接つなぐ
            self.parent[x] = self.find(self.parent[x])
            return self.parent[x]
    
    def unite(self, x, y):
        """要素x, yの属するグループを統合"""
        x = self.find(x)
        y = self.find(y)
        
        if x == y:
            return
        
        # サイズが大きい方に小さい方をマージする
        if self.parent[x] > self.parent[y]:
            x, y = y, x
        
        self.parent[x] += self.parent[y]
        self.parent[y] = x
    
    def same(self, x, y):
        """要素x, yが同じグループに属するかどうか"""
        return self.find(x) == self.find(y)
    
    def size(self, x):
        """要素xが属するグループのサイズ"""
        return -self.parent[self.find(x)]
    
    def groups(self):
        """全てのグループのリストを返す"""
        roots = {}
        for i in range(self.n):
            root = self.find(i)
            if root not in roots:
                roots[root] = []
            roots[root].append(i)
        return list(roots.values())


# 使用例
def example_usage():
    uf = UnionFind(8)  # 0~7の8個の要素
    uf.unite(0, 1)
    uf.unite(1, 2)
    uf.unite(3, 4)
    uf.unite(5, 6)
    uf.unite(6, 7)
    uf.unite(2, 5)
    
    print(uf.same(0, 7))  # True: 0と7は同じグループ
    print(uf.same(0, 3))  # False: 0と3は異なるグループ
    print(uf.size(0))     # 6: 0が属するグループのサイズ（0,1,2,5,6,7）
    print(uf.groups())    # [[0, 1, 2, 5, 6, 7], [3, 4]]: 全てのグループのリスト

# テスト
if __name__ == "__main__":
    example_usage()
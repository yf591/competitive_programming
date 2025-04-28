"""
クラスカル法（Kruskal's Algorithm）
- 最小全域木（Minimum Spanning Tree, MST）を求めるアルゴリズム
- 辺をコストの昇順にソートし、サイクルを作らないように選択していく貪欲法
- UnionFind（素集合データ構造）を用いて効率的に実装可能
- 主な用途:
  - ネットワーク設計（最小コストでの接続）
  - クラスタリング
- 計算量:
  - O(E log E) または O(E log V)（Vは頂点数、Eは辺数）
"""

import sys
sys.path.append('c:\\Users\\yf591\\yoshidev2\\competitive_programming\\algorithms')
from data_structures.union_find import UnionFind
from typing import List, Tuple, Set


def kruskal(n: int, edges: List[Tuple[int, int, int]]) -> Tuple[int, List[Tuple[int, int, int]]]:
    """
    クラスカル法による最小全域木の計算
    
    Args:
        n: ノードの数
        edges: 辺のリスト。各辺は(u, v, cost)の形式で、uとvを結ぶコストcostの辺を表す
        
    Returns:
        total_cost: 最小全域木の総コスト
        mst_edges: 最小全域木を構成する辺のリスト
    """
    # 辺をコストの昇順にソート
    edges.sort(key=lambda x: x[2])
    
    # UnionFindの初期化
    uf = UnionFind(n)
    
    mst_edges = []  # 最小全域木を構成する辺
    total_cost = 0  # 最小全域木の総コスト
    
    # 各辺について考慮
    for u, v, cost in edges:
        # 辺(u, v)の追加がサイクルを形成しない場合
        if not uf.same(u, v):
            uf.unite(u, v)  # 連結する
            mst_edges.append((u, v, cost))  # 最小全域木に追加
            total_cost += cost  # コストを加算
    
    return total_cost, mst_edges


def boruvka(n: int, edges: List[Tuple[int, int, int]]) -> Tuple[int, List[Tuple[int, int, int]]]:
    """
    ボルーフカのアルゴリズムによる最小全域木の計算
    クラスカル法の代替アルゴリズム、特に分散環境で有効
    
    Args:
        n: ノードの数
        edges: 辺のリスト。各辺は(u, v, cost)の形式
        
    Returns:
        total_cost: 最小全域木の総コスト
        mst_edges: 最小全域木を構成する辺のリスト
    """
    # UnionFindの初期化
    uf = UnionFind(n)
    
    # 辺の情報を整理（辺のインデックスを記録）
    indexed_edges = [(u, v, cost, i) for i, (u, v, cost) in enumerate(edges)]
    
    mst_edges_idx = set()  # 最小全域木の辺のインデックス
    total_cost = 0
    
    # 全ての頂点が連結されるまで繰り返す
    while uf.size(0) < n:
        # 各連結成分に対する最小コストの辺を見つける
        cheapest = [-1] * n  # cheapest[i]: 連結成分iに対する最小コストの辺のインデックス
        
        for u, v, cost, idx in indexed_edges:
            if idx in mst_edges_idx:  # 既に使用された辺はスキップ
                continue
                
            root_u = uf.find(u)
            root_v = uf.find(v)
            
            if root_u != root_v:  # 異なる連結成分を結ぶ辺
                if cheapest[root_u] == -1 or cost < edges[cheapest[root_u]][2]:
                    cheapest[root_u] = idx
                if cheapest[root_v] == -1 or cost < edges[cheapest[root_v]][2]:
                    cheapest[root_v] = idx
        
        # 各連結成分について、最小の辺を追加
        added_edges = set()
        for i in range(n):
            if cheapest[i] != -1 and cheapest[i] not in added_edges:
                u, v, cost = edges[cheapest[i]]
                if not uf.same(u, v):  # まだ連結されていない場合
                    uf.unite(u, v)
                    mst_edges_idx.add(cheapest[i])
                    added_edges.add(cheapest[i])
                    total_cost += cost
        
        if not added_edges:  # 新たに追加された辺がない場合（もう改善できない）
            break
    
    # 最小全域木を構成する辺のリストを作成
    mst_edges = [edges[idx] for idx in mst_edges_idx]
    
    return total_cost, mst_edges


def prim(graph: List[List[Tuple[int, int]]], start: int = 0) -> Tuple[int, List[Tuple[int, int, int]]]:
    """
    プリム法による最小全域木の計算
    ダイクストラ法に似た方法で最小全域木を計算する
    
    Args:
        graph: 隣接リスト表現されたグラフ
               graph[u]は(v, cost)のリストで、uとvの間にコストcostの辺が存在することを表す
        start: 開始ノード
        
    Returns:
        total_cost: 最小全域木の総コスト
        mst_edges: 最小全域木を構成する辺のリスト
    """
    import heapq
    n = len(graph)
    visited = [False] * n
    mst_edges = []
    total_cost = 0
    
    # 最小コストの辺を探すための優先度付きキュー
    # (コスト, 現在のノード, 隣接ノード)
    pq = [(0, start, -1)]  # 初期値は、コスト0のダミーエッジ
    
    while pq:
        cost, v, prev = heapq.heappop(pq)
        
        if visited[v]:
            continue
            
        visited[v] = True
        
        if prev != -1:  # 初期ノード以外
            mst_edges.append((prev, v, cost))
            total_cost += cost
        
        # 隣接ノードをキューに追加
        for next_v, next_cost in graph[v]:
            if not visited[next_v]:
                heapq.heappush(pq, (next_cost, next_v, v))
    
    return total_cost, mst_edges


# 使用例
def example():
    # 無向グラフの辺とコストのリスト（(u, v, cost)の形式）
    edges = [
        (0, 1, 7),
        (0, 2, 9),
        (0, 5, 14),
        (1, 2, 10),
        (1, 3, 15),
        (2, 3, 11),
        (2, 5, 2),
        (3, 4, 6),
        (4, 5, 9)
    ]
    
    n = 6  # ノード数
    
    print("===== クラスカル法 =====")
    total_cost, mst = kruskal(n, edges.copy())
    print(f"最小全域木の総コスト: {total_cost}")
    print("最小全域木を構成する辺:")
    for u, v, cost in mst:
        print(f"{u} -- {v}: コスト {cost}")
    
    print("\n===== ボルーフカのアルゴリズム =====")
    total_cost, mst = boruvka(n, edges.copy())
    print(f"最小全域木の総コスト: {total_cost}")
    print("最小全域木を構成する辺:")
    for u, v, cost in mst:
        print(f"{u} -- {v}: コスト {cost}")
    
    # 隣接リスト形式のグラフを構築（プリム法用）
    graph = [[] for _ in range(n)]
    for u, v, cost in edges:
        graph[u].append((v, cost))
        graph[v].append((u, cost))  # 無向グラフなので両方向に追加
    
    print("\n===== プリム法 =====")
    total_cost, mst = prim(graph)
    print(f"最小全域木の総コスト: {total_cost}")
    print("最小全域木を構成する辺:")
    for u, v, cost in mst:
        print(f"{u} -- {v}: コスト {cost}")


if __name__ == "__main__":
    example()
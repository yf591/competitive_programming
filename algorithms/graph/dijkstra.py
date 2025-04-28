"""
ダイクストラ法（Dijkstra's Algorithm）
- 単一始点最短経路問題を解くアルゴリズム
- 辺の重みが非負の場合に使用可能
- 優先度付きキュー（ヒープ）を用いて効率的に実装可能
- 主な用途:
  - 最短経路の計算
  - 経路探索
  - ネットワークルーティング
- 計算量:
  - 優先度付きキュー使用: O(E log V)（Vは頂点数、Eは辺数）
  - 素朴な実装: O(V^2)
"""

import heapq
from typing import List, Tuple, Dict, Optional


def dijkstra(graph: List[List[Tuple[int, int]]], start: int) -> Tuple[List[int], List[int]]:
    """
    ダイクストラ法による単一始点最短経路の計算
    
    Args:
        graph: 隣接リスト表現された重み付きグラフ
              graph[u]は(v, cost)のリストで、uからvへコストcostの辺が存在することを表す
        start: 始点のノード番号
        
    Returns:
        dist: 始点から各ノードへの最短距離
        prev: 最短経路木の親ノード（経路復元用）
    """
    n = len(graph)
    dist = [float('inf')] * n  # 始点からの距離
    prev = [-1] * n  # 最短経路の直前のノード
    dist[start] = 0
    
    # (距離, ノード)のタプルを要素とする優先度付きキュー
    pq = [(0, start)]  
    
    while pq:
        # 未処理のノードのうち、始点からの距離が最小のものを取り出す
        d, v = heapq.heappop(pq)
        
        # 既に処理済みのノードは無視
        if d > dist[v]:
            continue
            
        # 隣接するノードの距離を更新
        for next_v, weight in graph[v]:
            if dist[v] + weight < dist[next_v]:
                dist[next_v] = dist[v] + weight
                prev[next_v] = v
                heapq.heappush(pq, (dist[next_v], next_v))
    
    return dist, prev


def reconstruct_path(prev: List[int], start: int, end: int) -> List[int]:
    """
    最短経路の復元
    
    Args:
        prev: ダイクストラ法で計算した最短経路木
        start: 始点
        end: 終点
        
    Returns:
        path: 最短経路を構成するノードのリスト（start -> end）
    """
    if prev[end] == -1 and end != start:
        return []  # 経路が存在しない
    
    path = []
    current = end
    while current != -1:
        path.append(current)
        current = prev[current]
    
    return path[::-1]  # 逆順にして始点から終点の順に並べる


def dijkstra_grid(grid: List[List[int]], start_row: int, start_col: int) -> List[List[int]]:
    """
    グリッド上でのダイクストラ法
    
    Args:
        grid: コストを表す2次元グリッド（grid[r][c]は座標(r,c)のコスト）
        start_row, start_col: 開始位置の座標
        
    Returns:
        dist: 始点からの最短距離を表す2次元配列
    """
    h, w = len(grid), len(grid[0])
    dist = [[float('inf')] * w for _ in range(h)]
    dist[start_row][start_col] = grid[start_row][start_col]  # 開始地点のコスト
    
    # (距離, 行, 列)のタプルを要素とする優先度付きキュー
    pq = [(grid[start_row][start_col], start_row, start_col)]
    
    # 上下左右の移動方向
    dx = [0, 1, 0, -1]
    dy = [-1, 0, 1, 0]
    
    while pq:
        d, r, c = heapq.heappop(pq)
        
        # 既に処理済みの位置は無視
        if d > dist[r][c]:
            continue
            
        # 上下左右の4方向を調べる
        for i in range(4):
            nr, nc = r + dy[i], c + dx[i]  # 次の位置
            
            # グリッド内かつ未処理の場合
            if 0 <= nr < h and 0 <= nc < w:
                new_dist = dist[r][c] + grid[nr][nc]  # 新しい距離
                
                if new_dist < dist[nr][nc]:
                    dist[nr][nc] = new_dist
                    heapq.heappush(pq, (new_dist, nr, nc))
    
    return dist


def dijkstra_with_path(graph: List[List[Tuple[int, int]]], start: int, end: int) -> Tuple[int, List[int]]:
    """
    ダイクストラ法による最短経路の計算と経路復元
    
    Args:
        graph: 隣接リスト表現された重み付きグラフ
        start: 始点のノード番号
        end: 終点のノード番号
        
    Returns:
        distance: 最短距離
        path: 最短経路を構成するノードのリスト
    """
    dist, prev = dijkstra(graph, start)
    path = reconstruct_path(prev, start, end)
    
    if not path:
        return float('inf'), []
    
    return dist[end], path


# 使用例
def example():
    # 0-indexedの重み付き有向グラフ（隣接リスト表現）
    # graph[u] = [(v, cost), ...] は u->v のコストが cost であることを表す
    graph = [
        [(1, 7), (2, 9), (5, 14)],  # ノード0の隣接ノードとコスト
        [(0, 7), (2, 10), (3, 15)], # ノード1の隣接ノードとコスト
        [(0, 9), (1, 10), (3, 11), (5, 2)], # ノード2の隣接ノードとコスト
        [(1, 15), (2, 11), (4, 6)], # ノード3の隣接ノードとコスト
        [(3, 6), (5, 9)],           # ノード4の隣接ノードとコスト
        [(0, 14), (2, 2), (4, 9)]   # ノード5の隣接ノードとコスト
    ]
    
    start_node = 0
    end_node = 4
    
    print("===== ダイクストラ法 =====")
    dist, prev = dijkstra(graph, start_node)
    print(f"ノード{start_node}からの最短距離:")
    for i, d in enumerate(dist):
        print(f"ノード{i}: {d}")
    
    print("\n===== 経路復元 =====")
    path = reconstruct_path(prev, start_node, end_node)
    if path:
        print(f"ノード{start_node}からノード{end_node}への最短経路: {' -> '.join(map(str, path))}")
        print(f"最短距離: {dist[end_node]}")
    else:
        print(f"ノード{start_node}からノード{end_node}への経路はありません")
    
    # グリッド上のダイクストラ法の例
    grid = [
        [1, 3, 1, 2, 8, 1],
        [4, 1, 2, 1, 5, 2],
        [1, 5, 3, 4, 1, 3],
        [2, 3, 2, 1, 3, 2],
        [5, 1, 4, 2, 1, 1]
    ]
    
    print("\n===== グリッド上のダイクストラ法 =====")
    start_row, start_col = 0, 0
    dist_grid = dijkstra_grid(grid, start_row, start_col)
    
    print(f"({start_row}, {start_col})からの最短距離:")
    for row in dist_grid:
        print(" ".join(f"{d:2d}" if d != float('inf') else "∞" for d in row))


if __name__ == "__main__":
    example()
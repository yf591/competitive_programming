"""
フロイド・ワーシャル法（Floyd-Warshall Algorithm）
- 全点対最短経路問題を解くアルゴリズム
- 主な用途:
  - グラフの全ての頂点ペア間の最短距離を求める
  - 経路復元（実際の最短経路を求める）
  - 推移的閉包の計算
- 特徴:
  - 動的計画法に基づくアルゴリズム
  - 負の辺を含むグラフでも動作する（負の閉路がない場合）
  - 密グラフに対して効率的
- 計算量:
  - 時間計算量: O(V^3) (Vは頂点数)
  - 空間計算量: O(V^2)
"""

from typing import List, Tuple, Optional, Union, Dict
import math


def floyd_warshall(n: int, edges: List[Tuple[int, int, int]]) -> List[List[float]]:
    """
    フロイド・ワーシャル法による全点対最短経路の計算
    
    Args:
        n: 頂点の数（0からn-1まで）
        edges: 辺のリスト (from, to, weight)
        
    Returns:
        List[List[float]]: 距離行列（dist[i][j]はiからjへの最短距離）
                           到達不可能な場合はfloat('inf')
    """
    # 距離行列の初期化
    dist = [[float('inf')] * n for _ in range(n)]
    
    # 自分自身への距離は0
    for i in range(n):
        dist[i][i] = 0
    
    # 直接の辺を距離行列に設定
    for u, v, w in edges:
        dist[u][v] = min(dist[u][v], w)  # 多重辺の場合は最小の重みを採用
    
    # フロイド・ワーシャル法の主要部分
    for k in range(n):      # 中継点
        for i in range(n):  # 始点
            for j in range(n):  # 終点
                if dist[i][k] < float('inf') and dist[k][j] < float('inf'):
                    dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
    
    return dist


def floyd_warshall_with_path(n: int, edges: List[Tuple[int, int, int]]) -> Tuple[List[List[float]], List[List[int]]]:
    """
    経路復元機能付きのフロイド・ワーシャル法
    
    Args:
        n: 頂点の数（0からn-1まで）
        edges: 辺のリスト (from, to, weight)
        
    Returns:
        Tuple[List[List[float]], List[List[int]]]: 
            (距離行列, 次の頂点行列)
            next[i][j]はiからjへの最短経路でiの次に訪れる頂点
    """
    # 距離行列の初期化
    dist = [[float('inf')] * n for _ in range(n)]
    next_vertex = [[-1] * n for _ in range(n)]  # -1は経路がないことを示す
    
    # 自分自身への距離は0
    for i in range(n):
        dist[i][i] = 0
        next_vertex[i][i] = i
    
    # 直接の辺を距離行列に設定
    for u, v, w in edges:
        if w < dist[u][v]:  # 多重辺の場合は最小の重みを採用
            dist[u][v] = w
            next_vertex[u][v] = v
    
    # フロイド・ワーシャル法の主要部分
    for k in range(n):      # 中継点
        for i in range(n):  # 始点
            for j in range(n):  # 終点
                if dist[i][k] < float('inf') and dist[k][j] < float('inf'):
                    if dist[i][k] + dist[k][j] < dist[i][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]
                        next_vertex[i][j] = next_vertex[i][k]  # iからjへの経路はiからkへの経路と同じ次の頂点から始まる
    
    return dist, next_vertex


def reconstruct_path(next_vertex: List[List[int]], start: int, end: int) -> List[int]:
    """
    next_vertex行列から最短経路を再構築する
    
    Args:
        next_vertex: 次の頂点行列（floyd_warshall_with_pathの出力）
        start: 開始頂点
        end: 終了頂点
        
    Returns:
        List[int]: 頂点のリスト（経路がない場合は空リスト）
    """
    if next_vertex[start][end] == -1:  # 経路がない
        return []
    
    path = [start]
    while start != end:
        start = next_vertex[start][end]
        path.append(start)
    
    return path


def detect_negative_cycle(dist: List[List[float]], n: int) -> bool:
    """
    負の閉路の検出
    
    Args:
        dist: フロイド・ワーシャル法で計算した距離行列
        n: 頂点の数
        
    Returns:
        bool: 負の閉路が存在する場合はTrue
    """
    # 対角成分に負の値があれば負の閉路が存在する
    for i in range(n):
        if dist[i][i] < 0:
            return True
    
    return False


def transitive_closure(graph: List[List[bool]]) -> List[List[bool]]:
    """
    隣接行列の推移的閉包を計算
    
    Args:
        graph: 隣接行列（graph[i][j]はiからjへの辺があるかどうか）
        
    Returns:
        List[List[bool]]: 推移的閉包
    """
    n = len(graph)
    closure = [row[:] for row in graph]  # グラフをコピー
    
    # フロイド・ワーシャル法の変形
    for k in range(n):
        for i in range(n):
            for j in range(n):
                closure[i][j] = closure[i][j] or (closure[i][k] and closure[k][j])
    
    return closure


def shortest_path_with_constraints(
    n: int,
    edges: List[Tuple[int, int, int]],
    must_visit: List[int]
) -> Tuple[float, List[int]]:
    """
    特定の頂点を必ず通る最短経路を計算
    
    Args:
        n: 頂点の数（0からn-1まで）
        edges: 辺のリスト (from, to, weight)
        must_visit: 必ず通らなければならない頂点のリスト
        
    Returns:
        Tuple[float, List[int]]: (最短距離, 経路)
                                  経路が存在しない場合は (float('inf'), [])
    """
    # フロイド・ワーシャル法で全点対最短経路を計算
    dist, next_vertex = floyd_warshall_with_path(n, edges)
    
    # 必ず通る頂点を含める
    points = [0] + must_visit + [n-1]  # 始点と終点を追加
    m = len(points)
    
    # TSP的な動的計画法で解く（ただし順序は考慮しない）
    # dp[S][i] = 集合Sの頂点を通って、最後にiにいる最短距離
    dp = {}
    
    # 初期化：始点から各must_visitへ
    for i in range(1, m):
        dp[(1 << i, i)] = dist[points[0]][points[i]]
    
    # 集合の大きさを1つずつ増やす
    for size in range(2, m):
        for subset in get_subsets(m, size):
            if not (subset & (1 << 0)):  # 始点は含まない
                for i in range(1, m):
                    if subset & (1 << i):  # iが部分集合に含まれている
                        prev_subset = subset & ~(1 << i)
                        if prev_subset == 0:
                            continue
                        
                        for j in range(m):
                            if prev_subset & (1 << j):  # jが前の部分集合に含まれている
                                if (prev_subset, j) in dp:
                                    current_dist = dp.get((subset, i), float('inf'))
                                    new_dist = dp[(prev_subset, j)] + dist[points[j]][points[i]]
                                    dp[(subset, i)] = min(current_dist, new_dist)
    
    # 全てのmust_visitを通った後の最短距離
    final_subset = (1 << m) - 1  # 全ての頂点を含む
    final_dist = float('inf')
    for i in range(1, m - 1):  # 最後の頂点（終点）以外
        if (final_subset, i) in dp:
            final_dist = min(final_dist, dp[(final_subset, i)] + dist[points[i]][points[m-1]])
    
    # 経路が存在しない場合
    if final_dist == float('inf'):
        return float('inf'), []
    
    # 経路復元（実装省略 - 複雑なため）
    # 完全な経路復元には追加のデータ構造が必要
    
    return final_dist, []


def get_subsets(n: int, size: int) -> List[int]:
    """
    nビットのうち、ちょうどsize個のビットが立っている全ての組み合わせを返す
    """
    def backtrack(start: int, remaining: int, current: int, result: List[int]):
        if remaining == 0:
            result.append(current)
            return
        
        for i in range(start, n):
            backtrack(i + 1, remaining - 1, current | (1 << i), result)
    
    result = []
    backtrack(0, size, 0, result)
    return result


def johnson_algorithm(n: int, edges: List[Tuple[int, int, int]]) -> List[List[float]]:
    """
    ジョンソンのアルゴリズムによる全点対最短経路（疎グラフ向け）
    
    Args:
        n: 頂点の数（0からn-1まで）
        edges: 辺のリスト (from, to, weight)
        
    Returns:
        List[List[float]]: 距離行列
    """
    # この実装は省略します。フロイド・ワーシャル法との比較のために記載しています。
    # 実際の実装には、ベルマンフォード法とダイクストラ法が必要です。
    pass


# 使用例
def example():
    # 頂点は0, 1, 2, 3の4つ
    n = 4
    
    # 辺のリスト (from, to, weight)
    edges = [
        (0, 1, 3),
        (0, 2, 8),
        (0, 3, 4),
        (1, 2, 1),
        (2, 0, 2),
        (3, 2, 1)
    ]
    
    print("===== フロイド・ワーシャル法 =====")
    dist = floyd_warshall(n, edges)
    
    print("距離行列:")
    for row in dist:
        print([d if d != float('inf') else 'inf' for d in row])
    
    print("\n===== 経路復元 =====")
    dist, next_vertex = floyd_warshall_with_path(n, edges)
    
    # 0から2への最短経路
    path = reconstruct_path(next_vertex, 0, 2)
    print(f"0から2への最短経路: {path}")
    print(f"距離: {dist[0][2]}")
    
    # 3から1への最短経路
    path = reconstruct_path(next_vertex, 3, 1)
    print(f"3から1への最短経路: {path}")
    print(f"距離: {dist[3][1]}")
    
    print("\n===== 推移的閉包 =====")
    # 隣接行列
    adj_matrix = [
        [False, True, False, True],
        [False, False, True, False],
        [True, False, False, False],
        [False, False, True, False]
    ]
    
    closure = transitive_closure(adj_matrix)
    print("推移的閉包:")
    for row in closure:
        print(row)


if __name__ == "__main__":
    example()
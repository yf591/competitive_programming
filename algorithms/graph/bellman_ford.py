"""
ベルマン-フォード法（Bellman-Ford Algorithm）
- 単一始点最短経路問題を解くアルゴリズム
- 負の辺が存在するグラフでも使用可能
- 負の閉路（サイクル）の検出も可能
- 主な用途:
  - 負の辺を含むグラフの最短経路の計算
  - 負の閉路の検出
- 計算量:
  - O(V * E)（Vは頂点数、Eは辺数）
"""

from typing import List, Tuple, Dict, Optional


def bellman_ford(edges: List[Tuple[int, int, int]], n: int, start: int) -> Tuple[Optional[List[float]], bool]:
    """
    ベルマン-フォード法による単一始点最短経路の計算
    
    Args:
        edges: 辺のリスト [(u, v, w), ...] - uからvへ重みwの辺
        n: 頂点数
        start: 始点のノード番号
        
    Returns:
        Tuple[Optional[List[float]], bool]:
            - 最短距離のリスト（負の閉路が存在する場合はNone）
            - 負の閉路が存在するかどうかのフラグ
    """
    # 距離を初期化
    dist = [float('inf')] * n
    dist[start] = 0
    
    # 各辺について、n-1回のリラックス操作を行う
    for _ in range(n - 1):
        updated = False
        for u, v, w in edges:
            if dist[u] != float('inf') and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                updated = True
        
        # 更新がなければ早期に終了
        if not updated:
            break
    
    # 負の閉路の検出（もし存在するなら、さらに距離が短くなる）
    has_negative_cycle = False
    for u, v, w in edges:
        if dist[u] != float('inf') and dist[u] + w < dist[v]:
            has_negative_cycle = True
            break
    
    return (None if has_negative_cycle else dist, has_negative_cycle)


def bellman_ford_with_path(edges: List[Tuple[int, int, int]], n: int, start: int, end: int) -> Tuple[Optional[float], Optional[List[int]]]:
    """
    ベルマン-フォード法による単一始点最短経路の計算と経路復元
    
    Args:
        edges: 辺のリスト [(u, v, w), ...] - uからvへ重みwの辺
        n: 頂点数
        start: 始点のノード番号
        end: 終点のノード番号
        
    Returns:
        Tuple[Optional[float], Optional[List[int]]]:
            - 最短距離（負の閉路が存在する場合はNone）
            - 最短経路を構成するノードのリスト（負の閉路が存在する場合はNone）
    """
    # 距離を初期化
    dist = [float('inf')] * n
    dist[start] = 0
    
    # 経路復元用の前任ノードを記録
    prev = [-1] * n
    
    # 各辺について、n-1回のリラックス操作を行う
    for _ in range(n - 1):
        updated = False
        for u, v, w in edges:
            if dist[u] != float('inf') and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                prev[v] = u
                updated = True
        
        # 更新がなければ早期に終了
        if not updated:
            break
    
    # 負の閉路の検出
    for u, v, w in edges:
        if dist[u] != float('inf') and dist[u] + w < dist[v]:
            return None, None  # 負の閉路が存在する
    
    # 経路の復元
    if dist[end] == float('inf'):
        return float('inf'), []  # 到達不可能
    
    path = []
    curr = end
    while curr != -1:
        path.append(curr)
        curr = prev[curr]
    
    return dist[end], path[::-1]  # 経路を反転して返す


def detect_negative_cycle(edges: List[Tuple[int, int, int]], n: int) -> List[int]:
    """
    グラフ内の負の閉路を検出して、閉路を構成するノードを返す
    
    Args:
        edges: 辺のリスト [(u, v, w), ...] - uからvへ重みwの辺
        n: 頂点数
        
    Returns:
        List[int]: 負の閉路を構成するノードのリスト（存在しない場合は空リスト）
    """
    # 距離を初期化（任意の始点から開始）
    dist = [0] * n
    prev = [-1] * n
    
    # n回のリラックス操作を行い、n回目に更新があれば負の閉路が存在する
    x = -1  # 負の閉路の一部であるノード
    for i in range(n):
        x = -1
        for u, v, w in edges:
            if dist[u] != float('inf') and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                prev[v] = u
                x = v
    
    if x == -1:
        return []  # 負の閉路なし
    
    # xからn回辿って、確実に閉路内のノードに到達
    for _ in range(n):
        x = prev[x]
    
    # 閉路を構成するノードを収集
    cycle = []
    curr = x
    while True:
        cycle.append(curr)
        curr = prev[curr]
        if curr == x and len(cycle) > 1:
            break
    
    return cycle[::-1]  # 閉路を反転して返す


# 使用例
def example():
    # ベルマン-フォード法の基本的な使用例
    edges = [
        (0, 1, 5),   # 0 -> 1, 重み 5
        (0, 2, 4),   # 0 -> 2, 重み 4
        (1, 3, 3),   # 1 -> 3, 重み 3
        (2, 1, -6),  # 2 -> 1, 重み -6（負の辺）
        (3, 2, 2),   # 3 -> 2, 重み 2
        (2, 4, 3),   # 2 -> 4, 重み 3
        (4, 3, -3)   # 4 -> 3, 重み -3（負の辺）
    ]
    
    n = 5  # ノード数
    start_node = 0
    end_node = 4
    
    print("===== ベルマン-フォード法 =====")
    distances, has_negative_cycle = bellman_ford(edges, n, start_node)
    
    if has_negative_cycle:
        print("グラフには負の閉路が存在します")
    else:
        print(f"ノード{start_node}からの最短距離:")
        for i, d in enumerate(distances):
            print(f"ノード{i}: {d}")
    
    print("\n===== 経路復元 =====")
    distance, path = bellman_ford_with_path(edges, n, start_node, end_node)
    
    if distance is None:
        print("グラフには負の閉路が存在します")
    elif distance == float('inf'):
        print(f"ノード{start_node}からノード{end_node}への経路は存在しません")
    else:
        print(f"ノード{start_node}からノード{end_node}への最短経路: {' -> '.join(map(str, path))}")
        print(f"最短距離: {distance}")
    
    # 負の閉路を含むグラフ
    negative_cycle_edges = [
        (0, 1, 1),
        (1, 2, 2),
        (2, 3, 3),
        (3, 1, -7)  # 1 -> 2 -> 3 -> 1 で合計 -2 の閉路
    ]
    
    print("\n===== 負の閉路の検出 =====")
    cycle = detect_negative_cycle(negative_cycle_edges, 4)
    
    if cycle:
        print(f"検出された負の閉路: {' -> '.join(map(str, cycle))}")
    else:
        print("負の閉路は検出されませんでした")


if __name__ == "__main__":
    example()
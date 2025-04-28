"""
幅優先探索（Breadth-First Search, BFS）
- グラフの探索アルゴリズムの一つ
- 始点からの距離が近い順に探索を行う
- キューを使って実装される
- 主な用途:
  - 最短経路問題（辺の重みが等しい場合）
  - 連結成分の探索
  - 二部グラフ判定
- 計算量:
  - O(V+E)（Vは頂点数、Eは辺数）
"""

from collections import deque

def bfs(graph, start):
    """
    グラフの幅優先探索を行い、始点からの距離を計算する
    
    Args:
        graph: 隣接リスト表現されたグラフ（graph[v]はノードvから到達可能なノードのリスト）
        start: 開始ノード
    
    Returns:
        dist: 始点からの距離の配列。到達不可能な場合は-1
    """
    n = len(graph)  # ノード数
    dist = [-1] * n  # 距離。-1は未訪問
    dist[start] = 0  # 開始点の距離は0
    
    queue = deque([start])  # 訪問予定のノードを格納するキュー
    
    while queue:
        v = queue.popleft()  # キューから取り出し
        
        # vの隣接ノードをすべて調べる
        for next_v in graph[v]:
            if dist[next_v] == -1:  # 未訪問の場合
                dist[next_v] = dist[v] + 1  # 距離を設定
                queue.append(next_v)  # キューに追加
    
    return dist

def bfs_grid(grid, start_row, start_col):
    """
    グリッド上の幅優先探索を行い、始点からの距離を計算する
    
    Args:
        grid: 2次元グリッド（'.'は通路、'#'は壁）
        start_row, start_col: 開始位置の座標
    
    Returns:
        dist: 始点からの距離の2次元配列。到達不可能な場合は-1
    """
    h, w = len(grid), len(grid[0])  # グリッドの高さと幅
    dist = [[-1] * w for _ in range(h)]  # 距離。-1は未訪問
    dist[start_row][start_col] = 0  # 開始点の距離は0
    
    queue = deque([(start_row, start_col)])  # 訪問予定のノードを格納するキュー
    
    # 上下左右の移動方向
    dx = [0, 1, 0, -1]
    dy = [-1, 0, 1, 0]
    
    while queue:
        r, c = queue.popleft()  # キューから取り出し
        
        # 上下左右の4方向を調べる
        for i in range(4):
            nr, nc = r + dy[i], c + dx[i]  # 次の位置
            
            # グリッド内かつ通路であり、未訪問の場合
            if (0 <= nr < h and 0 <= nc < w and 
                grid[nr][nc] == '.' and dist[nr][nc] == -1):
                dist[nr][nc] = dist[r][c] + 1  # 距離を設定
                queue.append((nr, nc))  # キューに追加
    
    return dist

# 使用例: グラフの場合
def example_graph():
    # 無向グラフの例（0-indexedの隣接リスト表現）
    graph = [
        [1, 2],     # ノード0に隣接するノード
        [0, 3, 4],  # ノード1に隣接するノード
        [0, 5],     # ノード2に隣接するノード
        [1],        # ノード3に隣接するノード
        [1, 6],     # ノード4に隣接するノード
        [2],        # ノード5に隣接するノード
        [4]         # ノード6に隣接するノード
    ]
    
    start = 0  # 開始ノード
    dist = bfs(graph, start)
    
    print(f"ノード{start}からの距離:")
    for i, d in enumerate(dist):
        print(f"ノード{i}: {d}")

# 使用例: グリッドの場合
def example_grid():
    grid = [
        "....#.",
        ".#....",
        "....#.",
        ".#....",
        "......"
    ]
    
    grid = [list(row) for row in grid]  # 文字列から2次元配列に変換
    start_row, start_col = 0, 0  # 開始位置
    
    dist = bfs_grid(grid, start_row, start_col)
    
    print(f"({start_row}, {start_col})からの距離:")
    for row in dist:
        print(" ".join(str(d) for d in row))

# テスト
if __name__ == "__main__":
    print("===== グラフBFSの例 =====")
    example_graph()
    print("\n===== グリッドBFSの例 =====")
    example_grid()
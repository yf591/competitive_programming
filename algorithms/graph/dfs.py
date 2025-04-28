"""
深さ優先探索（Depth-First Search, DFS）
- グラフの探索アルゴリズムの一つ
- 可能な限り深く探索を進め、行き止まりになったら戻るという探索方法
- 再帰またはスタックを使って実装される
- 主な用途:
  - 連結成分の探索
  - トポロジカルソート
  - サイクル検出
  - 強連結成分分解
- 計算量:
  - O(V+E)（Vは頂点数、Eは辺数）
"""

from typing import List, Set, Dict, Tuple, Optional


def dfs(graph: List[List[int]], start: int) -> List[int]:
    """
    再帰によるDFS
    
    Args:
        graph: 隣接リスト表現されたグラフ（graph[v]はノードvから到達可能なノードのリスト）
        start: 開始ノード
        
    Returns:
        visited: 訪問したノードのリスト（訪問順）
    """
    n = len(graph)
    visited = [False] * n  # 訪問済みかどうかのフラグ
    result = []
    
    def dfs_recursive(v: int):
        visited[v] = True
        result.append(v)  # 訪問順にノードを追加
        
        for next_v in graph[v]:
            if not visited[next_v]:
                dfs_recursive(next_v)
    
    dfs_recursive(start)
    return result


def dfs_iterative(graph: List[List[int]], start: int) -> List[int]:
    """
    スタックを使った非再帰的DFS
    
    Args:
        graph: 隣接リスト表現されたグラフ
        start: 開始ノード
        
    Returns:
        visited: 訪問したノードのリスト（訪問順）
    """
    n = len(graph)
    visited = [False] * n
    result = []
    
    stack = [start]  # 探索候補のノードを格納するスタック
    
    while stack:
        v = stack.pop()  # スタックの一番上から取り出す
        
        if visited[v]:
            continue
            
        visited[v] = True
        result.append(v)
        
        # 隣接ノードをスタックに追加（逆順で追加することで、
        # グラフ上での「左から右」の順序で探索できる）
        for next_v in reversed(graph[v]):
            if not visited[next_v]:
                stack.append(next_v)
    
    return result


def dfs_grid(grid: List[List[str]], start_row: int, start_col: int, 
             path_char: str = '.', wall_char: str = '#') -> List[List[bool]]:
    """
    グリッド上のDFS
    
    Args:
        grid: 2次元グリッド（path_charは通路、wall_charは壁）
        start_row, start_col: 開始位置の座標
        path_char: 通路を表す文字
        wall_char: 壁を表す文字
        
    Returns:
        visited: 訪問済みの位置を表す2次元配列
    """
    h, w = len(grid), len(grid[0])  # グリッドの高さと幅
    visited = [[False] * w for _ in range(h)]  # 訪問済みかどうかのフラグ
    
    # 上下左右の移動方向
    dx = [0, 1, 0, -1]
    dy = [-1, 0, 1, 0]
    
    def dfs_recursive(r: int, c: int):
        visited[r][c] = True
        
        # 上下左右の4方向を調べる
        for i in range(4):
            nr, nc = r + dy[i], c + dx[i]  # 次の位置
            
            # グリッド内かつ通路であり、未訪問の場合
            if (0 <= nr < h and 0 <= nc < w and 
                grid[nr][nc] == path_char and not visited[nr][nc]):
                dfs_recursive(nr, nc)
    
    # 開始位置が通路であれば探索開始
    if grid[start_row][start_col] == path_char:
        dfs_recursive(start_row, start_col)
    
    return visited


def topological_sort(graph: List[List[int]]) -> List[int]:
    """
    DFSを用いたトポロジカルソート
    
    Args:
        graph: 隣接リスト表現された有向グラフ
        
    Returns:
        List[int]: トポロジカル順序（DAGでない場合は空リスト）
    """
    n = len(graph)
    visited = [False] * n
    temp_mark = [False] * n  # 一時マーク（サイクル検出用）
    order = []  # トポロジカル順序
    
    def dfs_topo(v: int) -> bool:
        if temp_mark[v]:  # サイクルを検出
            return False
        if visited[v]:
            return True
        
        temp_mark[v] = True  # 一時マークをつける
        
        # 全ての隣接ノードを訪問
        for next_v in graph[v]:
            if not dfs_topo(next_v):
                return False
        
        temp_mark[v] = False  # 一時マークを外す
        visited[v] = True
        order.append(v)  # 帰りがけ順でノードを追加
        return True
    
    # すべてのノードについてDFSを実行
    for i in range(n):
        if not visited[i]:
            if not dfs_topo(i):
                return []  # サイクルが存在する場合
    
    # 帰りがけ順の逆がトポロジカル順序
    return order[::-1]


def find_cycle(graph: List[List[int]]) -> List[int]:
    """
    DFSを用いたサイクル検出
    
    Args:
        graph: 隣接リスト表現されたグラフ
        
    Returns:
        List[int]: サイクルを構成するノードのリスト（存在しない場合は空リスト）
    """
    n = len(graph)
    color = [0] * n  # 0:未訪問, 1:訪問中, 2:訪問済み
    parent = [-1] * n  # 各ノードの親ノード
    cycle_start = -1
    cycle_end = -1
    
    def dfs_cycle(v: int, p: int) -> bool:
        nonlocal cycle_start, cycle_end
        color[v] = 1  # 訪問中
        
        for next_v in graph[v]:
            if next_v == p:  # 親への辺はスキップ（無向グラフの場合）
                continue
                
            if color[next_v] == 0:  # 未訪問
                parent[next_v] = v
                if dfs_cycle(next_v, v):
                    return True
            elif color[next_v] == 1:  # サイクル検出
                cycle_end = v
                cycle_start = next_v
                return True
        
        color[v] = 2  # 訪問済み
        return False
    
    # すべてのノードについてDFSを実行
    for i in range(n):
        if color[i] == 0:
            if dfs_cycle(i, -1):
                break
    
    if cycle_start == -1:
        return []  # サイクルなし
    
    # サイクルを構成するノードを格納
    cycle = []
    cycle.append(cycle_start)
    v = cycle_end
    while v != cycle_start:
        cycle.append(v)
        v = parent[v]
    cycle.append(cycle_start)  # サイクルを閉じる
    
    return cycle


# 使用例
def example():
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
    
    print("===== DFS（再帰版）=====")
    print(f"ノード0からの訪問順: {dfs(graph, 0)}")
    
    print("===== DFS（非再帰版）=====")
    print(f"ノード0からの訪問順: {dfs_iterative(graph, 0)}")
    
    # DFSの応用例：トポロジカルソート
    dag = [
        [1, 2],  # ノード0の出辺
        [3],     # ノード1の出辺
        [3],     # ノード2の出辺
        [4],     # ノード3の出辺
        [],      # ノード4の出辺
        [0, 2]   # ノード5の出辺
    ]
    
    print("\n===== トポロジカルソート =====")
    topo_order = topological_sort(dag)
    if topo_order:
        print(f"トポロジカル順序: {topo_order}")
    else:
        print("DAGではありません（サイクルが存在します）")
    
    # サイクル検出
    cyclic_graph = [
        [1, 2],  # ノード0の隣接ノード
        [3],     # ノード1の隣接ノード
        [3],     # ノード2の隣接ノード
        [4],     # ノード3の隣接ノード
        [1]      # ノード4の隣接ノード（1へのエッジでサイクルが形成される）
    ]
    
    print("\n===== サイクル検出 =====")
    cycle = find_cycle(cyclic_graph)
    if cycle:
        print(f"検出されたサイクル: {cycle}")
    else:
        print("サイクルは存在しません")
    
    # グリッドDFSの例
    grid = [
        list("....#."),
        list(".#...."),
        list("....#."),
        list(".#...."),
        list("......")
    ]
    
    print("\n===== グリッドDFS =====")
    visited = dfs_grid(grid, 0, 0)
    print("訪問済みの位置:")
    for row in visited:
        print("".join("#" if v else "." for v in row))


if __name__ == "__main__":
    example()
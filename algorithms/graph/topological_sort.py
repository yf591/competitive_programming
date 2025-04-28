"""
トポロジカルソート（Topological Sort）
- 有向非巡回グラフ（DAG）の全ての頂点を辺の方向に矛盾しないように一列に並べるアルゴリズム
- 主な用途:
  - タスクスケジューリング
  - 依存関係の解決
  - コンパイラでの宣言順序の決定
- 特徴:
  - サイクルを持つグラフには適用できない
  - 結果は一意でない場合がある
- 計算量:
  - O(V + E)（Vは頂点数、Eは辺数）
"""

from typing import List, Dict, Set, Optional, Deque
from collections import defaultdict, deque


def topological_sort_dfs(graph: Dict[int, List[int]]) -> List[int]:
    """
    DFSを用いたトポロジカルソート
    
    Args:
        graph: 有向グラフの隣接リスト表現（keyはノード、valueはそのノードからの辺の先のノードのリスト）
        
    Returns:
        List[int]: トポロジカル順序で並べられたノードのリスト、サイクルが存在する場合は空リスト
    """
    def dfs(node: int) -> bool:
        """
        ノードをDFSで探索し、後退辺（サイクル）があるかを確認
        
        Args:
            node: 現在のノード
            
        Returns:
            bool: サイクルがない場合はTrue、ある場合はFalse
        """
        # 既に訪問済みのノードに再度来た場合、サイクル
        if node in visiting:
            return False
        
        # 既に処理済みのノードは再訪問しない
        if node in visited:
            return True
        
        visiting.add(node)
        
        # 隣接ノードを探索
        for neighbor in graph.get(node, []):
            if not dfs(neighbor):
                return False
        
        # 探索が完了したノードを処理済みに移動
        visiting.remove(node)
        visited.add(node)
        
        # 結果リストの先頭に挿入（逆順になるため）
        result.insert(0, node)
        return True
    
    # 全てのノードを取得
    all_nodes = set(graph.keys()).union(*[set(neighbors) for neighbors in graph.values()])
    
    visiting = set()  # 現在訪問中のノード（サイクル検出用）
    visited = set()   # 処理済みのノード
    result = []       # トポロジカル順序のノード
    
    # 全てのノードについてDFSを実行
    for node in all_nodes:
        if node not in visited:
            if not dfs(node):
                return []  # サイクルが存在する場合は空リストを返す
    
    return result


def topological_sort_kahn(graph: Dict[int, List[int]]) -> List[int]:
    """
    Kahnのアルゴリズム（入次数ベース）を用いたトポロジカルソート
    
    Args:
        graph: 有向グラフの隣接リスト表現（keyはノード、valueはそのノードからの辺の先のノードのリスト）
        
    Returns:
        List[int]: トポロジカル順序で並べられたノードのリスト、サイクルが存在する場合は空リスト
    """
    # 全てのノードを取得
    all_nodes = set(graph.keys()).union(*[set(neighbors) for neighbors in graph.values()])
    
    # 各ノードの入次数を計算
    in_degree = {node: 0 for node in all_nodes}
    for node in graph:
        for neighbor in graph.get(node, []):
            in_degree[neighbor] = in_degree.get(neighbor, 0) + 1
    
    # 入次数が0のノードをキューに追加
    queue = deque([node for node in all_nodes if in_degree[node] == 0])
    result = []
    
    # キューが空になるまで処理
    while queue:
        node = queue.popleft()
        result.append(node)
        
        # 隣接ノードの入次数を減らし、入次数が0になったらキューに追加
        for neighbor in graph.get(node, []):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    
    # 全てのノードが処理されなかった場合、サイクルが存在する
    return result if len(result) == len(all_nodes) else []


def all_topological_sorts(graph: Dict[int, List[int]]) -> List[List[int]]:
    """
    可能なすべてのトポロジカルソートを生成（小規模グラフ用）
    
    Args:
        graph: 有向グラフの隣接リスト表現
        
    Returns:
        List[List[int]]: すべての可能なトポロジカルソートのリスト
    """
    # 全てのノードを取得
    all_nodes = set(graph.keys()).union(*[set(neighbors) for neighbors in graph.values()])
    
    # 各ノードの入次数を計算
    in_degree = {node: 0 for node in all_nodes}
    for node in graph:
        for neighbor in graph.get(node, []):
            in_degree[neighbor] = in_degree.get(neighbor, 0) + 1
    
    # 結果を格納するリスト
    all_sorts = []
    
    # 現在の順序を保持する配列
    current_sort = []
    
    def backtrack():
        # すべてのノードが処理済みの場合、結果に追加
        if len(current_sort) == len(all_nodes):
            all_sorts.append(current_sort.copy())
            return
        
        # 入次数が0のすべてのノードを試す
        for node in all_nodes:
            if in_degree[node] == 0 and node not in current_sort:
                # ノードを選択
                current_sort.append(node)
                
                # 隣接ノードの入次数を一時的に減らす
                for neighbor in graph.get(node, []):
                    in_degree[neighbor] -= 1
                
                # 再帰的に次のノードを選択
                backtrack()
                
                # バックトラック: ノードを取り消し、入次数を元に戻す
                current_sort.pop()
                for neighbor in graph.get(node, []):
                    in_degree[neighbor] += 1
    
    # バックトラッキング開始
    backtrack()
    
    return all_sorts


def is_dag(graph: Dict[int, List[int]]) -> bool:
    """
    グラフが有向非巡回グラフ（DAG）かどうかを判定
    
    Args:
        graph: 有向グラフの隣接リスト表現
        
    Returns:
        bool: DAGの場合はTrue、サイクルがある場合はFalse
    """
    # トポロジカルソートが可能であればDAG
    return len(topological_sort_kahn(graph)) > 0


def find_cycle(graph: Dict[int, List[int]]) -> List[int]:
    """
    有向グラフ内のサイクルを検出
    
    Args:
        graph: 有向グラフの隣接リスト表現
        
    Returns:
        List[int]: サイクルを構成するノードのリスト（存在しない場合は空リスト）
    """
    # 全てのノードを取得
    all_nodes = set(graph.keys()).union(*[set(neighbors) for neighbors in graph.values()])
    
    # 訪問状態を記録
    # 0: 未訪問, 1: 訪問中, 2: 訪問済み
    state = {node: 0 for node in all_nodes}
    parent = {node: -1 for node in all_nodes}
    
    cycle_start = -1
    cycle_end = -1
    
    def dfs(node: int) -> bool:
        """
        DFSでサイクルを検出
        
        Args:
            node: 現在のノード
            
        Returns:
            bool: サイクルが見つかった場合はTrue
        """
        nonlocal cycle_start, cycle_end
        
        state[node] = 1  # 訪問中
        
        for neighbor in graph.get(node, []):
            if state[neighbor] == 0:
                parent[neighbor] = node
                if dfs(neighbor):
                    return True
            elif state[neighbor] == 1:
                # 訪問中のノードに戻った場合、サイクル発見
                cycle_end = node
                cycle_start = neighbor
                return True
        
        state[node] = 2  # 訪問済み
        return False
    
    # 全ノードについてDFSを実行
    for node in all_nodes:
        if state[node] == 0 and dfs(node):
            break
    
    # サイクルが見つからなかった場合
    if cycle_start == -1:
        return []
    
    # サイクルを復元
    cycle = []
    cycle.append(cycle_start)
    current = cycle_end
    while current != cycle_start:
        cycle.append(current)
        current = parent[current]
    cycle.append(cycle_start)  # サイクルを閉じる
    
    # 正しい順序に並べ替える
    return cycle[::-1]


# 使用例
def example():
    # 単純なDAG
    graph1 = {
        0: [1, 2],
        1: [3],
        2: [3],
        3: [4],
        4: []
    }
    
    # サイクルを持つグラフ
    graph2 = {
        0: [1],
        1: [2],
        2: [0, 3],
        3: [4],
        4: []
    }
    
    print("===== トポロジカルソート（DFS） =====")
    topo_sort_dfs = topological_sort_dfs(graph1)
    print(f"DAGのトポロジカルソート: {topo_sort_dfs}")
    
    topo_sort_dfs_cycle = topological_sort_dfs(graph2)
    print(f"サイクルあり: {'トポロジカル順序なし' if not topo_sort_dfs_cycle else topo_sort_dfs_cycle}")
    
    print("\n===== トポロジカルソート（Kahn） =====")
    topo_sort_kahn = topological_sort_kahn(graph1)
    print(f"DAGのトポロジカルソート: {topo_sort_kahn}")
    
    topo_sort_kahn_cycle = topological_sort_kahn(graph2)
    print(f"サイクルあり: {'トポロジカル順序なし' if not topo_sort_kahn_cycle else topo_sort_kahn_cycle}")
    
    print("\n===== すべてのトポロジカル順序 =====")
    # 小さいグラフで全トポロジカル順序を列挙
    small_graph = {
        0: [2],
        1: [2],
        2: [3],
        3: []
    }
    all_sorts = all_topological_sorts(small_graph)
    print(f"可能なトポロジカル順序の数: {len(all_sorts)}")
    for i, sort_order in enumerate(all_sorts):
        print(f"順序 {i+1}: {sort_order}")
    
    print("\n===== サイクル検出 =====")
    print(f"graph1はDAGか？: {is_dag(graph1)}")
    print(f"graph2はDAGか？: {is_dag(graph2)}")
    
    cycle = find_cycle(graph2)
    if cycle:
        print(f"検出されたサイクル: {' -> '.join(map(str, cycle))}")
    else:
        print("サイクルは検出されませんでした")


if __name__ == "__main__":
    example()
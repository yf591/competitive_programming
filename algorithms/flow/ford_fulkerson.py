"""
フォード・ファルカーソンアルゴリズム（Ford-Fulkerson Algorithm）
- 最大流問題を解くアルゴリズム
- 増加路を繰り返し見つけて流量を増加させる
- 主な用途:
  - ネットワークフロー問題
  - 二部グラフのマッチング
  - 最小カット問題
- 計算量:
  - O(F * E)（Fは最大流量、Eは辺の数）
  - 容量が整数の場合は必ず停止する
"""

from typing import Dict, Set


def ford_fulkerson(graph: Dict[int, Dict[int, int]], source: int, sink: int) -> int:
    """
    フォード・ファルカーソンアルゴリズムによる最大流の計算
    
    Args:
        graph: 隣接リスト表現の容量グラフ
              graph[u][v]はuからvへの辺の容量
        source: 始点（ソース）
        sink: 終点（シンク）
        
    Returns:
        始点から終点への最大流量
    """
    def find_path(u: int, flow: int) -> int:
        """
        深さ優先探索で増加路を見つける
        
        Args:
            u: 現在のノード
            flow: 現在のパスでの最小流量
            
        Returns:
            見つかった増加路での流量（見つからなければ0）
        """
        if u == sink:
            return flow
            
        for v in list(graph.get(u, {})):  # 辞書のキーを変更するためリストにコピー
            capacity = graph[u][v]
            if capacity > 0 and v not in visited:
                visited.add(v)
                min_flow = find_path(v, min(flow, capacity))
                
                if min_flow > 0:
                    # 順方向の容量を減らす
                    graph[u][v] -= min_flow
                    
                    # 逆方向の容量を増やす（残余グラフの構築）
                    if v not in graph:
                        graph[v] = {}
                    if u not in graph[v]:
                        graph[v][u] = 0
                    graph[v][u] += min_flow
                    
                    return min_flow
        
        return 0
    
    # 最大流量
    max_flow = 0
    
    # 増加路が見つかる限り繰り返す
    while True:
        visited = {source}  # 訪問済みノード
        path_flow = find_path(source, float('inf'))
        
        # 増加路が見つからなければ終了
        if path_flow == 0:
            break
        
        max_flow += path_flow
    
    return max_flow


def ford_fulkerson_with_paths(graph: Dict[int, Dict[int, int]], source: int, sink: int) -> tuple:
    """
    最大流とその流れを計算する拡張版フォード・ファルカーソンアルゴリズム
    
    Args:
        graph: 隣接リスト表現の容量グラフ
        source: 始点（ソース）
        sink: 終点（シンク）
        
    Returns:
        (最大流量, フロー辞書)
        フロー辞書は{(u, v): flow}の形式で、各辺の実際の流量を表す
    """
    # グラフのディープコピーを作成
    residual_graph = {}
    for u in graph:
        residual_graph[u] = {}
        for v, cap in graph[u].items():
            residual_graph[u][v] = cap
    
    # 各辺のフローを記録する辞書
    flow_dict = {}
    
    def find_path(u: int, flow: int) -> int:
        """DFSで増加路を見つける"""
        if u == sink:
            return flow
            
        for v in list(residual_graph.get(u, {})):
            capacity = residual_graph[u][v]
            if capacity > 0 and v not in visited:
                visited.add(v)
                min_flow = find_path(v, min(flow, capacity))
                
                if min_flow > 0:
                    residual_graph[u][v] -= min_flow
                    
                    if v not in residual_graph:
                        residual_graph[v] = {}
                    if u not in residual_graph[v]:
                        residual_graph[v][u] = 0
                    residual_graph[v][u] += min_flow
                    
                    # フローを記録
                    edge = (u, v)
                    reverse_edge = (v, u)
                    
                    if edge in flow_dict:
                        flow_dict[edge] += min_flow
                    else:
                        flow_dict[edge] = min_flow
                        
                    # 逆向きの辺の場合、元のフローから削除
                    if reverse_edge in flow_dict:
                        flow_dict[reverse_edge] -= min_flow
                        if flow_dict[reverse_edge] <= 0:
                            del flow_dict[reverse_edge]
                    
                    return min_flow
        
        return 0
    
    # 最大流量
    max_flow = 0
    
    # 増加路が見つかる限り繰り返す
    while True:
        visited = {source}
        path_flow = find_path(source, float('inf'))
        
        if path_flow == 0:
            break
        
        max_flow += path_flow
    
    # 逆向きのフローを除去して最終的なフローのみを返す
    final_flow = {edge: flow for edge, flow in flow_dict.items() if flow > 0}
    
    return max_flow, final_flow


def bipartite_matching(graph: Dict[int, list], left_nodes: Set[int], right_nodes: Set[int]) -> Dict[int, int]:
    """
    二部グラフの最大マッチングを求める
    
    Args:
        graph: 隣接リスト表現のグラフ
        left_nodes: 左側のノード集合
        right_nodes: 右側のノード集合
        
    Returns:
        マッチング（左側ノード→右側ノード）のディクショナリ
    """
    # フォード・ファルカーソン用のフローネットワークを構築
    flow_graph = {}
    
    # ソースとシンクを追加
    source = max(left_nodes.union(right_nodes)) + 1
    sink = source + 1
    
    # ソースから左側ノードへのエッジ
    flow_graph[source] = {}
    for left in left_nodes:
        flow_graph[source][left] = 1
    
    # 右側ノードからシンクへのエッジ
    for right in right_nodes:
        if right not in flow_graph:
            flow_graph[right] = {}
        flow_graph[right][sink] = 1
    
    # 左側から右側へのエッジ
    for left in left_nodes:
        if left not in flow_graph:
            flow_graph[left] = {}
        for right in graph.get(left, []):
            flow_graph[left][right] = 1
    
    # フォード・ファルカーソンで最大流を求める
    max_flow, flow_dict = ford_fulkerson_with_paths(flow_graph, source, sink)
    
    # マッチング結果の抽出
    matching = {}
    for (u, v), flow in flow_dict.items():
        if u in left_nodes and v in right_nodes and flow > 0:
            matching[u] = v
            
    return matching


# 使用例
def example():
    # グラフの隣接リスト表現（容量つきグラフ）
    # graph[u][v] = c は、uからvへのc単位の容量がある辺を表す
    graph = {
        0: {1: 16, 2: 13},
        1: {2: 10, 3: 12},
        2: {1: 4, 4: 14},
        3: {2: 9, 5: 20},
        4: {3: 7, 5: 4}
    }
    
    source = 0  # 始点
    sink = 5    # 終点
    
    # 最大流を計算
    max_flow = ford_fulkerson(graph, source, sink)
    print(f"最大流量: {max_flow}")
    
    # 流れの詳細を含む最大流を計算
    graph_copy = {
        0: {1: 16, 2: 13},
        1: {2: 10, 3: 12},
        2: {1: 4, 4: 14},
        3: {2: 9, 5: 20},
        4: {3: 7, 5: 4}
    }
    max_flow, flow_dict = ford_fulkerson_with_paths(graph_copy, source, sink)
    print(f"最大流量: {max_flow}")
    print("実際の流れ:")
    for (u, v), flow in sorted(flow_dict.items()):
        print(f"辺 {u} -> {v}: {flow}")
    
    # 二部グラフマッチングの例
    bipartite_graph = {
        1: [4, 5],
        2: [4, 6],
        3: [5, 6]
    }
    left_nodes = {1, 2, 3}
    right_nodes = {4, 5, 6}
    
    matching = bipartite_matching(bipartite_graph, left_nodes, right_nodes)
    print("\n二部グラフの最大マッチング:")
    for left, right in matching.items():
        print(f"{left} -> {right}")
    print(f"マッチングサイズ: {len(matching)}")


if __name__ == "__main__":
    example()
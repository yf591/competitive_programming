# 競技プログラミング アルゴリズム・データ構造ドキュメント

ここでは`utils/helper_functions.py`と`utils/template.py`の実装内容を参照して、競技プログラミングでよく使用されるアルゴリズムとデータ構造の概要、実装のポイント、計算量などをまとめました。

## 目次

1. [グラフアルゴリズム](#グラフアルゴリズム)
2. [文字列アルゴリズム](#文字列アルゴリズム)
3. [数学的アルゴリズム](#数学的アルゴリズム)
4. [データ構造](#データ構造)
5. [動的計画法（DP）](#動的計画法dp)
6. [貪欲法](#貪欲法)
7. [二分探索](#二分探索)
8. [計算幾何学](#計算幾何学)
9. [フローアルゴリズム](#フローアルゴリズム)
10. [入出力と最適化](#入出力と最適化)
11. [その他のテクニック](#その他のテクニック)
12. [競技プログラミングでの注意点](#競技プログラミングでの注意点)

## グラフアルゴリズム

### 深さ優先探索（DFS）

- **概要**: グラフの各頂点を可能な限り深く探索していく手法
- **用途**: 連結成分の検出、トポロジカルソート、サイクル検出など
- **時間計算量**: O(V + E) (V: 頂点数, E: 辺数)

```python
def dfs(graph, start, visited=None):
    if visited is None:
        visited = set()
    visited.add(start)
    for neighbor in graph[start]:
        if neighbor not in visited:
            dfs(graph, neighbor, visited)
    return visited
```

### 幅優先探索（BFS）

- **概要**: グラフの各頂点を階層的に探索していく手法
- **用途**: 最短経路問題（重みなしグラフ）、連結成分の検出など
- **時間計算量**: O(V + E) (V: 頂点数, E: 辺数)

```python
from collections import deque

def bfs(graph, start):
    visited = set([start])
    queue = deque([start])
    while queue:
        vertex = queue.popleft()
        for neighbor in graph[vertex]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return visited
```

### ダイクストラのアルゴリズム

- **概要**: 始点から各頂点への最短経路を求めるアルゴリズム
- **用途**: 重み付きグラフでの最短経路問題
- **時間計算量**: O((V + E) log V) (優先度付きキュー使用時)
- **注意**: 負の辺がある場合は使用できない

```python
import heapq

def dijkstra(graph, start):
    distances = {vertex: float('infinity') for vertex in graph}
    distances[start] = 0
    priority_queue = [(0, start)]
    
    while priority_queue:
        current_distance, current_vertex = heapq.heappop(priority_queue)
        
        # すでに見つかった経路の方が短い場合はスキップ
        if current_distance > distances[current_vertex]:
            continue
            
        for neighbor, weight in graph.get(current_vertex, []):
            distance = current_distance + weight
            
            if distance < distances.get(neighbor, float('infinity')):
                distances[neighbor] = distance
                heapq.heappush(priority_queue, (distance, neighbor))
                
    return distances
```

### ベルマン-フォード法

- **概要**: 始点から各頂点への最短経路を求めるアルゴリズム
- **用途**: 負の辺を含むグラフでの最短経路問題、負のサイクル検出
- **時間計算量**: O(V * E)

```python
def bellman_ford(graph, start):
    distances = {node: float('infinity') for node in graph}
    distances[start] = 0
    
    for _ in range(len(graph) - 1):
        for u in graph:
            for v, w in graph[u].items():
                if distances[u] + w < distances[v]:
                    distances[v] = distances[u] + w
    
    # 負の閉路の検出
    for u in graph:
        for v, w in graph[u].items():
            if distances[u] + w < distances[v]:
                return None  # 負の閉路が存在
    
    return distances
```

### クラスカルのアルゴリズム（最小全域木）

- **概要**: 最小全域木を求めるアルゴリズム
- **用途**: ネットワーク設計、クラスタリングなど
- **時間計算量**: O(E log E) または O(E log V)

```python
def kruskal(graph, vertices):
    edges = [(w, u, v) for u in graph for v, w in graph[u].items()]
    edges.sort()  # 辺の重みでソート
    
    # Union-Find データ構造
    uf = UnionFind(vertices)
    
    mst = []
    for w, u, v in edges:
        if not uf.is_same(u, v):
            uf.union(u, v)
            mst.append((u, v, w))
    
    return mst
```

### トポロジカルソート

- **概要**: 有向非巡回グラフ（DAG）において、頂点を辺の方向に沿って順序付けるアルゴリズム
- **用途**: タスクスケジューリング、依存関係の解決など
- **時間計算量**: O(V + E)

```python
from collections import deque

def topological_sort(graph):
    in_degree = {node: 0 for node in graph}
    for node in graph:
        for neighbor in graph[node]:
            in_degree[neighbor] = in_degree.get(neighbor, 0) + 1
    
    queue = deque([node for node in graph if in_degree[node] == 0])
    result = []
    
    while queue:
        node = queue.popleft()
        result.append(node)
        
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    
    if len(result) != len(graph):
        return None  # グラフにサイクルが存在する
    
    return result
```

## 文字列アルゴリズム

### KMP（Knuth-Morris-Pratt）アルゴリズム

- **概要**: 文字列の中からパターンを効率的に検索するアルゴリズム
- **用途**: パターンマッチング
- **時間計算量**: O(n + m) (n: テキスト長, m: パターン長)

```python
def kmp_preprocess(pattern):
    """KMPの失敗関数（部分一致テーブル）を計算"""
    m = len(pattern)
    lps = [0] * m  # 最長の接頭辞かつ接尾辞
    
    for i in range(1, m):
        j = lps[i - 1]
        while j > 0 and pattern[i] != pattern[j]:
            j = lps[j - 1]
        if pattern[i] == pattern[j]:
            j += 1
        lps[i] = j
        
    return lps

def kmp_search(text, pattern):
    """KMPアルゴリズムでパターンを検索"""
    if not pattern:
        return list(range(len(text) + 1))
    
    if not text:
        return []
    
    n, m = len(text), len(pattern)
    lps = kmp_preprocess(pattern)
    occurrences = []
    
    j = 0  # パターンのインデックス
    for i in range(n):  # テキストのインデックス
        while j > 0 and text[i] != pattern[j]:
            j = lps[j - 1]
            
        if text[i] == pattern[j]:
            j += 1
            
        if j == m:
            occurrences.append(i - m + 1)
            j = lps[j - 1]
            
    return occurrences
```

### Z アルゴリズム

- **概要**: 文字列の各位置で、元の文字列との共通接頭辞の長さを計算するアルゴリズム
- **用途**: パターンマッチング、文字列の周期性の解析
- **時間計算量**: O(n)

```python
def z_function(s):
    n = len(s)
    z = [0] * n
    z[0] = n
    
    l, r = 0, 0  # [l, r]は最右の共通接頭辞
    for i in range(1, n):
        if i <= r:
            z[i] = min(r - i + 1, z[i - l])
        
        # 共通接頭辞を拡張
        while i + z[i] < n and s[z[i]] == s[i + z[i]]:
            z[i] += 1
        
        # 最右の共通接頭辞を更新
        if i + z[i] - 1 > r:
            l, r = i, i + z[i] - 1
            
    return z
```

### ラビン-カープ アルゴリズム

- **概要**: ハッシュを使用してパターンマッチングを行うアルゴリズム
- **用途**: 複数パターンのマッチング
- **時間計算量**: 平均 O(n + m)、最悪 O(n * m)

```python
def rabin_karp(text, pattern, q=101):
    n, m = len(text), len(pattern)
    if m == 0 or m > n:
        return []
    
    d = 256  # 文字セットのサイズ
    h = pow(d, m - 1, q)
    p, t = 0, 0
    
    # パターンと最初のウィンドウのハッシュを計算
    for i in range(m):
        p = (d * p + ord(pattern[i])) % q
        t = (d * t + ord(text[i])) % q
    
    result = []
    
    for i in range(n - m + 1):
        if p == t:
            # ハッシュが一致した場合、文字ごとに確認
            if text[i:i+m] == pattern:
                result.append(i)
        
        if i < n - m:
            t = (d * (t - ord(text[i]) * h) + ord(text[i + m])) % q
            if t < 0:
                t += q
    
    return result
```

### マナカーのアルゴリズム

- **概要**: 文字列内のすべての回文を線形時間で見つけるアルゴリズム
- **用途**: 最長回文部分文字列の検索
- **時間計算量**: O(n)

```python
def manacher(s):
    # 文字間に#を挿入して奇数長の文字列に変換
    t = '#' + '#'.join(s) + '#'
    n = len(t)
    p = [0] * n  # p[i]はt[i]を中心とする回文の半径
    
    c = 0  # 現在の中心
    r = 0  # 現在の回文の右端
    
    for i in range(n):
        if r > i:
            p[i] = min(r - i, p[2 * c - i])
        
        # 中心iの周りを拡張
        while i - p[i] - 1 >= 0 and i + p[i] + 1 < n and t[i - p[i] - 1] == t[i + p[i] + 1]:
            p[i] += 1
        
        # 必要に応じて中心と右端を更新
        if i + p[i] > r:
            c, r = i, i + p[i]
    
    # 元の文字列における最長回文部分文字列の長さと開始位置を求める
    max_len = max(p)
    center = p.index(max_len)
    
    # 元の文字列における開始位置を計算
    start = (center - max_len) // 2
    
    return s[start:start + max_len]
```

## 数学的アルゴリズム

### 素数判定と素因数分解

- **概要**: 素数の判定と、数を素因数に分解するアルゴリズム
- **時間計算量**
  - 素数判定: O(√n)
  - 素因数分解: O(√n)

```python
def is_prime(n):
    """Trial divisionによる素数判定"""
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def prime_factors(n):
    """素因数分解（結果はリスト）"""
    factors = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors
```

### エラトステネスの篩

- **概要**: 特定の範囲内のすべての素数を効率的に列挙するアルゴリズム
- **時間計算量**: O(n log log n)

```python
def sieve_of_eratosthenes(n):
    """n以下の素数をすべて返す"""
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    
    for i in range(2, int(math.sqrt(n)) + 1):
        if sieve[i]:
            for j in range(i * i, n + 1, i):
                sieve[j] = False
                
    return [i for i in range(n + 1) if sieve[i]]
```

### 拡張ユークリッドの互除法

- **概要**: ax + by = gcd(a, b) を満たす整数 x, y を求めるアルゴリズム
- **用途**: モジュラ逆元の計算、中国剰余定理など
- **時間計算量**: O(log(min(a, b)))

```python
def extended_gcd(a, b):
    """拡張ユークリッドの互除法"""
    if a == 0:
        return (b, 0, 1)
    
    gcd, x1, y1 = extended_gcd(b % a, a)
    x = y1 - (b // a) * x1
    y = x1
    
    return (gcd, x, y)
```

### 高速べき乗法（繰り返し二乗法）

- **概要**: 大きな指数の累乗を効率的に計算するアルゴリズム
- **時間計算量**: O(log n)

```python
def modular_exponentiation(base, exponent, modulus):
    """高速べき乗法（繰り返し二乗法）"""
    if modulus == 1:
        return 0
    
    result = 1
    base = base % modulus
    
    while exponent > 0:
        if exponent % 2 == 1:
            result = (result * base) % modulus
        exponent = exponent >> 1
        base = (base * base) % modulus
    
    return result
```

### モジュラ逆元

- **概要**: a * x ≡ 1 (mod m) となる x を求めるアルゴリズム
- **時間計算量**: O(log m) (拡張ユークリッド法を使用)

```python
def mod_inverse(a, m):
    """aのmod mにおける逆元を返す"""
    g, x, y = extended_gcd(a, m)
    if g != 1:
        raise ValueError(f"Modular inverse doesn't exist for {a} mod {m}")
    else:
        return x % m
```

### 最大公約数と最小公倍数

- **概要**: 2つの整数の最大公約数と最小公倍数を計算するアルゴリズム
- **時間計算量**: O(log(min(a, b)))

```python
def gcd(a, b):
    """ユークリッドの互除法による最大公約数"""
    while b:
        a, b = b, a % b
    return a

def lcm(a, b):
    """最小公倍数"""
    return a * b // gcd(a, b)
```

### 組み合わせ計算

- **概要**: 組み合わせ (nCr) や順列 (nPr) の計算を効率的に行うクラス
- **時間計算量**: 前計算 O(n)、クエリ O(1)

```python
class Combinatorics:
    """組み合わせ計算のためのクラス（ファクトリアルの前計算）"""
    
    def __init__(self, n, mod=10**9 + 7):
        """
        n以下の値に対する階乗を前計算
        """
        self.mod = mod
        self.fact = [1] * (n + 1)  # factorial
        self.inv_fact = [1] * (n + 1)  # inverse factorial
        
        # 階乗と逆階乗の前計算
        for i in range(1, n + 1):
            self.fact[i] = (self.fact[i - 1] * i) % mod
            
        # フェルマーの小定理を使用
        self.inv_fact[n] = pow(self.fact[n], mod - 2, mod)
        
        # 他の逆階乗を計算
        for i in range(n - 1, 0, -1):
            self.inv_fact[i] = (self.inv_fact[i + 1] * (i + 1)) % mod
    
    def combination(self, n, r):
        """組み合わせ C(n,r) = n! / (r! * (n-r)!)"""
        if r < 0 or r > n:
            return 0
        return (self.fact[n] * self.inv_fact[r] % self.mod * self.inv_fact[n - r] % self.mod)
    
    def permutation(self, n, r):
        """順列 P(n,r) = n! / (n-r)!"""
        if r < 0 or r > n:
            return 0
        return (self.fact[n] * self.inv_fact[n - r] % self.mod)
    
    def catalan(self, n):
        """カタラン数 Cat(n) = C(2n,n) / (n+1)"""
        return (self.combination(2 * n, n) * pow(n + 1, -1, self.mod)) % self.mod
```

## データ構造

### Union-Find（Disjoint Set）

- **概要**: 素集合を効率的に扱うデータ構造
- **用途**: 連結成分の管理、クラスカルのアルゴリズムなど
- **時間計算量**
  - 結合操作: O(α(n)) ≈ O(1) (アッカーマン関数の逆関数)
  - 検索操作: O(α(n)) ≈ O(1)

```python
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.size = [1] * n
        self.count = n  # 連結成分の数
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # 経路圧縮
        return self.parent[x]
    
    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x == root_y:
            return False
        
        # ランクによる結合
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
            self.size[root_y] += self.size[root_x]
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
            self.size[root_x] += self.size[root_y]
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1
            self.size[root_x] += self.size[root_y]
        
        self.count -= 1  # 連結成分数を減らす
        return True
    
    def get_size(self, x):
        return self.size[self.find(x)]
    
    def is_same(self, x, y):
        return self.find(x) == self.find(y)
```

### セグメント木

- **概要**: 区間クエリと点更新を効率的に行うデータ構造
- **用途**: 区間の和、最小値、最大値などのクエリ
- **時間計算量**
  - 構築: O(n)
  - クエリ・更新: O(log n)

```python
class SegmentTree:
    def __init__(self, arr, operation='sum'):
        """
        セグメント木の初期化
        
        Args:
            arr: 元の配列
            operation: クエリ操作の種類 ('sum', 'min', 'max')
        """
        self.n = len(arr)
        self.operation = operation
        
        # 操作関数の定義
        if operation == 'sum':
            self.op = lambda x, y: x + y
            self.identity = 0
        elif operation == 'min':
            self.op = lambda x, y: min(x, y)
            self.identity = float('inf')
        elif operation == 'max':
            self.op = lambda x, y: max(x, y)
            self.identity = float('-inf')
        else:
            raise ValueError("Unsupported operation")
        
        # 木の構築
        self.tree = [self.identity] * (4 * self.n)
        self._build(arr, 1, 0, self.n - 1)
    
    def _build(self, arr, node, start, end):
        if start == end:
            self.tree[node] = arr[start]
            return
        
        mid = (start + end) // 2
        self._build(arr, 2 * node, start, mid)
        self._build(arr, 2 * node + 1, mid + 1, end)
        self.tree[node] = self.op(self.tree[2 * node], self.tree[2 * node + 1])
    
    def update(self, idx, val):
        self._update(1, 0, self.n - 1, idx, val)
    
    def _update(self, node, start, end, idx, val):
        if start == end:
            self.tree[node] = val
            return
        
        mid = (start + end) // 2
        if idx <= mid:
            self._update(2 * node, start, mid, idx, val)
        else:
            self._update(2 * node + 1, mid + 1, end, idx, val)
        
        self.tree[node] = self.op(self.tree[2 * node], self.tree[2 * node + 1])
    
    def query(self, left, right):
        return self._query(1, 0, self.n - 1, left, right)
    
    def _query(self, node, start, end, left, right):
        # 範囲外
        if start > right or end < left:
            return self.identity
        
        # 完全に含まれる
        if left <= start and end <= right:
            return self.tree[node]
        
        # 部分的に含まれる
        mid = (start + end) // 2
        left_result = self._query(2 * node, start, mid, left, right)
        right_result = self._query(2 * node + 1, mid + 1, end, left, right)
        
        return self.op(left_result, right_result)
```

### 遅延伝播セグメント木

- **概要**: 区間更新と区間クエリを効率的に行うデータ構造
- **用途**: 区間和の計算、区間更新
- **時間計算量**
  - 構築: O(n)
  - クエリ・更新: O(log n)

```python
class LazySegmentTree:
    def __init__(self, arr, operation='sum'):
        """遅延伝播セグメント木の初期化"""
        self.n = len(arr)
        self.operation = operation
        
        # 操作関数の定義
        if operation == 'sum':
            self.op = lambda x, y: x + y
            self.identity = 0
        elif operation == 'min':
            self.op = lambda x, y: min(x, y)
            self.identity = float('inf')
        elif operation == 'max':
            self.op = lambda x, y: max(x, y)
            self.identity = float('-inf')
        else:
            raise ValueError("Unsupported operation")
        
        # 木の構築
        self.tree = [self.identity] * (4 * self.n)
        self.lazy = [0] * (4 * self.n)  # 遅延更新
        self._build(arr, 1, 0, self.n - 1)
    
    def _build(self, arr, node, start, end):
        """セグメント木の構築"""
        if start == end:
            self.tree[node] = arr[start]
            return
        
        mid = (start + end) // 2
        self._build(arr, 2 * node, start, mid)
        self._build(arr, 2 * node + 1, mid + 1, end)
        self.tree[node] = self.op(self.tree[2 * node], self.tree[2 * node + 1])
    
    def _propagate(self, node, start, end):
        """遅延値の伝播"""
        if self.lazy[node] != 0:
            self.tree[node] += (end - start + 1) * self.lazy[node]  # sum用
            
            if start != end:  # 葉ノードでない場合、子へ伝播
                self.lazy[2 * node] += self.lazy[node]
                self.lazy[2 * node + 1] += self.lazy[node]
                
            self.lazy[node] = 0  # 遅延値をリセット
    
    def update_range(self, left, right, val):
        """区間[left, right]にvalを加算"""
        self._update_range(1, 0, self.n - 1, left, right, val)
    
    def _update_range(self, node, start, end, left, right, val):
        """区間更新の内部実装"""
        self._propagate(node, start, end)
        
        # 範囲外
        if start > right or end < left:
            return
        
        # 完全に含まれる
        if left <= start and end <= right:
            self.tree[node] += (end - start + 1) * val  # sum用
            
            if start != end:  # 葉ノードでない場合、遅延更新をマーク
                self.lazy[2 * node] += val
                self.lazy[2 * node + 1] += val
                
            return
        
        # 部分的に含まれる
        mid = (start + end) // 2
        self._update_range(2 * node, start, mid, left, right, val)
        self._update_range(2 * node + 1, mid + 1, end, left, right, val)
        
        self.tree[node] = self.op(self.tree[2 * node], self.tree[2 * node + 1])
    
    def query_range(self, left, right):
        """区間[left, right]に対するクエリ"""
        return self._query_range(1, 0, self.n - 1, left, right)
    
    def _query_range(self, node, start, end, left, right):
        """区間クエリの内部実装"""
        # 範囲外
        if start > right or end < left:
            return self.identity
        
        self._propagate(node, start, end)
        
        # 完全に含まれる
        if left <= start and end <= right:
            return self.tree[node]
        
        # 部分的に含まれる
        mid = (start + end) // 2
        left_result = self._query_range(2 * node, start, mid, left, right)
        right_result = self._query_range(2 * node + 1, mid + 1, end, left, right)
        
        return self.op(left_result, right_result)
```

### フェニック木（Binary Indexed Tree, BIT）

- **概要**: 累積和を効率的に管理するデータ構造
- **用途**: 累積和の計算、点更新
- **時間計算量**
  - 更新: O(log n)
  - クエリ: O(log n)

```python
class FenwickTree:
    def __init__(self, n):
        self.size = n
        self.tree = [0] * (n + 1)  # 1-indexed
    
    def update(self, idx, delta):
        """位置idxの要素にdeltaを加算"""
        idx += 1  # 1-indexedに変換
        while idx <= self.size:
            self.tree[idx] += delta
            idx += idx & -idx  # 最下位ビットを加算
    
    def prefix_sum(self, idx):
        """位置0からidxまでの和を計算"""
        idx += 1  # 1-indexedに変換
        result = 0
        while idx > 0:
            result += self.tree[idx]
            idx -= idx & -idx  # 最下位ビットを減算
        return result
    
    def range_sum(self, left, right):
        """範囲[left, right]の和を計算"""
        return self.prefix_sum(right) - (self.prefix_sum(left - 1) if left > 0 else 0)
```

### 2次元フェニック木

- **概要**: 2次元平面上での累積和を効率的に管理するデータ構造
- **用途**: 2次元累積和の計算、点更新
- **時間計算量**
  - 更新: O(log n * log m)
  - クエリ: O(log n * log m)

```python
class FenwickTree2D:
    def __init__(self, rows, cols):
        """2次元フェニック木の初期化"""
        self.rows = rows
        self.cols = cols
        self.tree = [[0] * (cols + 1) for _ in range(rows + 1)]  # 1-indexed
    
    def update(self, row, col, delta):
        """位置(row, col)の要素にdeltaを加算"""
        row += 1  # 1-indexedに変換
        col += 1
        
        i = row
        while i <= self.rows:
            j = col
            while j <= self.cols:
                self.tree[i][j] += delta
                j += j & -j  # jの最下位ビットを加算
            i += i & -i  # iの最下位ビットを加算
    
    def prefix_sum(self, row, col):
        """左上(0,0)から(row,col)までの長方形の和"""
        row += 1  # 1-indexedに変換
        col += 1
        result = 0
        
        i = row
        while i > 0:
            j = col
            while j > 0:
                result += self.tree[i][j]
                j -= j & -j  # jの最下位ビットを減算
            i -= i & -i  # iの最下位ビットを減算
            
        return result
    
    def range_sum(self, row1, col1, row2, col2):
        """長方形領域[(row1,col1), (row2,col2)]の和"""
        # 包除原理を使用
        result = self.prefix_sum(row2, col2)
        
        if row1 > 0:
            result -= self.prefix_sum(row1 - 1, col2)
        if col1 > 0:
            result -= self.prefix_sum(row2, col1 - 1)
        if row1 > 0 and col1 > 0:
            result += self.prefix_sum(row1 - 1, col1 - 1)
            
        return result
```

### トライ木

- **概要**: 文字列の集合を効率的に格納・検索するデータ構造
- **用途**: 辞書の実装、自動補完機能など
- **時間計算量**
  - 挿入・検索・削除: O(m) (m: 文字列の長さ)

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False

class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word):
        """単語をトライ木に挿入"""
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True
    
    def search(self, word):
        """単語が存在するかを確認"""
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end_of_word
    
    def starts_with(self, prefix):
        """指定したプレフィックスから始まる単語が存在するか確認"""
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True
```

## 動的計画法（DP）

### 最長増加部分列（LIS）

- **概要**: 配列の最長増加部分列の長さを求めるアルゴリズム
- **時間計算量**: O(n²) または O(n log n) (二分探索使用時)

```python
def longest_increasing_subsequence(arr):
    """最長増加部分列の長さを求める"""
    if not arr:
        return 0
        
    n = len(arr)
    dp = [1] * n
    
    for i in range(1, n):
        for j in range(i):
            if arr[i] > arr[j]:
                dp[i] = max(dp[i], dp[j] + 1)
                
    return max(dp)

def longest_increasing_subsequence_fast(arr):
    """二分探索を使用した高速な実装"""
    if not arr:
        return 0
    
    tails = []
    
    for x in arr:
        idx = bisect.bisect_left(tails, x)
        if idx == len(tails):
            tails.append(x)
        else:
            tails[idx] = x
    
    return len(tails)
```

### 最長共通部分列（LCS）

- **概要**: 2つの配列の最長共通部分列の長さを求めるアルゴリズム
- **時間計算量**: O(nm) (n, m: 配列の長さ)

```python
def longest_common_subsequence(s1, s2):
    """最長共通部分列の長さを求める"""
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
                
    return dp[m][n]
```

### ナップサック問題

- **概要**: 容量制限のあるナップサックに価値の最大化を目指してアイテムを詰めるアルゴリズム
- **時間計算量**: O(nW) (n: アイテム数, W: ナップサックの容量)

```python
def knapsack(weights, values, capacity):
    """0/1ナップサック問題を解く"""
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            if weights[i - 1] <= w:
                dp[i][w] = max(values[i - 1] + dp[i - 1][w - weights[i - 1]], dp[i - 1][w])
            else:
                dp[i][w] = dp[i - 1][w]
                
    return dp[n][capacity]
```

### 編集距離

- **概要**: 2つの文字列間の編集距離（挿入、削除、置換の最小回数）を計算するアルゴリズム
- **時間計算量**: O(nm) (n, m: 文字列の長さ)

```python
def edit_distance(s1, s2):
    """編集距離（レーベンシュタイン距離）を計算"""
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # 基底ケース: 空文字列への変換
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],      # 削除
                    dp[i][j - 1],      # 挿入
                    dp[i - 1][j - 1]   # 置換
                )
                
    return dp[m][n]
```

## 貪欲法

### 区間スケジューリング

- **概要**: 重複しないように最大数の区間を選択するアルゴリズム
- **時間計算量**: O(n log n) (ソートに依存)

```python
def interval_scheduling(intervals):
    """区間スケジューリング問題を解く"""
    # 終了時間でソート
    intervals.sort(key=lambda x: x[1])
    
    result = []
    end_time = float('-inf')
    
    for start, end in intervals:
        if start >= end_time:
            result.append((start, end))
            end_time = end
    
    return result
```

### コイン問題

- **概要**: 金額を指定された硬貨で支払う際の最小硬貨数を求めるアルゴリズム
- **時間計算量**: O(n) (n: コインの種類)
- **注意**: 貪欲法は常に最適解を与えるとは限らない

```python
def coin_change_greedy(coins, amount):
    """
    貪欲法によるコイン問題
    注意: コインの種類によっては最適解を得られない場合あり
    """
    # コインを降順にソート
    coins.sort(reverse=True)
    
    count = 0
    for coin in coins:
        # 現在のコインを可能な限り使用
        count += amount // coin
        amount %= coin
    
    return count if amount == 0 else -1
```

## 二分探索

### 基本的な二分探索

- **概要**: ソート済み配列から要素を効率的に検索するアルゴリズム
- **時間計算量**: O(log n)

```python
def binary_search(arr, target):
    """
    二分探索でtargetの位置を探す
    
    Returns:
        見つかった場合はインデックス、見つからない場合は-1
    """
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
            
    return -1
```

### 二分探索の応用（最左・最右の要素）

- **概要**: 条件を満たす最初または最後の要素を見つける二分探索の変種
- **時間計算量**: O(log n)

```python
def binary_search_leftmost(arr, target):
    """
    targetの最左（最初）の出現位置を返す
    存在しない場合は挿入すべき位置を返す
    """
    return bisect.bisect_left(arr, target)

def binary_search_rightmost(arr, target):
    """
    targetの最右（最後）の出現位置の次の位置を返す
    存在しない場合は挿入すべき位置を返す
    """
    return bisect.bisect_right(arr, target)
```

### 判定問題に対する二分探索

- **概要**: 解の範囲で二分探索を行い、判定関数によって結果を絞り込むアルゴリズム
- **時間計算量**: O(log n * 判定関数の計算量)

```python
def binary_search_predicate(low, high, predicate, epsilon=1e-9):
    """
    条件を満たす値を二分探索で求める（浮動小数点用）
    
    Args:
        low: 探索範囲の下限
        high: 探索範囲の上限
        predicate: 条件関数（条件を満たす場合にTrueを返す）
        epsilon: 許容誤差
    """
    while high - low > epsilon:
        mid = (low + high) / 2
        if predicate(mid):
            high = mid
        else:
            low = mid
    return high

def discrete_binary_search(predicate, low, high):
    """
    条件を満たす最小の整数xを二分探索で求める
    
    Args:
        predicate: 条件関数（条件を満たす場合にTrueを返す）
        low: 探索範囲の下限（含む）
        high: 探索範囲の上限（含む）
    """
    while low < high:
        mid = (low + high) // 2
        if predicate(mid):
            high = mid
        else:
            low = mid + 1
    
    return low if predicate(low) else high + 1
```

## 計算幾何学

### 点と線分

- **概要**: 点と線分の関係を扱うアルゴリズム
- **時間計算量**: O(1)

```python
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __eq__(self, other):
        if not isinstance(other, Point):
            return False
        return abs(self.x - other.x) < 1e-9 and abs(self.y - other.y) < 1e-9
    
    def __hash__(self):
        return hash((self.x, self.y))
    
    def distance(self, other):
        """他の点までのユークリッド距離"""
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

def cross_product(p1, p2, p3):
    """
    外積 (p2-p1)×(p3-p1) を計算
    
    Returns:
        正の値: p3はp1→p2の左側
        0: p1, p2, p3は同一線上
        負の値: p3はp1→p2の右側
    """
    return (p2.x - p1.x) * (p3.y - p1.y) - (p2.y - p1.y) * (p3.x - p1.x)
```

### 凸包

- **概要**: 点の集合を囲む最小の凸多角形（凸包）を求めるアルゴリズム
- **時間計算量**: O(n log n) (ソートに依存)

```python
def convex_hull(points):
    """Graham's scanアルゴリズムで凸包を計算"""
    if len(points) <= 2:
        return points.copy()
    
    # y座標が最小（同じ場合はx座標が最小）の点を見つける
    p0 = min(points, key=lambda p: (p.y, p.x))
    
    # p0を基準に偏角でソート
    def polar_angle(p):
        if p == p0:
            return float('-inf')
        return math.atan2(p.y - p0.y, p.x - p0.x)
    
    sorted_points = sorted(points, key=polar_angle)
    
    # 凸包の構築
    hull = [sorted_points[0], sorted_points[1]]
    
    for i in range(2, len(sorted_points)):
        while len(hull) > 1 and cross_product(hull[-2], hull[-1], sorted_points[i]) <= 0:
            hull.pop()
        hull.append(sorted_points[i])
    
    return hull
```

## フローアルゴリズム

### フォード・ファルカーソンのアルゴリズム

- **概要**: 最大流問題を解くアルゴリズム
- **時間計算量**: O(F * E) (F: 最大流量, E: 辺の数)

```python
def ford_fulkerson(graph, source, sink):
    """
    最大流問題を解くフォード・ファルカーソンのアルゴリズム
    
    Args:
        graph: グラフの隣接リスト表現（graph[u][v]はuからvへの容量）
        source: 始点
        sink: 終点
    """
    def dfs(u, flow):
        if u == sink:
            return flow
            
        for v in graph.get(u, {}):
            capacity = graph[u][v]
            if capacity > 0 and v not in visited:
                visited.add(v)
                min_flow = dfs(v, min(flow, capacity))
                if min_flow > 0:
                    graph[u][v] -= min_flow
                    if v not in graph or u not in graph[v]:
                        if v not in graph:
                            graph[v] = {}
                        graph[v][u] = min_flow
                    else:
                        graph[v][u] += min_flow
                    return min_flow
        return 0
    
    max_flow = 0
    while True:
        visited = {source}
        path_flow = dfs(source, float('inf'))
        if path_flow == 0:
            break
        max_flow += path_flow
    
    return max_flow
```

## 入出力と最適化

### 高速入出力

- **概要**: 競技プログラミングで入出力を効率化するテクニック
- **用途**: 大量データの読み込みや出力の高速化

```python
# 標準入力の読み込みを高速化
input = sys.stdin.readline

def read_int():
    """一つの整数を読み込む"""
    return int(input())

def read_ints():
    """スペース区切りの整数列を読み込む"""
    return list(map(int, input().split()))

def read_str():
    """一つの文字列を読み込む"""
    return input().strip()

def read_strs():
    """スペース区切りの文字列列を読み込む"""
    return input().split()

def read_int_grid(h, w):
    """整数のグリッドを読み込む"""
    return [list(map(int, input().split())) for _ in range(h)]

def read_str_grid(h):
    """文字列のグリッドを読み込む"""
    return [input().strip() for _ in range(h)]
```

### コードの最適化

- **概要**: Pythonで実行速度を向上させるテクニック
- **用途**: 制限時間内に処理を完了させる

```python
import sys
sys.setrecursionlimit(10**6)  # 再帰制限を緩和

# 定数を事前定義
MOD = 10**9 + 7
INF = float('inf')

# 関数をキャッシュ（DP等に有効）
@lru_cache(maxsize=None)
def fib(n):
    if n <= 1:
        return n
    return fib(n-1) + fib(n-2)
```

## その他のテクニック

### 座標圧縮

- **概要**: 大きな範囲の値を0からの連続した整数に変換するテクニック
- **用途**: 値の範囲が大きい場合の効率化
- **時間計算量**: O(n log n) (ソートに依存)

```python
def coordinate_compression(arr):
    """
    座標圧縮を行い、圧縮後の配列と元の値へのマッピングを返す
    """
    sorted_arr = sorted(set(arr))
    compress_map = {val: i for i, val in enumerate(sorted_arr)}
    compressed = [compress_map[x] for x in arr]
    return compressed, {i: val for i, val in enumerate(sorted_arr)}
```

### 素因数分解の高速化

- **概要**: 事前計算した最小素因数を使用して素因数分解を高速化するテクニック
- **用途**: 複数の数の素因数分解が必要な場合
- **時間計算量**: 前計算 O(n log log n)、クエリ O(log n)

```python
def precompute_spf(n):
    """n以下の各数の最小素因数を前計算"""
    spf = list(range(n + 1))
    for i in range(4, n + 1, 2):
        spf[i] = 2
    
    for i in range(3, int(math.sqrt(n)) + 1):
        if spf[i] == i:  # iは素数
            for j in range(i * i, n + 1, i):
                if spf[j] == j:  # まだ設定されていない
                    spf[j] = i
    
    return spf

def fast_prime_factors(n, precomputed_spf):
    """前計算した最小素因数を使用して素因数分解"""
    factors = []
    while n > 1:
        factors.append(precomputed_spf[n])
        n //= precomputed_spf[n]
    return factors
```

### 行列演算

- **概要**: 行列の乗算や累乗を効率的に計算するアルゴリズム
- **用途**: 漸化式の高速計算、グラフの経路数計算など
- **時間計算量** 
  - 乗算: O(n³)
  - 累乗: O(n³ log k) (kは指数)

```python
def matrix_multiply(A, B, mod=None):
    """行列の乗算"""
    if not A or not B or not A[0] or not B[0]:
        return [[]]
        
    n, m = len(A), len(A[0])
    p, q = len(B), len(B[0])
    
    if m != p:
        raise ValueError("Matrix dimensions incompatible for multiplication")
    
    result = [[0] * q for _ in range(n)]
    
    for i in range(n):
        for j in range(q):
            for k in range(m):
                result[i][j] += A[i][k] * B[k][j]
                if mod:
                    result[i][j] %= mod
                    
    return result

def matrix_power(A, power, mod=None):
    """行列の累乗（繰り返し二乗法）"""
    if not A or len(A) != len(A[0]):
        raise ValueError("Matrix must be square")
        
    if power < 0:
        raise ValueError("Power must be non-negative")
        
    n = len(A)
    
    # 単位行列
    result = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
    
    # 繰り返し二乗法
    while power > 0:
        if power & 1:  # powerが奇数
            result = matrix_multiply(result, A, mod)
        A = matrix_multiply(A, A, mod)
        power >>= 1
        
    return result
```

## 競技プログラミングでの注意点

### 実装上の注意

1. **インデックスエラーに注意**: 配列の範囲外アクセスは一般的なバグの原因です
2. **整数オーバーフロー**: Pythonでは通常問題ありませんが、他言語では注意が必要
3. **浮動小数点精度**: 浮動小数点数の比較には適切な誤差（epsilon）を設定する
4. **再帰制限**: 深い再帰を使う場合は`sys.setrecursionlimit()`で制限を緩和する

### パフォーマンスの最適化

1. **適切なアルゴリズム選択**: 問題の制約に合わせて効率的なアルゴリズムを選択する
2. **データ構造の選択**: 操作の種類に応じて適切なデータ構造を使用する
3. **ループとリスト内包表記**: リスト内包表記は通常のループより高速
4. **入出力の最適化**: 大量のデータを処理する場合は入出力を最適化する

### テストと検証

1. **エッジケースのテスト**: 境界条件、空の入力、最大値などをテスト
2. **小さな例での手計算**: 小規模な例で結果を手計算して検証
3. **ランダムテストの活用**: ランダムな入力を生成して別の解法と結果を比較
4. **制限の確認**: 時間制限とメモリ制限を考慮してアルゴリズムを選択

### コーディングスタイル

1. **わかりやすい変数名**: 意味のある変数名を使用すると理解しやすく、デバッグも容易になる
2. **コードの整理**: 関連する機能をまとめて関数化し、コードを整理する
3. **一貫性のある命名**: 一貫性のある命名規則を使用する
4. **コメントの追加**: 複雑なロジックにはコメントを追加する

### テンプレートの活用

競技プログラミングでは、よく使用するコードをテンプレートとして準備しておくことが効率的です。`template.py`を活用して、素早く問題に取り組みましょう。

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Competitive Programming Template

This template provides a standard starting point for competitive programming problems.
It includes common imports, input/output handling functions, and optimizations
for faster execution.
"""

import sys
import math
import heapq
import bisect
import collections
from typing import List, Dict, Set, Tuple, Optional, Union, Any
from collections import defaultdict, deque, Counter
from itertools import permutations, combinations, product, accumulate
from functools import lru_cache, reduce

# ----- Constants -----
MOD = 10**9 + 7
INF = float('inf')
NINF = float('-inf')

# ----- Input/Output Optimization -----
input = sys.stdin.readline
sys.setrecursionlimit(10**6)  # Increase recursion limit

def read_int():
    """Read a single integer."""
    return int(input())

def read_ints():
    """Read multiple integers from a single line."""
    return list(map(int, input().split()))

def read_str():
    """Read a single string."""
    return input().strip()

def read_strs():
    """Read multiple strings from a single line."""
    return input().split()

def read_int_grid(h: int, w: int) -> List[List[int]]:
    """Read a 2D grid of integers."""
    return [list(map(int, input().split())) for _ in range(h)]

def read_str_grid(h: int) -> List[str]:
    """Read a 2D grid as strings."""
    return [input().strip() for _ in range(h)]

def print_grid(grid: List[List[Any]]) -> None:
    """Print a 2D grid."""
    for row in grid:
        print(*row)

def yes_no(condition: bool) -> str:
    """Return 'Yes' or 'No' based on condition."""
    return "Yes" if condition else "No"

def YES_NO(condition: bool) -> str:
    """Return 'YES' or 'NO' based on condition."""
    return "YES" if condition else "NO"

# ----- Main Execution -----
def solve():
    """Main solution function."""
    # Your solution code goes here
    pass

if __name__ == "__main__":
    # For multiple test cases, uncomment the following lines:
    # t = read_int()
    # for _ in range(t):
    #     solve()
    
    # For a single test case:
    solve()


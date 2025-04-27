#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Helper functions for competitive programming.
This module contains various algorithms and functions commonly used in competitive programming.
"""

import math
import heapq
import bisect
import collections
from typing import List, Dict, Tuple, Set, Union, Optional, Any, Iterator

# Number Theory Functions
def gcd(a: int, b: int) -> int:
    """
    Calculate the greatest common divisor of two integers using Euclidean algorithm.
    
    Args:
        a: First integer
        b: Second integer
        
    Returns:
        Greatest common divisor of a and b
    """
    while b:
        a, b = b, a % b
    return a


def lcm(a: int, b: int) -> int:
    """
    Calculate the least common multiple of two integers.
    
    Args:
        a: First integer
        b: Second integer
        
    Returns:
        Least common multiple of a and b
    """
    return a * b // gcd(a, b)


def is_prime(n: int) -> bool:
    """
    Check if a number is prime using trial division.
    
    Args:
        n: The number to check
        
    Returns:
        True if n is prime, False otherwise
    """
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


def prime_factors(n: int) -> List[int]:
    """
    Get all prime factors of a number.
    
    Args:
        n: The number to factorize
        
    Returns:
        List of prime factors (may contain duplicates)
    """
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


def sieve_of_eratosthenes(n: int) -> List[int]:
    """
    Generate all prime numbers up to n using the Sieve of Eratosthenes.
    
    Args:
        n: Upper limit
        
    Returns:
        List of prime numbers less than or equal to n
    """
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    
    for i in range(2, int(math.sqrt(n)) + 1):
        if sieve[i]:
            for j in range(i * i, n + 1, i):
                sieve[j] = False
                
    return [i for i in range(n + 1) if sieve[i]]


def extended_gcd(a: int, b: int) -> Tuple[int, int, int]:
    """
    Extended Euclidean Algorithm to find gcd(a, b) and coefficients x, y such that ax + by = gcd(a, b)
    
    Args:
        a: First integer
        b: Second integer
        
    Returns:
        Tuple (gcd, x, y) where gcd is the greatest common divisor and ax + by = gcd
    """
    if a == 0:
        return (b, 0, 1)
    
    gcd, x1, y1 = extended_gcd(b % a, a)
    x = y1 - (b // a) * x1
    y = x1
    
    return (gcd, x, y)


def mod_inverse(a: int, m: int) -> int:
    """
    Calculate the modular multiplicative inverse of a modulo m.
    
    Args:
        a: The number to find the inverse for
        m: The modulus
        
    Returns:
        Modular multiplicative inverse of a modulo m
        
    Raises:
        ValueError: If inverse doesn't exist
    """
    g, x, y = extended_gcd(a, m)
    if g != 1:
        raise ValueError(f"Modular inverse doesn't exist for {a} mod {m}")
    else:
        return x % m


# Combinatorics
class Combinatorics:
    """Class for efficient calculation of combinatorial values with precomputed factorials"""
    
    def __init__(self, n: int, mod: int = 10**9 + 7):
        """
        Initialize combinatorics calculator with precomputed factorials.
        
        Args:
            n: Maximum value for which to precompute factorials
            mod: Modulus for calculations (default is 10^9 + 7)
        """
        self.mod = mod
        self.fact = [1] * (n + 1)  # factorial
        self.inv_fact = [1] * (n + 1)  # inverse factorial
        
        # Precompute factorials and inverse factorials
        for i in range(1, n + 1):
            self.fact[i] = (self.fact[i - 1] * i) % mod
            
        # Calculate inverse of factorial[n] using Fermat's little theorem
        self.inv_fact[n] = pow(self.fact[n], mod - 2, mod)
        
        # Calculate other inverse factorials
        for i in range(n - 1, 0, -1):
            self.inv_fact[i] = (self.inv_fact[i + 1] * (i + 1)) % mod
    
    def combination(self, n: int, r: int) -> int:
        """
        Calculate C(n, r) = n! / (r! * (n-r)!) with modulo.
        
        Args:
            n: Total number of items
            r: Number of items to choose
            
        Returns:
            C(n, r) % mod
        """
        if r < 0 or r > n:
            return 0
        return (self.fact[n] * self.inv_fact[r] % self.mod * self.inv_fact[n - r] % self.mod)
    
    def permutation(self, n: int, r: int) -> int:
        """
        Calculate P(n, r) = n! / (n-r)! with modulo.
        
        Args:
            n: Total number of items
            r: Number of positions to fill
            
        Returns:
            P(n, r) % mod
        """
        if r < 0 or r > n:
            return 0
        return (self.fact[n] * self.inv_fact[n - r] % self.mod)
    
    def catalan(self, n: int) -> int:
        """
        Calculate the nth Catalan number with modulo.
        
        Args:
            n: Index of Catalan number to calculate
            
        Returns:
            nth Catalan number % mod
        """
        return (self.combination(2 * n, n) * pow(n + 1, -1, self.mod)) % self.mod


# Graph Algorithms
def bfs(graph: Dict[int, List[int]], start: int) -> Dict[int, int]:
    """
    Perform Breadth-First Search on a graph.
    
    Args:
        graph: Adjacency list representation of the graph
        start: Starting vertex
        
    Returns:
        Dictionary mapping each reachable vertex to its distance from start
    """
    visited = {start: 0}
    queue = collections.deque([start])
    
    while queue:
        node = queue.popleft()
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                visited[neighbor] = visited[node] + 1
                queue.append(neighbor)
                
    return visited


def dfs(graph: Dict[int, List[int]], start: int) -> Set[int]:
    """
    Perform Depth-First Search on a graph.
    
    Args:
        graph: Adjacency list representation of the graph
        start: Starting vertex
        
    Returns:
        Set of vertices reachable from start
    """
    visited = set()
    
    def _dfs(node: int):
        visited.add(node)
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                _dfs(neighbor)
    
    _dfs(start)
    return visited


def dijkstra(graph: Dict[int, List[Tuple[int, int]]], start: int) -> Dict[int, int]:
    """
    Dijkstra's algorithm for finding shortest paths from start to all vertices.
    
    Args:
        graph: Weighted adjacency list where graph[u] contains (v, weight) pairs
        start: Starting vertex
        
    Returns:
        Dictionary mapping each vertex to the shortest distance from start
    """
    distances = {vertex: float('infinity') for vertex in graph}
    distances[start] = 0
    priority_queue = [(0, start)]
    
    while priority_queue:
        current_distance, current_vertex = heapq.heappop(priority_queue)
        
        # If we've already found a better path, skip
        if current_distance > distances[current_vertex]:
            continue
            
        for neighbor, weight in graph.get(current_vertex, []):
            distance = current_distance + weight
            
            if distance < distances.get(neighbor, float('infinity')):
                distances[neighbor] = distance
                heapq.heappush(priority_queue, (distance, neighbor))
                
    return distances


class UnionFind:
    """Union-Find (Disjoint Set) data structure implementation"""
    
    def __init__(self, n: int):
        """
        Initialize Union-Find with n elements.
        
        Args:
            n: Number of elements
        """
        self.parent = list(range(n))
        self.rank = [0] * n
        self.size = [1] * n
        self.count = n  # Number of connected components
    
    def find(self, x: int) -> int:
        """
        Find the representative (root) of the set containing x.
        
        Args:
            x: Element to find the representative for
            
        Returns:
            The representative of the set containing x
        """
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]
    
    def union(self, x: int, y: int) -> bool:
        """
        Union the sets containing x and y.
        
        Args:
            x: First element
            y: Second element
            
        Returns:
            True if x and y were in different sets before union, False otherwise
        """
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x == root_y:
            return False
        
        # Union by rank
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
        
        self.count -= 1  # Decrease the number of connected components
        return True
    
    def get_size(self, x: int) -> int:
        """
        Get the size of the set containing x.
        
        Args:
            x: Element to find the set size for
            
        Returns:
            Size of the set containing x
        """
        return self.size[self.find(x)]
    
    def is_same(self, x: int, y: int) -> bool:
        """
        Check if x and y are in the same set.
        
        Args:
            x: First element
            y: Second element
            
        Returns:
            True if x and y are in the same set, False otherwise
        """
        return self.find(x) == self.find(y)


# Binary Search Utilities
def binary_search(arr: List[int], target: int) -> int:
    """
    Binary search to find the index of target in a sorted array.
    
    Args:
        arr: Sorted array to search in
        target: Value to search for
        
    Returns:
        Index of target if found, -1 otherwise
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


def binary_search_leftmost(arr: List[int], target: int) -> int:
    """
    Find the leftmost (first) occurrence of target in a sorted array.
    
    Args:
        arr: Sorted array to search in
        target: Value to search for
        
    Returns:
        Index of the leftmost occurrence of target, or index where it would be inserted
    """
    return bisect.bisect_left(arr, target)


def binary_search_rightmost(arr: List[int], target: int) -> int:
    """
    Find the rightmost (last) occurrence of target in a sorted array.
    
    Args:
        arr: Sorted array to search in
        target: Value to search for
        
    Returns:
        Index after the rightmost occurrence of target, or index where it would be inserted
    """
    return bisect.bisect_right(arr, target)


# String Algorithms
def z_function(s: str) -> List[int]:
    """
    Z-function (Z-algorithm) for string matching.
    Z[i] is the length of the longest substring starting from s[i]
    which is also a prefix of s.
    
    Args:
        s: Input string
        
    Returns:
        Z array where Z[i] is the length of the longest common prefix of s and s[i:]
    """
    n = len(s)
    z = [0] * n
    z[0] = n
    
    # [l, r] is the rightmost substring that is also a prefix
    l, r = 0, 0
    for i in range(1, n):
        if i <= r:
            z[i] = min(r - i + 1, z[i - l])
            
        # Try to extend the match
        while i + z[i] < n and s[z[i]] == s[i + z[i]]:
            z[i] += 1
            
        # Update the rightmost match
        if i + z[i] - 1 > r:
            l, r = i, i + z[i] - 1
            
    return z


def kmp_preprocess(pattern: str) -> List[int]:
    """
    Compute the Knuth-Morris-Pratt (KMP) failure function.
    
    Args:
        pattern: Pattern string
        
    Returns:
        Failure function (partial match table)
    """
    m = len(pattern)
    lps = [0] * m  # Longest proper prefix which is also suffix
    
    for i in range(1, m):
        j = lps[i - 1]
        while j > 0 and pattern[i] != pattern[j]:
            j = lps[j - 1]
        if pattern[i] == pattern[j]:
            j += 1
        lps[i] = j
        
    return lps


def kmp_search(text: str, pattern: str) -> List[int]:
    """
    Knuth-Morris-Pratt algorithm for pattern matching.
    
    Args:
        text: Text to search in
        pattern: Pattern to search for
        
    Returns:
        List of starting indices of all occurrences of pattern in text
    """
    if not pattern:
        return list(range(len(text) + 1))
    
    if not text:
        return []
    
    n, m = len(text), len(pattern)
    lps = kmp_preprocess(pattern)
    occurrences = []
    
    j = 0  # Index for pattern
    for i in range(n):  # Index for text
        while j > 0 and text[i] != pattern[j]:
            j = lps[j - 1]
            
        if text[i] == pattern[j]:
            j += 1
            
        if j == m:
            occurrences.append(i - m + 1)
            j = lps[j - 1]
            
    return occurrences


# Segment Tree Implementation
class SegmentTree:
    """
    Segment Tree implementation for range queries and point updates.
    Example operations include sum, min, max, etc.
    """
    
    def __init__(self, arr: List[int], operation: str = 'sum'):
        """
        Initialize Segment Tree from an array.
        
        Args:
            arr: Input array
            operation: Type of query operation ('sum', 'min', 'max')
        """
        self.n = len(arr)
        self.operation = operation
        
        # Define the operation function
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
        
        # Build the tree
        self.tree = [self.identity] * (4 * self.n)
        self._build(arr, 1, 0, self.n - 1)
    
    def _build(self, arr: List[int], node: int, start: int, end: int) -> None:
        """
        Recursively build the segment tree.
        
        Args:
            arr: Input array
            node: Current node index in tree
            start: Start index of the current segment
            end: End index of the current segment
        """
        if start == end:
            self.tree[node] = arr[start]
            return
        
        mid = (start + end) // 2
        self._build(arr, 2 * node, start, mid)
        self._build(arr, 2 * node + 1, mid + 1, end)
        self.tree[node] = self.op(self.tree[2 * node], self.tree[2 * node + 1])
    
    def update(self, idx: int, val: int) -> None:
        """
        Update the value at index idx to val.
        
        Args:
            idx: Index to update
            val: New value
        """
        self._update(1, 0, self.n - 1, idx, val)
    
    def _update(self, node: int, start: int, end: int, idx: int, val: int) -> None:
        """
        Recursively update the segment tree.
        
        Args:
            node: Current node index in tree
            start: Start index of the current segment
            end: End index of the current segment
            idx: Index to update
            val: New value
        """
        if start == end:
            self.tree[node] = val
            return
        
        mid = (start + end) // 2
        if idx <= mid:
            self._update(2 * node, start, mid, idx, val)
        else:
            self._update(2 * node + 1, mid + 1, end, idx, val)
        
        self.tree[node] = self.op(self.tree[2 * node], self.tree[2 * node + 1])
    
    def query(self, left: int, right: int) -> int:
        """
        Query the segment tree for the operation result in range [left, right].
        
        Args:
            left: Left boundary of the query range
            right: Right boundary of the query range
            
        Returns:
            Operation result for the range [left, right]
        """
        return self._query(1, 0, self.n - 1, left, right)
    
    def _query(self, node: int, start: int, end: int, left: int, right: int) -> int:
        """
        Recursively query the segment tree.
        
        Args:
            node: Current node index in tree
            start: Start index of the current segment
            end: End index of the current segment
            left: Left boundary of the query range
            right: Right boundary of the query range
            
        Returns:
            Operation result for the range [left, right]
        """
        # No overlap
        if start > right or end < left:
            return self.identity
        
        # Complete overlap
        if left <= start and end <= right:
            return self.tree[node]
        
        # Partial overlap
        mid = (start + end) // 2
        left_result = self._query(2 * node, start, mid, left, right)
        right_result = self._query(2 * node + 1, mid + 1, end, left, right)
        
        return self.op(left_result, right_result)


# Lazy Propagation Segment Tree for range updates
class LazySegmentTree:
    """
    Segment Tree with lazy propagation for range updates and range queries.
    """
    
    def __init__(self, arr: List[int], operation: str = 'sum'):
        """
        Initialize Lazy Segment Tree from an array.
        
        Args:
            arr: Input array
            operation: Type of query operation ('sum', 'min', 'max')
        """
        self.n = len(arr)
        self.operation = operation
        
        # Define the operation function
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
        
        # Build the tree
        self.tree = [self.identity] * (4 * self.n)
        self.lazy = [0] * (4 * self.n)  # Lazy updates
        self._build(arr, 1, 0, self.n - 1)
    
    def _build(self, arr: List[int], node: int, start: int, end: int) -> None:
        """
        Recursively build the segment tree.
        
        Args:
            arr: Input array
            node: Current node index in tree
            start: Start index of the current segment
            end: End index of the current segment
        """
        if start == end:
            self.tree[node] = arr[start]
            return
        
        mid = (start + end) // 2
        self._build(arr, 2 * node, start, mid)
        self._build(arr, 2 * node + 1, mid + 1, end)
        self.tree[node] = self.op(self.tree[2 * node], self.tree[2 * node + 1])
    
    def _propagate(self, node: int, start: int, end: int) -> None:
        """
        Propagate lazy updates downward.
        
        Args:
            node: Current node index in tree
            start: Start index of the current segment
            end: End index of the current segment
        """
        if self.lazy[node] != 0:
            self.tree[node] += (end - start + 1) * self.lazy[node]  # For sum
            
            if start != end:  # If not leaf node, propagate to children
                self.lazy[2 * node] += self.lazy[node]
                self.lazy[2 * node + 1] += self.lazy[node]
                
            self.lazy[node] = 0  # Reset lazy value
    
    def update_range(self, left: int, right: int, val: int) -> None:
        """
        Update the range [left, right] by adding val to each element.
        
        Args:
            left: Left boundary of the update range
            right: Right boundary of the update range
            val: Value to add to each element in the range
        """
        self._update_range(1, 0, self.n - 1, left, right, val)
    
    def _update_range(self, node: int, start: int, end: int, left: int, right: int, val: int) -> None:
        """
        Recursively update a range in the segment tree.
        
        Args:
            node: Current node index in tree
            start: Start index of the current segment
            end: End index of the current segment
            left: Left boundary of the update range
            right: Right boundary of the update range
            val: Value to add to each element in the range
        """
        self._propagate(node, start, end)
        
        # No overlap
        if start > right or end < left:
            return
        
        # Complete overlap
        if left <= start and end <= right:
            self.tree[node] += (end - start + 1) * val  # For sum
            
            if start != end:  # If not leaf node, mark children for lazy update
                self.lazy[2 * node] += val
                self.lazy[2 * node + 1] += val
                
            return
        
        # Partial overlap
        mid = (start + end) // 2
        self._update_range(2 * node, start, mid, left, right, val)
        self._update_range(2 * node + 1, mid + 1, end, left, right, val)
        
        self.tree[node] = self.op(self.tree[2 * node], self.tree[2 * node + 1])
    
    def query_range(self, left: int, right: int) -> int:
        """
        Query the segment tree for the operation result in range [left, right].
        
        Args:
            left: Left boundary of the query range
            right: Right boundary of the query range
            
        Returns:
            Operation result for the range [left, right]
        """
        return self._query_range(1, 0, self.n - 1, left, right)
    
    def _query_range(self, node: int, start: int, end: int, left: int, right: int) -> int:
        """
        Recursively query the segment tree.
        
        Args:
            node: Current node index in tree
            start: Start index of the current segment
            end: End index of the current segment
            left: Left boundary of the query range
            right: Right boundary of the query range
            
        Returns:
            Operation result for the range [left, right]
        """
        # No overlap
        if start > right or end < left:
            return self.identity
        
        self._propagate(node, start, end)
        
        # Complete overlap
        if left <= start and end <= right:
            return self.tree[node]
        
        # Partial overlap
        mid = (start + end) // 2
        left_result = self._query_range(2 * node, start, mid, left, right)
        right_result = self._query_range(2 * node + 1, mid + 1, end, left, right)
        
        return self.op(left_result, right_result)


# Fenwick Tree (Binary Indexed Tree)
class FenwickTree:
    """
    Fenwick Tree (Binary Indexed Tree) implementation for efficient range sum queries
    and point updates in O(log n) time.
    """
    
    def __init__(self, n: int):
        """
        Initialize Fenwick Tree with size n.
        
        Args:
            n: Size of the tree
        """
        self.size = n
        self.tree = [0] * (n + 1)  # 1-indexed
    
    def update(self, idx: int, delta: int) -> None:
        """
        Add delta to the element at index idx.
        
        Args:
            idx: Index to update (0-indexed)
            delta: Value to add
        """
        idx += 1  # Convert to 1-indexed
        while idx <= self.size:
            self.tree[idx] += delta
            idx += idx & -idx  # Add least significant bit
    
    def prefix_sum(self, idx: int) -> int:
        """
        Calculate the sum of elements from index 0 to idx (inclusive).
        
        Args:
            idx: End index of the prefix sum (0-indexed)
            
        Returns:
            Sum of elements in range [0, idx]
        """
        idx += 1  # Convert to 1-indexed
        result = 0
        while idx > 0:
            result += self.tree[idx]
            idx -= idx & -idx  # Remove least significant bit
        return result
    
    def range_sum(self, left: int, right: int) -> int:
        """
        Calculate the sum of elements from index left to right (inclusive).
        
        Args:
            left: Start index of the range (0-indexed)
            right: End index of the range (0-indexed)
            
        Returns:
            Sum of elements in range [left, right]
        """
        return self.prefix_sum(right) - (self.prefix_sum(left - 1) if left > 0 else 0)


# 2D Fenwick Tree
class FenwickTree2D:
    """
    2D Fenwick Tree for efficient 2D range sum queries and point updates.
    """
    
    def __init__(self, rows: int, cols: int):
        """
        Initialize 2D Fenwick Tree with given dimensions.
        
        Args:
            rows: Number of rows
            cols: Number of columns
        """
        self.rows = rows
        self.cols = cols
        self.tree = [[0] * (cols + 1) for _ in range(rows + 1)]  # 1-indexed
    
    def update(self, row: int, col: int, delta: int) -> None:
        """
        Add delta to the element at position (row, col).
        
        Args:
            row: Row index (0-indexed)
            col: Column index (0-indexed)
            delta: Value to add
        """
        row += 1  # Convert to 1-indexed
        col += 1
        
        i = row
        while i <= self.rows:
            j = col
            while j <= self.cols:
                self.tree[i][j] += delta
                j += j & -j  # Add least significant bit of j
            i += i & -i  # Add least significant bit of i
    
    def prefix_sum(self, row: int, col: int) -> int:
        """
        Calculate the sum of elements in the rectangle from (0,0) to (row,col).
        
        Args:
            row: Row index (0-indexed)
            col: Column index (0-indexed)
            
        Returns:
            Sum of elements in rectangle [(0,0), (row,col)]
        """
        row += 1  # Convert to 1-indexed
        col += 1
        result = 0
        
        i = row
        while i > 0:
            j = col
            while j > 0:
                result += self.tree[i][j]
                j -= j & -j  # Remove least significant bit of j
            i -= i & -i  # Remove least significant bit of i
            
        return result
    
    def range_sum(self, row1: int, col1: int, row2: int, col2: int) -> int:
        """
        Calculate the sum of elements in the rectangle from (row1,col1) to (row2,col2).
        
        Args:
            row1: Top row index (0-indexed)
            col1: Left column index (0-indexed)
            row2: Bottom row index (0-indexed)
            col2: Right column index (0-indexed)
            
        Returns:
            Sum of elements in rectangle [(row1,col1), (row2,col2)]
        """
        # Calculate using inclusion-exclusion principle
        result = self.prefix_sum(row2, col2)
        
        if row1 > 0:
            result -= self.prefix_sum(row1 - 1, col2)
        if col1 > 0:
            result -= self.prefix_sum(row2, col1 - 1)
        if row1 > 0 and col1 > 0:
            result += self.prefix_sum(row1 - 1, col1 - 1)
            
        return result


# Dynamic Programming Utilities
def longest_increasing_subsequence(arr: List[int]) -> int:
    """
    Find the length of the longest increasing subsequence.
    
    Args:
        arr: Input array
        
    Returns:
        Length of the longest increasing subsequence
    """
    if not arr:
        return 0
        
    n = len(arr)
    dp = [1] * n
    
    for i in range(1, n):
        for j in range(i):
            if arr[i] > arr[j]:
                dp[i] = max(dp[i], dp[j] + 1)
                
    return max(dp)


def longest_common_subsequence(s1: str, s2: str) -> int:
    """
    Find the length of the longest common subsequence between two strings.
    
    Args:
        s1: First string
        s2: Second string
        
    Returns:
        Length of the longest common subsequence
    """
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
                
    return dp[m][n]


def knapsack(weights: List[int], values: List[int], capacity: int) -> int:
    """
    Solve the 0/1 knapsack problem.
    
    Args:
        weights: List of item weights
        values: List of item values
        capacity: Knapsack capacity
        
    Returns:
        Maximum value that can be put in the knapsack
    """
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            if weights[i - 1] <= w:
                dp[i][w] = max(values[i - 1] + dp[i - 1][w - weights[i - 1]], dp[i - 1][w])
            else:
                dp[i][w] = dp[i - 1][w]
                
    return dp[n][capacity]


def edit_distance(s1: str, s2: str) -> int:
    """
    Calculate the minimum edit distance (Levenshtein distance) between two strings.
    
    Args:
        s1: First string
        s2: Second string
        
    Returns:
        Minimum number of operations (insert, delete, replace) to convert s1 to s2
    """
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Base cases: empty string transformations
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
                    dp[i - 1][j],      # Delete
                    dp[i][j - 1],      # Insert
                    dp[i - 1][j - 1]   # Replace
                )
                
    return dp[m][n]


# Matrix Operations
def matrix_multiply(A: List[List[int]], B: List[List[int]], mod: int = None) -> List[List[int]]:
    """
    Multiply two matrices A and B.
    
    Args:
        A: First matrix
        B: Second matrix
        mod: Modulus for the result (optional)
        
    Returns:
        Result of A * B
        
    Raises:
        ValueError: If matrix dimensions are incompatible
    """
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


def matrix_power(A: List[List[int]], power: int, mod: int = None) -> List[List[int]]:
    """
    Calculate A^power using binary exponentiation.
    
    Args:
        A: Square matrix
        power: Power to raise A to
        mod: Modulus for the result (optional)
        
    Returns:
        Result of A^power
        
    Raises:
        ValueError: If matrix is not square or power is negative
    """
    if not A or len(A) != len(A[0]):
        raise ValueError("Matrix must be square")
        
    if power < 0:
        raise ValueError("Power must be non-negative")
        
    n = len(A)
    
    # Identity matrix
    result = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
    
    # Binary exponentiation
    while power > 0:
        if power & 1:  # power is odd
            result = matrix_multiply(result, A, mod)
        A = matrix_multiply(A, A, mod)
        power >>= 1
        
    return result


# Geometry Utilities
class Point:
    """2D point representation"""
    
    def __init__(self, x: float, y: float):
        """
        Initialize a 2D point.
        
        Args:
            x: x-coordinate
            y: y-coordinate
        """
        self.x = x
        self.y = y
    
    def __eq__(self, other):
        if not isinstance(other, Point):
            return False
        return abs(self.x - other.x) < 1e-9 and abs(self.y - other.y) < 1e-9
    
    def __hash__(self):
        return hash((self.x, self.y))
    
    def distance(self, other) -> float:
        """
        Calculate Euclidean distance to another point.
        
        Args:
            other: Another Point object
            
        Returns:
            Euclidean distance between this point and other
        """
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)


def cross_product(p1: Point, p2: Point, p3: Point) -> float:
    """
    Calculate the cross product (p2-p1) × (p3-p1).
    
    Args:
        p1: First point
        p2: Second point
        p3: Third point
        
    Returns:
        Cross product value
    """
    return (p2.x - p1.x) * (p3.y - p1.y) - (p2.y - p1.y) * (p3.x - p1.x)


def convex_hull(points: List[Point]) -> List[Point]:
    """
    Compute the convex hull of a set of points using Graham's scan algorithm.
    
    Args:
        points: List of Point objects
        
    Returns:
        List of Point objects forming the convex hull
    """
    if len(points) <= 2:
        return points.copy()
    
    # Find the point with the lowest y-coordinate (and leftmost if tied)
    p0 = min(points, key=lambda p: (p.y, p.x))
    
    # Sort points by polar angle with respect to p0
    def polar_angle(p):
        if p == p0:
            return -math.inf
        return math.atan2(p.y - p0.y, p.x - p0.x)
    
    sorted_points = sorted(points, key=polar_angle)
    
    # Build the convex hull
    hull = [sorted_points[0], sorted_points[1]]
    
    for i in range(2, len(sorted_points)):
        while len(hull) > 1 and cross_product(hull[-2], hull[-1], sorted_points[i]) <= 0:
            hull.pop()
        hull.append(sorted_points[i])
    
    return hull


# Flow Algorithms
def ford_fulkerson(graph: Dict[int, Dict[int, int]], source: int, sink: int) -> int:
    """
    Ford-Fulkerson algorithm for maximum flow.
    
    Args:
        graph: Adjacency list representation of the flow network,
               where graph[u][v] represents the capacity from u to v
        source: Source vertex
        sink: Sink vertex
        
    Returns:
        Maximum flow from source to sink
    """
    def dfs(u: int, flow: int):
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


# Trie Implementation
class TrieNode:
    """Node in a Trie"""
    
    def __init__(self):
        """Initialize an empty trie node"""
        self.children = {}
        self.is_end_of_word = False


class Trie:
    """
    Trie (prefix tree) implementation for efficient string operations.
    """
    
    def __init__(self):
        """Initialize an empty trie"""
        self.root = TrieNode()
    
    def insert(self, word: str) -> None:
        """
        Insert a word into the trie.
        
        Args:
            word: Word to insert
        """
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True
    
    def search(self, word: str) -> bool:
        """
        Search for a word in the trie.
        
        Args:
            word: Word to search for
            
        Returns:
            True if the word exists in the trie, False otherwise
        """
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end_of_word
    
    def starts_with(self, prefix: str) -> bool:
        """
        Check if there is any word in the trie that starts with the given prefix.
        
        Args:
            prefix: Prefix to check
            
        Returns:
            True if there is a word with the given prefix, False otherwise
        """
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True


# Utility Functions for Competitive Programming
def fast_input() -> Iterator[str]:
    """
    Generator for fast input reading.
    
    Yields:
        Tokens from input
    """
    import sys
    for line in sys.stdin:
        yield from line.split()


def binary_search_predicate(low: float, high: float, predicate, epsilon: float = 1e-9) -> float:
    """
    Binary search for a value satisfying a predicate function.
    
    Args:
        low: Lower bound
        high: Upper bound
        predicate: Function that returns True for values that satisfy the condition
        epsilon: Precision for floating-point comparison
        
    Returns:
        The smallest value in [low, high] that satisfies the predicate
        or high+epsilon if no such value exists
    """
    while high - low > epsilon:
        mid = (low + high) / 2
        if predicate(mid):
            high = mid
        else:
            low = mid
    return high


def coordinate_compression(arr: List[int]) -> Tuple[List[int], Dict[int, int]]:
    """
    Compress coordinates to reduce the range of values.
    
    Args:
        arr: Original array with potentially large range
        
    Returns:
        Tuple of (compressed array, mapping from compressed to original)
    """
    sorted_arr = sorted(set(arr))
    compress_map = {val: i for i, val in enumerate(sorted_arr)}
    compressed = [compress_map[x] for x in arr]
    return compressed, {i: val for i, val in enumerate(sorted_arr)}


def discrete_binary_search(predicate, low: int, high: int) -> int:
    """
    Binary search for the smallest integer x in [low, high] such that predicate(x) is True.
    
    Args:
        predicate: Function that returns True for valid values
        low: Lower bound (inclusive)
        high: Upper bound (inclusive)
        
    Returns:
        Smallest x such that predicate(x) is True, or high+1 if none exists
    """
    while low < high:
        mid = (low + high) // 2
        if predicate(mid):
            high = mid
        else:
            low = mid + 1
    
    return low if predicate(low) else high + 1


def fast_prime_factors(n: int, precomputed_spf: List[int]) -> List[int]:
    """
    Get prime factors using precomputed smallest prime factor array.
    
    Args:
        n: Number to factorize
        precomputed_spf: Precomputed smallest prime factor for each number
        
    Returns:
        List of prime factors
    """
    factors = []
    while n > 1:
        factors.append(precomputed_spf[n])
        n //= precomputed_spf[n]
    return factors


def precompute_spf(n: int) -> List[int]:
    """
    Precompute smallest prime factor for numbers up to n.
    
    Args:
        n: Upper limit
        
    Returns:
        List where spf[i] is the smallest prime factor of i
    """
    spf = list(range(n + 1))
    for i in range(4, n + 1, 2):
        spf[i] = 2
    
    for i in range(3, int(math.sqrt(n)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, n + 1, i):
                if spf[j] == j:  # Update if not already set
                    spf[j] = i
    
    return spf


def sieve_of_divisors(n: int) -> List[List[int]]:
    """
    Generate all divisors for numbers up to n.
    
    Args:
        n: Upper limit
        
    Returns:
        List where divisors[i] is the list of all divisors of i
    """
    divisors = [[] for _ in range(n + 1)]
    
    for i in range(1, n + 1):
        for j in range(i, n + 1, i):
            divisors[j].append(i)
    
    return divisors


def modular_exponentiation(base: int, exponent: int, modulus: int) -> int:
    """
    Calculate (base^exponent) % modulus efficiently.
    
    Args:
        base: Base of the exponentiation
        exponent: Exponent
        modulus: Modulus
        
    Returns:
        (base^exponent) % modulus
    """
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


# Fast I/O for Competitive Programming
def read_int():
    """Read a single integer from input"""
    return int(input())


def read_ints():
    """Read multiple integers from a single line"""
    return list(map(int, input().split()))


def read_array(n):
    """Read an array of n integers"""
    return list(map(int, input().split()))


def read_matrix(n, m):
    """Read an n×m matrix of integers"""
    return [list(map(int, input().split())) for _ in range(n)]


def read_char_matrix(n, m):
    """Read an n×m matrix of characters"""
    return [list(input()) for _ in range(n)]


if __name__ == "__main__":
    # Example usage
    pass
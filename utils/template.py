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
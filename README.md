# Competitive Programming Repository

This repository is for my personal practice and solutions for competitive programming problems.

## Contents

This repository may contain solutions and code related to problems from various online judge platforms, including but not limited to:

* AtCoder
* Codeforces
* AIZU Online Judge (AOJ)
* LeetCode
* ...

The code is primarily written in:

* Python, C++ etc.

The structure is generally organized by platform and then potentially by contest or problem set.

```
competitive_programming/
├── atcoder/
│   ├── practice/
│   └── contest/
├── aizu_online_judge/
│   ├── practice/
│   └── contest/
├── codeforces/
│   ├── practice/
│   └── contest/
├── algorithms/
│   ├── data_structures/
│   ├── dp/
│   ├── flow/
│   ├── graph/
│   ├── math/
│   ├── search/
│   └── string/
├── utils/
└── ...
```

## Algorithms Collection

This repository contains implementations of common algorithms and data structures used in competitive programming. Below is a summary of the implemented algorithms:

### Data Structures
- **binary_indexed_tree.py**: Implementation of Fenwick Tree (BIT) for efficient cumulative sum and point updates
- **coordinate_compression.py**: Algorithm to compress large ranges of values into consecutive integers
- **segment_tree.py**: Efficient data structure for range queries and point updates
- **sparse_table.py**: Data structure for fast Range Minimum Queries (RMQ)
- **trie.py**: Tree data structure for efficient storage and retrieval of strings
- **union_find.py**: Disjoint Set Union (DSU) data structure for managing disjoint sets

### Dynamic Programming (DP)
- **basic_dp.py**: Implementation of basic dynamic programming techniques
- **knapsack.py**: Various knapsack problem implementations (0/1, fractional, unbounded)
- **sequence_dp.py**: Array/string-related DP algorithms like Longest Increasing Subsequence (LIS), Longest Common Subsequence (LCS), and Edit Distance

### Flow Algorithms
- **ford_fulkerson.py**: Ford-Fulkerson algorithm for maximum flow problems and bipartite matching

### Graph Algorithms
- **bellman_ford.py**: Single-source shortest path algorithm for graphs with negative weights
- **bfs.py**: Breadth-first search for shortest paths and connected components
- **dfs.py**: Depth-first search for graph traversal
- **dijkstra.py**: Efficient single-source shortest path algorithm for graphs with non-negative weights
- **floyd_warshall.py**: All-pairs shortest path algorithm
- **kruskal.py**: Algorithm for finding minimum spanning tree
- **topological_sort.py**: Ordering vertices in directed acyclic graph

### Math Algorithms
- **combinations.py**: Efficient calculation of permutations, combinations and related counting functions
- **fast_power.py**: Binary exponentiation (repeated squaring) for efficient power calculation
- **fft.py**: Fast Fourier Transform for polynomial multiplication and convolution
- **game_theory.py**: Algorithms for combinatorial game theory including Nim, Grundy numbers, and game values
- **matrix.py**: Matrix operations including multiplication, exponentiation, and applications to recurrence relations
- **modular.py**: Modular arithmetic operations, modular inverse, and modular exponentiation
- **number_theory.py**: Number theory algorithms including GCD, LCM, extended Euclidean algorithm, Euler's totient function, Chinese remainder theorem, and more
- **prime.py**: Prime-related algorithms including primality test, prime factorization, and Sieve of Eratosthenes

### Search Algorithms
- **binary_search.py**: Efficient binary search and its applications on sorted arrays

### String Algorithms
- **kmp.py**: Knuth-Morris-Pratt algorithm for efficient string pattern matching
- **manacher.py**: Linear-time algorithm for finding longest palindromic substrings
- **rabin_karp.py**: Hash-based pattern matching algorithm
- **z_algorithm.py**: Algorithm for efficient prefix matching at each position in a string

## Usage

This is a personal repository for my own learning and reference. Feel free to explore the code if you are interested.

## License

This repository is licensed under the [Apache License 2.0](https://github.com/yf591/competitive_programming/blob/main/LICENSE). See the `LICENSE` file for more information.

## Notes

* This repository is a work in progress and will be updated as I solve more problems.
* The solutions provided here are my own and may not always be the most optimal or efficient.

---

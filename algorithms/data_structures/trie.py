"""
トライ木（Trie）
- 文字列を格納するための木構造データ構造
- 共通の接頭辞を共有することでメモリ効率が良い
- 主な操作:
  - 文字列の挿入
  - 文字列の検索
  - 接頭辞を持つ文字列の検索
- 特徴:
  - 検索・挿入・削除の計算量はO(L)（Lは文字列の長さ）
  - 共通接頭辞を共有するためメモリ効率が良い
- 用途:
  - 辞書の実装
  - 自動補完
  - パターンマッチング
  - 最長共通接頭辞の検索
"""

from typing import Dict, List, Optional, Set
from collections import defaultdict


class TrieNode:
    """
    トライ木のノード
    
    Attributes:
        children: 子ノードの辞書（キーは文字、値は子ノード）
        is_end_of_word: このノードが単語の終端かどうか
        word: このノードが単語の終端である場合、その単語
    """
    
    def __init__(self):
        self.children = {}  # 子ノードの辞書
        self.is_end_of_word = False  # 単語の終端かどうか
        self.word = None  # 単語の終端の場合、その単語
        self.count = 0  # このノードを通過する単語の数


class Trie:
    """
    トライ木の実装
    
    Attributes:
        root: トライ木のルートノード
    """
    
    def __init__(self):
        """トライ木の初期化"""
        self.root = TrieNode()
    
    def insert(self, word: str):
        """
        単語をトライ木に挿入
        
        Args:
            word: 挿入する単語
        """
        node = self.root
        
        # 各文字に対応するノードを作成または辿る
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
            node.count += 1  # このノードを通過する単語数を増やす
            
        # 単語の終端をマーク
        node.is_end_of_word = True
        node.word = word
    
    def search(self, word: str) -> bool:
        """
        単語がトライ木に存在するか検索
        
        Args:
            word: 検索する単語
            
        Returns:
            bool: 単語が存在するかどうか
        """
        node = self._find_node(word)
        return node is not None and node.is_end_of_word
    
    def starts_with(self, prefix: str) -> bool:
        """
        指定された接頭辞から始まる単語がトライ木に存在するか検索
        
        Args:
            prefix: 接頭辞
            
        Returns:
            bool: 接頭辞から始まる単語が存在するかどうか
        """
        return self._find_node(prefix) is not None
    
    def _find_node(self, prefix: str) -> Optional[TrieNode]:
        """
        指定された接頭辞に対応するノードを返す
        
        Args:
            prefix: 接頭辞
            
        Returns:
            TrieNode: 接頭辞に対応するノード、存在しない場合はNone
        """
        node = self.root
        
        # 接頭辞に対応するノードを辿る
        for char in prefix:
            if char not in node.children:
                return None
            node = node.children[char]
            
        return node
    
    def get_words_with_prefix(self, prefix: str) -> List[str]:
        """
        指定された接頭辞から始まる全ての単語を取得
        
        Args:
            prefix: 接頭辞
            
        Returns:
            List[str]: 接頭辞から始まる単語のリスト
        """
        result = []
        node = self._find_node(prefix)
        
        if node:
            self._collect_words(node, result)
            
        return result
    
    def _collect_words(self, node: TrieNode, result: List[str]):
        """
        ノード以下の全ての単語を収集（DFS）
        
        Args:
            node: 開始ノード
            result: 結果を格納するリスト
        """
        if node.is_end_of_word:
            result.append(node.word)
            
        for char, child in node.children.items():
            self._collect_words(child, result)
    
    def delete(self, word: str) -> bool:
        """
        単語をトライ木から削除
        
        Args:
            word: 削除する単語
            
        Returns:
            bool: 削除に成功したかどうか
        """
        return self._delete_recursive(self.root, word, 0)
    
    def _delete_recursive(self, node: TrieNode, word: str, depth: int) -> bool:
        """
        単語を再帰的に削除
        
        Args:
            node: 現在のノード
            word: 削除する単語
            depth: 現在の深さ（文字位置）
            
        Returns:
            bool: このノードが削除可能かどうか
        """
        # 単語の終端に達した場合
        if depth == len(word):
            # 単語が存在しない場合
            if not node.is_end_of_word:
                return False
                
            # 単語の終端マークを解除
            node.is_end_of_word = False
            node.word = None
            
            # 子ノードがない場合、このノードは削除可能
            return len(node.children) == 0
        
        char = word[depth]
        
        # 文字に対応する子ノードがない場合
        if char not in node.children:
            return False
            
        # 子ノードを再帰的に削除
        should_delete_child = self._delete_recursive(node.children[char], word, depth + 1)
        
        # 子ノードが削除可能な場合
        if should_delete_child:
            del node.children[char]
            # このノードが単語の終端でなく、他に子ノードがない場合、このノードも削除可能
            return not node.is_end_of_word and len(node.children) == 0
            
        return False
    
    def count_words_with_prefix(self, prefix: str) -> int:
        """
        指定された接頭辞から始まる単語の数を取得
        
        Args:
            prefix: 接頭辞
            
        Returns:
            int: 接頭辞から始まる単語の数
        """
        node = self._find_node(prefix)
        if node:
            return node.count
        return 0
    
    def get_longest_common_prefix(self) -> str:
        """
        トライ木内の全ての単語の最長共通接頭辞を取得
        
        Returns:
            str: 最長共通接頭辞
        """
        if not self.root.children:
            return ""
            
        prefix = []
        node = self.root
        
        # 分岐が1つだけの間は共通接頭辞
        while len(node.children) == 1:
            char = next(iter(node.children))
            prefix.append(char)
            node = node.children[char]
            
            # 単語の終端に達した場合は終了
            if node.is_end_of_word:
                break
                
        return ''.join(prefix)


# 使用例
def example():
    trie = Trie()
    words = [
        "apple", "app", "apricot", "banana", "bat", "batman", "battle"
    ]
    
    print("===== トライ木 =====")
    
    # 単語の挿入
    for word in words:
        trie.insert(word)
    print(f"挿入した単語: {words}")
    
    # 単語の検索
    test_words = ["apple", "app", "appl", "ban", "batman"]
    for word in test_words:
        print(f"'{word}' はトライ木に存在する？: {trie.search(word)}")
    
    # 接頭辞の検索
    prefixes = ["ap", "ba", "bat", "z"]
    for prefix in prefixes:
        print(f"接頭辞 '{prefix}' で始まる単語は存在する？: {trie.starts_with(prefix)}")
    
    # 接頭辞から始まる単語の取得
    for prefix in ["ap", "ba"]:
        words = trie.get_words_with_prefix(prefix)
        print(f"接頭辞 '{prefix}' から始まる単語: {words}")
    
    # 単語の削除
    delete_words = ["apple", "bat"]
    for word in delete_words:
        trie.delete(word)
    print(f"\n削除した単語: {delete_words}")
    
    # 削除後の検索
    for word in ["apple", "app", "bat", "batman"]:
        print(f"'{word}' はトライ木に存在する？: {trie.search(word)}")
    
    # 接頭辞から始まる単語の数
    for prefix in ["ap", "ba", "bat"]:
        count = trie.count_words_with_prefix(prefix)
        print(f"接頭辞 '{prefix}' から始まる単語の数: {count}")
    
    # 共通接頭辞の例
    common_prefix_trie = Trie()
    common_prefix_words = ["flower", "flow", "flight"]
    for word in common_prefix_words:
        common_prefix_trie.insert(word)
    
    print(f"\n単語 {common_prefix_words} の最長共通接頭辞: {common_prefix_trie.get_longest_common_prefix()}")


if __name__ == "__main__":
    example()
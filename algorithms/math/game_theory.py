"""
ゲーム理論アルゴリズム集 (Game Theory Algorithms)
- 組合せゲーム理論に関するアルゴリズム群
- 主な実装:
  - Nim (ニム) ゲーム
  - Grundy数 (Sprague-Grundy 定理)
  - Hackenbush
  - Subtraction game
- 計算量:
  - Nim関連: O(n) (nは山の数)
  - Grundy数計算: ゲームに依存
- 応用:
  - 先手・後手の必勝判定
  - ゲームの最適戦略
"""

from typing import List, Dict, Set, Tuple, Optional
import functools


def nim_sum(piles: List[int]) -> int:
    """
    Nimゲームの山の状態からNim和を計算
    
    Args:
        piles: 各山の石の数のリスト
    
    Returns:
        Nim和（XOR和）
    """
    result = 0
    for pile in piles:
        result ^= pile
    return result


def nim_winning_move(piles: List[int]) -> Optional[Tuple[int, int]]:
    """
    Nimゲームで勝つための手を計算
    
    Args:
        piles: 各山の石の数のリスト
    
    Returns:
        (山のインデックス, 取る石の数)の組、または勝つ手がなければNone
    """
    nim_xor = nim_sum(piles)
    
    # nim_xor == 0 なら勝つ手はない
    if nim_xor == 0:
        return None
    
    # 各山について調べる
    for i, pile in enumerate(piles):
        # 取る石の数を計算
        target = pile ^ nim_xor
        
        # targetがpileより小さければ、pile - targetだけ石を取れば良い
        if target < pile:
            return (i, pile - target)
    
    # ここに到達することは理論上ない
    return None


def is_nim_winning_position(piles: List[int]) -> bool:
    """
    Nimゲームの状態が勝ち状態かどうかを判定
    
    Args:
        piles: 各山の石の数のリスト
    
    Returns:
        現在の手番のプレイヤーが必勝なら True、そうでなければ False
    """
    return nim_sum(piles) != 0


def mex(s: Set[int]) -> int:
    """
    集合Sに含まれない最小の非負整数（Minimum Excludant）を返す
    
    Args:
        s: 非負整数の集合
    
    Returns:
        s に含まれない最小の非負整数
    """
    i = 0
    while i in s:
        i += 1
    return i


def grundy_number(state: int, move_func, memo=None) -> int:
    """
    与えられた状態のGrundy数を計算（抽象化版）
    
    Args:
        state: ゲームの状態
        move_func: 現在の状態から次の状態を生成する関数
        memo: Grundy数の記憶領域（オプション）
    
    Returns:
        状態のGrundy数
    """
    if memo is None:
        memo = {}
    
    if state in memo:
        return memo[state]
    
    next_states = move_func(state)
    
    # 可能な次の状態のGrundy数を計算
    grundy_values = set()
    for next_state in next_states:
        grundy_values.add(grundy_number(next_state, move_func, memo))
    
    # mexを計算
    memo[state] = mex(grundy_values)
    return memo[state]


def grundy_number_subtraction_game(n: int, subtraction_set: List[int], memo: Dict[int, int] = None) -> int:
    """
    Subtraction gameの特定の状態のGrundy数を計算
    
    Args:
        n: 現在の石の数
        subtraction_set: 取ることができる石の数の集合
        memo: 記憶領域（オプション）
    
    Returns:
        状態nのGrundy数
    """
    if memo is None:
        memo = {}
    
    if n in memo:
        return memo[n]
    
    # 石が0個の場合
    if n == 0:
        return 0
    
    # 可能な次の状態のGrundy数を集める
    next_grundy = set()
    for s in subtraction_set:
        if n >= s:
            next_grundy.add(grundy_number_subtraction_game(n - s, subtraction_set, memo))
    
    # mexを計算
    memo[n] = mex(next_grundy)
    return memo[n]


def compute_nim_equivalent(n: int, subtraction_set: List[int]) -> List[int]:
    """
    0からnまでの各状態に対するNim等価性を計算
    
    Args:
        n: 最大の状態
        subtraction_set: 取ることができる石の数の集合
    
    Returns:
        各状態のGrundy数のリスト
    """
    grundy = [0] * (n + 1)
    
    for i in range(1, n + 1):
        next_grundy = set()
        for s in subtraction_set:
            if i >= s:
                next_grundy.add(grundy[i - s])
        grundy[i] = mex(next_grundy)
    
    return grundy


def grundy_number_game_of_nim(piles: List[int]) -> int:
    """
    Nimゲームの状態のGrundy数を計算（Nimゲームの場合はNim和に等しい）
    
    Args:
        piles: 各山の石の数のリスト
    
    Returns:
        Nimゲームの状態のGrundy数（Nim和）
    """
    return nim_sum(piles)


def grundy_number_take_break_game(piles: List[int], memo: Dict[Tuple[int, ...], int] = None) -> int:
    """
    Take-and-Break gameのGrundy数を計算
    １つの山から石を取り、残った石を2つの山に分けることができるゲーム
    
    Args:
        piles: 各山の石の数のリスト
        memo: 記憶領域（オプション）
    
    Returns:
        Take-and-Break gameの状態のGrundy数
    """
    if memo is None:
        memo = {}
    
    state = tuple(sorted(piles))
    
    if state in memo:
        return memo[state]
    
    # 空の状態の場合
    if not state or all(pile == 0 for pile in state):
        return 0
    
    # 可能な次の状態のGrundy数を集める
    next_grundy = set()
    
    # 各山について試す
    for i, pile in enumerate(state):
        if pile > 0:
            # 石を全て取る場合
            new_piles = list(state[:i] + state[i+1:])
            next_grundy.add(grundy_number_take_break_game(new_piles, memo))
            
            # 石を一部取って分割する場合
            for j in range(1, (pile + 1) // 2):
                new_piles = list(state[:i] + state[i+1:]) + [j, pile - j]
                next_grundy.add(grundy_number_take_break_game(new_piles, memo))
    
    # mexを計算
    memo[state] = mex(next_grundy)
    return memo[state]


def moore_nim_bouton(n: int, k: int) -> bool:
    """
    Moore's Nim-Boutonゲームの勝敗判定
    n個の石があり、各ターンで1からk個の石を取れるゲーム
    
    Args:
        n: 石の数
        k: 1回に取れる最大の石の数
    
    Returns:
        現在の手番のプレイヤーが必勝ならTrue、そうでなければFalse
    """
    # nが(k+1)で割り切れなければ先手必勝
    return n % (k + 1) != 0


def game_of_divisors(n: int) -> bool:
    """
    約数ゲームの勝敗判定
    n個の石があり、各ターンでnの約数（n自身を除く）だけ石を取れるゲーム
    
    Args:
        n: 石の数
    
    Returns:
        現在の手番のプレイヤーが必勝ならTrue、そうでなければFalse
    """
    return n % 2 == 0


def wythoff_game(a: int, b: int) -> bool:
    """
    Wythoffゲームの勝敗判定
    2つの山の石があり、一方の山から好きなだけ取るか、両方の山から同じ数だけ取れるゲーム
    
    Args:
        a: 1つ目の山の石の数
        b: 2つ目の山の石の数（a ≤ b）
    
    Returns:
        現在の手番のプレイヤーが必勝ならTrue、そうでなければFalse
    """
    if a > b:
        a, b = b, a
    
    phi = (1 + 5 ** 0.5) / 2  # 黄金比
    
    # Beatty sequenceの計算
    n = int((b - a) * phi)
    m = int(n * phi)
    
    # 勝敗の判定
    return a != n or b != m


def fisher_game(piles: List[int]) -> bool:
    """
    Fisherゲーム（Nimの変種）の勝敗判定
    石を取るとき、必ず前回と異なる山から取らなければならない
    初回は任意の山から取れる
    
    Args:
        piles: 各山の石の数のリスト
    
    Returns:
        現在の手番のプレイヤーが必勝ならTrue、そうでなければFalse
    """
    if len(piles) <= 1:
        return piles[0] > 0 if piles else False
    
    # 1山だけ石がある場合は石があれば勝ち
    if sum(1 for pile in piles if pile > 0) == 1:
        return True
    
    # 2山以上石がある場合はNimと同じ
    return nim_sum(piles) != 0


@functools.lru_cache(maxsize=None)
def sprague_grundy_subtraction_game(n: int, allowed: Tuple[int, ...]) -> int:
    """
    Subtraction gameのGrundy数を計算（キャッシュ付き）
    
    Args:
        n: 石の数
        allowed: 取ることができる石の数のタプル
    
    Returns:
        Grundy数
    """
    if n == 0:
        return 0
    
    next_grundy = set()
    for a in allowed:
        if n >= a:
            next_grundy.add(sprague_grundy_subtraction_game(n - a, allowed))
    
    return mex(next_grundy)


def is_p_position(piles: List[int], allowed: Tuple[int, ...]) -> bool:
    """
    複数のSubtraction gameを組み合わせた合併ゲームのP-position判定
    （Sprague-Grundy定理を用いて）
    
    Args:
        piles: 各ゲームの状態（石の数）
        allowed: 取ることができる石の数のタプル
    
    Returns:
        P-positionならTrue（後手必勝）、そうでなければFalse（先手必勝）
    """
    grundy_sum = 0
    for pile in piles:
        grundy_sum ^= sprague_grundy_subtraction_game(pile, allowed)
    
    return grundy_sum == 0


def grundy_misere_nim(piles: List[int]) -> bool:
    """
    反転Nim（Misère Nim）ゲームの勝敗判定
    通常のNimと同じルールだが、最後の石を取ったプレイヤーが負け
    
    Args:
        piles: 各山の石の数のリスト
    
    Returns:
        現在の手番のプレイヤーが必勝ならTrue、そうでなければFalse
    """
    # 全ての山が1個以下かチェック
    all_ones_or_zero = all(pile <= 1 for pile in piles)
    
    if all_ones_or_zero:
        # 1の山の数が奇数なら後手必勝
        ones_count = sum(1 for pile in piles if pile == 1)
        return ones_count % 2 == 0
    else:
        # 通常のNimと同じ
        return nim_sum(piles) != 0


def optimal_move_nim(piles: List[int]) -> Tuple[int, int]:
    """
    Nimゲームの最適な手を返す
    
    Args:
        piles: 各山の石の数のリスト
    
    Returns:
        (山のインデックス, 取る石の数)の組
    """
    move = nim_winning_move(piles)
    if move:
        return move
    
    # 勝ち手がない場合は任意の手（石がある最初の山から1つ取る）
    for i, pile in enumerate(piles):
        if pile > 0:
            return (i, 1)
    
    return (0, 0)  # 石がない場合


def staircase_nim(piles: List[int]) -> bool:
    """
    Staircase Nimゲームの勝敗判定
    石は階段状に並んでおり、各ターンでは1つの段から任意の数の石を取り、
    それを同じ段または下の段に移動できる
    
    Args:
        piles: 各段の石の数のリスト（下から上へ）
    
    Returns:
        現在の手番のプレイヤーが必勝ならTrue、そうでなければFalse
    """
    # 奇数番目の段の石の数のXOR和
    odd_sum = 0
    for i in range(0, len(piles), 2):
        odd_sum ^= piles[i]
    
    return odd_sum != 0


def grundy_sum_game(n: int) -> int:
    """
    Sum gameのGrundy数を計算
    プレイヤーは交互に値を選び、合計がnに達したら負け
    
    Args:
        n: 目標の合計値
    
    Returns:
        Grundy数
    """
    if n == 0:
        return 0
    
    # ニム等価性のパターン（0, 1, 0, 2）を利用
    return [0, 1, 0, 2][n % 4]


# 使用例
def example():
    # 通常のNimゲーム
    piles = [3, 4, 5]
    print(f"山の状態: {piles}")
    print(f"Nim和: {nim_sum(piles)}")
    print(f"先手必勝か？: {is_nim_winning_position(piles)}")
    
    winning_move = nim_winning_move(piles)
    if winning_move:
        i, count = winning_move
        print(f"勝つための手: 山{i+1}から{count}個取る")
    else:
        print("勝つ手はありません")
    
    # Subtraction game
    n = 20
    subtraction_set = [1, 3, 4]
    print(f"\n石の数: {n}, 取れる石の数: {subtraction_set}")
    
    grundy = compute_nim_equivalent(n, subtraction_set)
    print(f"0から{n}までのGrundy数: {grundy}")
    
    # Take-and-Break game
    piles_tb = [7]
    g = grundy_number_take_break_game(piles_tb)
    print(f"\nTake-and-Break game, 山の状態: {piles_tb}")
    print(f"Grundy数: {g}")
    print(f"先手必勝か？: {g != 0}")
    
    # Wythoff game
    a, b = 3, 5
    print(f"\nWythoff game, 山の状態: {a}, {b}")
    print(f"先手必勝か？: {wythoff_game(a, b)}")
    
    # Moore's Nim-Bouton
    n, k = 13, 3
    print(f"\nMoore's Nim-Bouton, 石の数: {n}, 最大取得数: {k}")
    print(f"先手必勝か？: {moore_nim_bouton(n, k)}")
    
    # Misère Nim
    piles_misere = [1, 2, 3]
    print(f"\nMisère Nim, 山の状態: {piles_misere}")
    print(f"先手必勝か？: {grundy_misere_nim(piles_misere)}")
    
    # Staircase Nim
    piles_stair = [1, 2, 3, 4]
    print(f"\nStaircase Nim, 山の状態: {piles_stair}")
    print(f"先手必勝か？: {staircase_nim(piles_stair)}")


if __name__ == "__main__":
    example()
"""
行列演算アルゴリズム集 (Matrix Algorithms)
- 線形代数に関するアルゴリズム群
- 主な実装:
  - 基本的な行列演算（加減乗算、冪乗）
  - ガウス・ジョルダン消去法
  - 逆行列の計算
  - 行列式の計算
  - 連立一次方程式の解法
- 計算量:
  - 行列加減算: O(n²)
  - 行列乗算: O(n³)（ナイーブな方法）
  - ガウス消去法: O(n³)
  - 行列式計算: O(n³)
"""

from typing import List, Tuple, Optional, Union, TypeVar
import copy
import math

# mod付き演算用の型定義
T = TypeVar('T', int, float)


class Matrix:
    """行列を表すクラス"""
    
    def __init__(self, data: List[List[T]], mod: Optional[int] = None):
        """
        行列の初期化
        
        Args:
            data: 行列データ（二次元リスト）
            mod: 剰余を取る値（オプション）
        """
        self.data = copy.deepcopy(data)
        self.rows = len(data)
        self.cols = len(data[0]) if self.rows > 0 else 0
        self.mod = mod
    
    def __str__(self) -> str:
        """行列の文字列表現"""
        result = []
        for row in self.data:
            result.append(' '.join(map(str, row)))
        return '\n'.join(result)
    
    def __repr__(self) -> str:
        """行列のプログラム表現"""
        return f"Matrix({self.data}, mod={self.mod})"
    
    def __add__(self, other: 'Matrix') -> 'Matrix':
        """行列の加算"""
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("行列のサイズが一致しません")
        
        result = [[0 for _ in range(self.cols)] for _ in range(self.rows)]
        
        for i in range(self.rows):
            for j in range(self.cols):
                result[i][j] = self.data[i][j] + other.data[i][j]
                if self.mod:
                    result[i][j] %= self.mod
        
        return Matrix(result, self.mod)
    
    def __sub__(self, other: 'Matrix') -> 'Matrix':
        """行列の減算"""
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("行列のサイズが一致しません")
        
        result = [[0 for _ in range(self.cols)] for _ in range(self.rows)]
        
        for i in range(self.rows):
            for j in range(self.cols):
                result[i][j] = self.data[i][j] - other.data[i][j]
                if self.mod:
                    result[i][j] %= self.mod
        
        return Matrix(result, self.mod)
    
    def __mul__(self, other: Union['Matrix', int, float]) -> 'Matrix':
        """
        行列の乗算
        行列 * 行列 または 行列 * スカラー
        """
        if isinstance(other, (int, float)):
            # スカラー乗算
            result = [[0 for _ in range(self.cols)] for _ in range(self.rows)]
            for i in range(self.rows):
                for j in range(self.cols):
                    result[i][j] = self.data[i][j] * other
                    if self.mod:
                        result[i][j] %= self.mod
            return Matrix(result, self.mod)
        
        # 行列乗算
        if self.cols != other.rows:
            raise ValueError("行列の次元が合いません")
        
        result = [[0 for _ in range(other.cols)] for _ in range(self.rows)]
        
        for i in range(self.rows):
            for j in range(other.cols):
                for k in range(self.cols):
                    result[i][j] += self.data[i][k] * other.data[k][j]
                if self.mod:
                    result[i][j] %= self.mod
        
        return Matrix(result, self.mod)
    
    def transpose(self) -> 'Matrix':
        """転置行列の計算"""
        result = [[0 for _ in range(self.rows)] for _ in range(self.cols)]
        
        for i in range(self.rows):
            for j in range(self.cols):
                result[j][i] = self.data[i][j]
        
        return Matrix(result, self.mod)
    
    def determinant(self) -> T:
        """
        行列式の計算
        
        Returns:
            行列式の値
        """
        if self.rows != self.cols:
            raise ValueError("正方行列でなければ行列式は計算できません")
        
        n = self.rows
        
        # 1x1または2x2行列は直接計算
        if n == 1:
            return self.data[0][0]
        elif n == 2:
            det = self.data[0][0] * self.data[1][1] - self.data[0][1] * self.data[1][0]
            if self.mod:
                det %= self.mod
            return det
        
        # ガウス消去法による計算（O(n³)）
        temp = copy.deepcopy(self.data)
        det = 1
        
        for i in range(n):
            # ピボット選択（部分ピボット選択）
            max_row = i
            for j in range(i + 1, n):
                if abs(temp[j][i]) > abs(temp[max_row][i]):
                    max_row = j
            
            # ピボットが0ならば行列式は0
            if abs(temp[max_row][i]) < 1e-10:
                return 0
            
            # 行の交換
            if i != max_row:
                temp[i], temp[max_row] = temp[max_row], temp[i]
                det = -det  # 行交換で符号反転
            
            # 対角成分を使って消去
            pivot = temp[i][i]
            det *= pivot
            
            for j in range(i + 1, n):
                factor = temp[j][i] / pivot
                for k in range(i, n):
                    temp[j][k] -= factor * temp[i][k]
        
        if self.mod:
            det %= self.mod
        
        return det
    
    def pow(self, exponent: int) -> 'Matrix':
        """
        行列のべき乗 A^n
        
        Args:
            exponent: 指数（非負整数）
        
        Returns:
            A^n
        """
        if exponent < 0:
            raise ValueError("負の指数はサポートされていません")
        
        if self.rows != self.cols:
            raise ValueError("正方行列でなければべき乗は計算できません")
        
        n = self.rows
        
        # 単位行列
        result = identity_matrix(n, self.mod)
        
        # 繰り返し二乗法
        base = copy.deepcopy(self)
        while exponent > 0:
            if exponent & 1:
                result = result * base
            base = base * base
            exponent >>= 1
        
        return result
    
    def inverse(self) -> 'Matrix':
        """
        逆行列の計算
        
        Returns:
            逆行列
        """
        if self.rows != self.cols:
            raise ValueError("正方行列でなければ逆行列は計算できません")
        
        n = self.rows
        aug = [row[:] + [0] * n for row in self.data]
        
        # 拡大行列の右側を単位行列にする
        for i in range(n):
            aug[i][n + i] = 1
        
        # ガウス・ジョルダン消去法
        for i in range(n):
            # ピボット選択
            max_row = i
            for j in range(i + 1, n):
                if abs(aug[j][i]) > abs(aug[max_row][i]):
                    max_row = j
            
            # ピボットが0ならば逆行列は存在しない
            if abs(aug[max_row][i]) < 1e-10:
                raise ValueError("行列は特異です（逆行列が存在しません）")
            
            # 行の交換
            if i != max_row:
                aug[i], aug[max_row] = aug[max_row], aug[i]
            
            # 対角成分を1にする
            pivot = aug[i][i]
            for j in range(i, 2 * n):
                aug[i][j] /= pivot
                if self.mod:
                    aug[i][j] %= self.mod
            
            # ピボット列の他の行を0にする
            for j in range(n):
                if j != i:
                    factor = aug[j][i]
                    for k in range(i, 2 * n):
                        aug[j][k] -= factor * aug[i][k]
                        if self.mod:
                            aug[j][k] %= self.mod
        
        # 右半分（逆行列）を抽出
        result = [[aug[i][j + n] for j in range(n)] for i in range(n)]
        
        return Matrix(result, self.mod)
    
    def gaussian_elimination(self, b: List[T]) -> List[T]:
        """
        ガウスの消去法による連立一次方程式 Ax = b の解
        
        Args:
            b: 右辺ベクトル
        
        Returns:
            解ベクトル x
        """
        if len(b) != self.rows:
            raise ValueError("右辺ベクトルのサイズが合いません")
        
        n = self.rows
        
        # 拡大係数行列を作成
        aug = [row[:] + [b[i]] for i, row in enumerate(self.data)]
        
        # 前進消去
        for i in range(n):
            # ピボット選択
            max_row = i
            for j in range(i + 1, n):
                if abs(aug[j][i]) > abs(aug[max_row][i]):
                    max_row = j
            
            # 行の交換
            if i != max_row:
                aug[i], aug[max_row] = aug[max_row], aug[i]
            
            # ピボットが0ならば解なしまたは不定
            if abs(aug[i][i]) < 1e-10:
                # 右辺も0なら不定、そうでなければ解なし
                if abs(aug[i][n]) < 1e-10:
                    continue  # 不定
                else:
                    raise ValueError("解が存在しません")
            
            # ピボット以下の行を消去
            for j in range(i + 1, n):
                factor = aug[j][i] / aug[i][i]
                aug[j][i] = 0  # 厳密に0にする
                for k in range(i + 1, n + 1):
                    aug[j][k] -= factor * aug[i][k]
        
        # 後退代入
        x = [0] * n
        for i in range(n - 1, -1, -1):
            if abs(aug[i][i]) < 1e-10:
                continue  # 自由変数
            
            x[i] = aug[i][n]
            for j in range(i + 1, n):
                x[i] -= aug[i][j] * x[j]
            x[i] /= aug[i][i]
            if self.mod:
                x[i] %= self.mod
        
        return x
    
    def rank(self) -> int:
        """
        行列の階数を計算
        
        Returns:
            階数
        """
        # 行簡約形に変換
        temp = copy.deepcopy(self.data)
        n, m = self.rows, self.cols
        
        r = 0  # 階数
        for c in range(m):
            # ピボット行を見つける
            pivot_row = None
            for i in range(r, n):
                if abs(temp[i][c]) > 1e-10:
                    pivot_row = i
                    break
            
            if pivot_row is not None:
                # 行を交換
                if r != pivot_row:
                    temp[r], temp[pivot_row] = temp[pivot_row], temp[r]
                
                # ピボット以下の行を消去
                for i in range(r + 1, n):
                    factor = temp[i][c] / temp[r][c]
                    temp[i][c] = 0  # 厳密に0にする
                    for j in range(c + 1, m):
                        temp[i][j] -= factor * temp[r][j]
                
                r += 1  # 階数を増やす
            
            if r == n:  # 全行が独立
                break
        
        return r
    
    def trace(self) -> T:
        """
        行列のトレース（対角成分の和）
        
        Returns:
            トレース
        """
        if self.rows != self.cols:
            raise ValueError("正方行列でなければトレースは計算できません")
        
        trace_val = sum(self.data[i][i] for i in range(self.rows))
        if self.mod:
            trace_val %= self.mod
        
        return trace_val


def identity_matrix(n: int, mod: Optional[int] = None) -> Matrix:
    """
    n×n単位行列を作成
    
    Args:
        n: 行列のサイズ
        mod: 剰余を取る値（オプション）
    
    Returns:
        単位行列
    """
    data = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        data[i][i] = 1
    
    return Matrix(data, mod)


def solve_linear_system(A: List[List[T]], b: List[T], mod: Optional[int] = None) -> List[T]:
    """
    連立一次方程式 Ax = b を解く
    
    Args:
        A: 係数行列
        b: 右辺ベクトル
        mod: 剰余を取る値（オプション）
    
    Returns:
        解ベクトル x
    """
    matrix = Matrix(A, mod)
    return matrix.gaussian_elimination(b)


def linear_recurrence_relation(initial: List[int], coef: List[int], n: int, mod: Optional[int] = None) -> int:
    """
    線形漸化式の第n項を行列累乗で計算
    例: F(n) = a*F(n-1) + b*F(n-2) + ... + z*F(n-k)
    
    Args:
        initial: 初期値 [F(0), F(1), ..., F(k-1)]
        coef: 係数 [a, b, ..., z]
        n: 求める項
        mod: 剰余を取る値（オプション）
    
    Returns:
        F(n)の値
    """
    k = len(initial)
    if len(coef) != k:
        raise ValueError("係数と初期値の数が一致しません")
    
    # n < k の場合は直接初期値を返す
    if n < k:
        return initial[n]
    
    # 遷移行列を作成
    # [a b c ...] [F(n-1)]   [F(n)  ]
    # [1 0 0 ...] [F(n-2)]   [F(n-1)]
    # [0 1 0 ...] [F(n-3)] = [F(n-2)]
    # [... ... .] [  ...  ]   [  ...  ]
    transition = [[0 for _ in range(k)] for _ in range(k)]
    
    # 最初の行に係数を設定
    for i in range(k):
        transition[0][i] = coef[i]
    
    # 他の行に単位行列の一部を設定
    for i in range(1, k):
        transition[i][i-1] = 1
    
    # 行列をn-k+1回乗算
    matrix = Matrix(transition, mod).pow(n - k + 1)
    
    # 結果を計算
    result = 0
    for i in range(k):
        result += matrix.data[0][i] * initial[k - 1 - i]
        if mod:
            result %= mod
    
    return result


def characteristic_polynomial(A: Matrix) -> List[T]:
    """
    行列の特性多項式の係数を計算（レバリエ・ファドゥーブの方法）
    
    Args:
        A: 係数行列
    
    Returns:
        特性多項式の係数 [a_n, a_(n-1), ..., a_1, a_0]
        ただし、多項式は a_n * x^n + a_(n-1) * x^(n-1) + ... + a_1 * x + a_0
    """
    if A.rows != A.cols:
        raise ValueError("正方行列でなければ特性多項式は計算できません")
    
    n = A.rows
    poly = [0] * (n + 1)
    poly[0] = 1  # 最高次数の係数
    
    # B_k = A^k を計算
    B = identity_matrix(n, A.mod)
    
    for k in range(1, n + 1):
        B = B * A
        poly[k] = -B.trace() / k
        
        # 前の係数を使って更新
        for j in range(1, k):
            poly[k] -= poly[j] * B.data[j - 1][j - 1] / k
    
    return poly


def eigenvalues(A: Matrix) -> List[complex]:
    """
    行列の固有値を計算（特性方程式を解く）
    注意: 近似解で、実用性に制限あり
    
    Args:
        A: 係数行列
    
    Returns:
        固有値のリスト
    """
    if A.rows != A.cols:
        raise ValueError("正方行列でなければ固有値は計算できません")
    
    # 特性多項式の係数を計算
    poly = characteristic_polynomial(A)
    
    # ここでは単純なために numpy の roots 関数を使用する想定
    # 実際のコードでは numpy をインポートして以下を使用
    # roots = np.roots(poly)
    # return roots
    
    # numpy がない場合、固有値計算は複雑なので空リストを返す
    return []


# 使用例
def example():
    # 行列の作成と基本演算
    A = Matrix([[1, 2], [3, 4]])
    B = Matrix([[5, 6], [7, 8]])
    
    print("行列A:")
    print(A)
    print("\n行列B:")
    print(B)
    
    print("\nA + B:")
    print(A + B)
    
    print("\nA - B:")
    print(A - B)
    
    print("\nA * B:")
    print(A * B)
    
    print("\nA^2:")
    print(A.pow(2))
    
    print("\nAの転置:")
    print(A.transpose())
    
    print("\nAの行列式:")
    print(A.determinant())
    
    try:
        print("\nAの逆行列:")
        print(A.inverse())
    except ValueError as e:
        print(f"エラー: {e}")
    
    # 連立一次方程式を解く
    A_system = [[2, 1, -1], 
                [-3, -1, 2], 
                [-2, 1, 2]]
    b = [8, -11, -3]
    
    print("\n連立方程式の解:")
    try:
        solution = Matrix(A_system).gaussian_elimination(b)
        print(solution)
    except ValueError as e:
        print(f"エラー: {e}")
    
    # mod付き行列演算
    mod = 1000000007
    A_mod = Matrix([[1, 2], [3, 4]], mod)
    B_mod = Matrix([[5, 6], [7, 8]], mod)
    
    print("\nmod付き行列乗算:")
    print(A_mod * B_mod)
    
    # 線形漸化式（フィボナッチ）の計算
    initial = [0, 1]  # F(0) = 0, F(1) = 1
    coef = [1, 1]     # F(n) = F(n-1) + F(n-2)
    n = 10
    
    print(f"\nフィボナッチ数列の第{n}項:")
    print(linear_recurrence_relation(initial, coef, n))


if __name__ == "__main__":
    example()
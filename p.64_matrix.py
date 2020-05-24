
# p.64_matrix
#A[i][j]는 i번째 행과j번째열에 속한 숫자를 의미한다. 
from typing import List
Matrix = List[List[float]]

A = [[1, 2, 3],  # A는 2행3열
     [4, 5, 6]]

B = [[1, 2],     # B 는 3행2열
     [3, 4],
     [5, 6]]
#파이썬에서는 첫번째 행을 행0 이라하고 첫번째 열도 열0 이라고 한다.      
from typing import Tuple

def shape(A: Matrix) -> Tuple[int, int]:
    """(열의 갯수, 행의개수)를 반환"""
    num_rows = len(A)
    num_cols = len(A[0]) if A else 0   # number of elements in first row
    return num_rows, num_cols

assert shape([[1, 2, 3], [4, 5, 6]]) == (2, 3)  # 2 rows, 3 columns

def get_row(A: Matrix, i: int) -> Vector:
    """A의 i번째 행을 반환"""
    return A[i]             # A[i] is already the ith row

def get_column(A: Matrix, j: int) -> Vector:
    """A의 j번째 열을 반환"""
    return [A_i[j]          # A_i의 행의 j번째 원소
            for A_i in A]   # 각A_i 행에 대해

from typing import Callable

def make_matrix(num_rows: int,
                num_cols: int,
                entry_fn: Callable[[int, int], float]) -> Matrix:
    
    return [[entry_fn(i, j)             # i 가 주어졌을때 리스트를 생성한다. 
             for j in range(num_cols)]  # [entry_fn(i, 0), ... ]
            for i in range(num_rows)]   # 각 i에 대해 하나의 리스트를 생성한다. 

def identity_matrix(n: int) -> Matrix:
    """ n x n 단위의 행렬을 반환"""
    return make_matrix(n, n, lambda i, j: 1 if i == j else 0)

assert identity_matrix(5) == [[1, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0],
                              [0, 0, 1, 0, 0],
                              [0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 1]]

data = [[70, 170, 40],
        [65, 120, 26],
        [77, 250, 19],
        # ....
       ]

friendships = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 4),
               (4, 5), (5, 6), (5, 7), (6, 8), (7, 8), (8, 9)]

#            사용자 0  1  2  3  4  5  6  7  8  9
#
friend_matrix = [[0, 1, 1, 0, 0, 0, 0, 0, 0, 0],  # user 0
                 [1, 0, 1, 1, 0, 0, 0, 0, 0, 0],  # user 1
                 [1, 1, 0, 1, 0, 0, 0, 0, 0, 0],  # user 2
                 [0, 1, 1, 0, 1, 0, 0, 0, 0, 0],  # user 3
                 [0, 0, 0, 1, 0, 1, 0, 0, 0, 0],  # user 4
                 [0, 0, 0, 0, 1, 0, 1, 1, 0, 0],  # user 5
                 [0, 0, 0, 0, 0, 1, 0, 0, 1, 0],  # user 6
                 [0, 0, 0, 0, 0, 1, 0, 0, 1, 0],  # user 7
                 [0, 0, 0, 0, 0, 0, 1, 1, 0, 1],  # user 8
                 [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]]  # user 9

assert friend_matrix[0][2] == 1, "0과 2는 친구이다, Ture"
assert friend_matrix[0][8] == 0, "0과 8은 친구가 아니다. False"

friends_of_five = [i
                   for i, is_friend in enumerate(friend_matrix[5])
                   if is_friend]
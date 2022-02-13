from locale import DAY_1
import numpy as np

x_dot_w = np.array([[0, 0, 0], [10, 10, 10]])
b = np.array([1, 2, 3])

print(x_dot_w)
print(x_dot_w + b)

dy = np.array([[1, 2, 3], [4, 5, 6]])
print(dy)

db = np.sum(dy, axis=0)
print(db)

# [[ 0  0  0]
#  [10 10 10]]
# [[ 1  2  3]
#  [11 12 13]]
# [[1 2 3]
#  [4 5 6]]
# [5 7 9]
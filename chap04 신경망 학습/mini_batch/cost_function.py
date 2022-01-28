import numpy as np

def sum_squares_error(y, t):
    return 0.5 * np.sum((y-t)** 2)

def cross_entropy_error(y, t):
    delta = 1e-7 # log 0 계산 방지를 위함
    return -np.sum(t * np.log(y + delta))

t = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
y1 = np.array([0.1, 0.05, 0.6, 0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0])
y2 = np.array([0.1,0.05,0.1,0,0.05,0.1,0,0.6,0,0])

print(sum_squares_error(y1, t))
print(sum_squares_error(y2, t))

print()

print(cross_entropy_error(y1, t))
print(cross_entropy_error(y2, t))

## result ##
# 0.09750000000000003
# 0.5975

# 0.510825457099338
# 2.302584092994546
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def main():
    # 0층 구성
    x = np.array([1.0, 0.5])
    w1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    b1 = np.array([0.1, 0.2, 0.3])

    # 0층 -> 1층 계산
    a1 = np.dot(x, w1) + b1
    # activation function : sigmoid
    z = sigmoid(a1)

    # 1층 구성
    w2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    b2 = np.array([0.1, 0.2])

    # 1층 -> 2층 계산
    a2 = np.dot(z, w2) + b2
    z2 = sigmoid(a2)

    # 2층 구성
    w3 = np.array([[0.1, 0.3], [0.2, 0.4]])
    b3 = np.array([0.1, 0.2])

    # 2층 -> 3층 출력층 계산
    a3 = np.dot(z2, w3) + b3
    y = a3

    print(y)

if __name__ == '__main__':
    main()
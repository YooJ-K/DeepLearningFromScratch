import numpy as np

def array_shape(x):
    print(x, np.ndim(x), x.shape)

def matrix_product(x, y):
    try:
        print(np.dot(x, y))
    except:
        print("곱이 불가능합니다.")

def main():
    # 1차원 배열
    A = np.array([1, 2, 3, 4])
    array_shape(A)

    # 2차원 배열
    B = np.array([[1, 2], [3, 4], [5, 6]])
    array_shape(B)

    # 행렬의 곱 (2, 2) x (2, 2)
    C = np.array([[1, 2], [3, 4]])
    D = np.array([[5, 6], [7, 8]])
    matrix_product(C, D)

    # 행렬의 곱 (2, 3) x (3, 2)
    E = np.array([[1, 2, 3], [4, 5, 6]])
    F = np.array([[1, 2], [3, 4], [5, 6]])
    matrix_product(E, F)

    # 행렬의 곱 (3, 2) x (2)
    G = np.array([[1, 2], [3, 4], [5, 6]])
    H = np.array([7, 8])
    matrix_product(G, H)

if __name__ == '__main__':
    main()
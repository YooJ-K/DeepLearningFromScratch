import numpy as np

def softmax(a):
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y

    # 이 함수의 문제점 : 오버플로우 확률이 매우 높다.

def new_softmax(a):
    c = np.max(a)
    return np.exp(a - c)/np.sum(np.exp(a - c))

def main():
    a = np.array([0.3, 2.9, 4.0])
    print(softmax(a))

    b = np.array([1010, 1000, 990])
    print(new_softmax(b))

if __name__ == '__main__':
    main()
#-*-coding: utf-8-*-
import numpy as np

def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]

        x[idx] = tmp_val + h
        fxh1 = f(x)

        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val
    
    return grad

def gradient_descent(f, init_x, lr=0.01, step_num=100):
    # f 최적화하려는 함수
    # step_num 경사법에 다른 반복 횟수
    
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad
    return x

def func(x):
    return x[0]**2 + x[1]**2

def main():
    init_x = np.array([-3.0, 4.0])
    result = gradient_descent(func, init_x=init_x, lr=0.1, step_num=100)
    print(result)

    lr_high = gradient_descent(func, init_x=init_x, lr=10.0, step_num=100)
    lr_low = gradient_descent(func, init_x=init_x, lr=1e-10, step_num=100)

    print(lr_high, lr_low)

if __name__=='__main__':
    main()

# result
# [ -6.11110793e-10   8.14814391e-10] => 0에 가까운 값이다.
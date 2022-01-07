import numpy as np

def step_function_1(x):
    if x > 0:
        return 1
    else:
        return 0

def step_function_2(x):
    y = x > 0
    return y.astype(np.bool_)
    # return y.astype(np.int_)

def main():
    x = np.array([-1.0, 1.0, 2.0])
    print(step_function_2(x))

if __name__=='__main__':
    main()
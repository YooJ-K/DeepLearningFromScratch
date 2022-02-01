#-*-coding: utf-8-*-

import os, sys
os.chdir(r'/Users/kimyoojin/Desktop/github/DeepLearningFromScratch/chap04 신경망 학습/common')
sys.path.append(os.getcwd())

import numpy as np
from functions import softmax, cross_entropy_error
from gradient import numerical_gradient

class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)
    
    def predict(self, x):
        return np.dot(x, self.W)
    
    def loss(self, x, t):
        # 손실 함수의 값을 구하는. x : input, t : correct answer lable
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss

def main():
    net = simpleNet()
    print(net.W) # random하게 2, 3의 형태로 matrix생성한다.

    x = np.array([0.6, 0.9])
    p = net.predict(x)
    t = np.array([0, 0, 1]) # 정답 레이블
    
    print(np.argmax(p)) # predict
    print(net.loss(x, t))

    f = lambda w: net.loss(x, t)

    dW = numerical_gradient(f, net.W)
    print(dW) # 신경망에서의 기울기. dL/dw_nn

if __name__ == '__main__':
    main()
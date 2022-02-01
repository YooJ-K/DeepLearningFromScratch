#-*-coding: utf-8-*-
# 2층 신경망을 하나의 클래스로 구현

import sys, os
os.chdir(r'/Users/kimyoojin/Desktop/github/DeepLearningFromScratch/chap04 신경망 학습/train_algorithm/common')
sys.path.append(os.getcwd())

import numpy as np

from functions import *
from gradient import numerical_gradient

class TwoLayerNet:
    # 입력 층 뉴런 수, 은닉층 뉴런 수, 출력층 뉴런 수
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):

        # params : 신경망의 매개변수를 보관하는 딕셔너리 변수
        self.params = {}

        # 가중치 매개변수를 초기화한다. => 신경망 학습의 성공을 좌우한다.
        # bias : 0으로 모두 초기화한다.
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    # 예측, x = 이미지 데이터
    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        return y
    
    # x : 이미지 데이터, t  정답 레이블
    def loss(self, x, t):
        y = self.predict(x)

        return cross_entropy_error(y, t)
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y - np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
    
    # 가중치 매개변수의 기울기를 구한다.
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)

        # grads : 기울기 보관하는 딕셔너리 변수
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads

def about_params_test():
    net = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)
    
    for p in ['W1', 'b1', 'W2', 'b2']:
        print(net.params[p].shape)
    
        # (784, 100)
        # (100,)
        # (100, 10)
        # (10,)

def about_grads_test():
    net = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)

    x = np.random.rand(100, 784) # 더미 입력 데이터 100장 분량
    t = np.random.rand(100, 10)  # 더미 정답 레이블 100장 분량

    grads = net.numerical_gradient(x, t) # 기울기 계산

    for p in ['W1', 'b1', 'W2', 'b2']:
        print(grads[p].shape)

        # (784, 100)
        # (100,)
        # (100, 10)
        # (10,)

#about_params_test()
about_grads_test()
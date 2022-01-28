import sys, os
sys.path.append(os.pardir)
import numpy as np
from mnist import load_mnist

# t_train, t_test : 레이블 1~10
(x_train, t_train), (x_test, t_test) = \
    load_mnist(normalize=True, one_hot_label=True)

print(x_train.shape)
print(t_train.shape)

## result ##
# (60000, 784)
# (60000, 10)
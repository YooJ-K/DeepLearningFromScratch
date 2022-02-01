import numpy as np

def softmax(a):
    c = np.max(a)
    return np.exp(a - c)/np.sum(np.exp(a - c))

def cross_entropy_error(y, t, one_hot_label=True):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]

    if one_hot_label:
        return -np.sum(t * np.log(y + 1e-7)) / batch_size
    else:
        return -np.sum(np.log(y[np.arrange(batch_size), t] + 1e-7)) / batch_size

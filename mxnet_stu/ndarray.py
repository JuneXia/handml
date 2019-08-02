import mxnet as mx
import numpy as np


arr = mx.nd.ones((2, 2))

b = mx.nd.arange(4).reshape((2, 2))

c = mx.nd.dot(arr, b)
print(c)

c1 = mx.nd.sum_axis(c, axis=1)


a = mx.nd.array(np.arange(6).reshape(6, 1))
b = a.broadcast_to((6, 4))


gpu_device = mx.gpu()

def fun():
    a = mx.nd.ones((5, 5))
    b = mx.nd.ones((5, 5))
    c = a + b
    print(c)

fun()

with mx.Context(gpu_device):
    fun()


a = mx.nd.ones((5, 5), mx.cpu())
b = mx.nd.ones((5, 5), gpu_device)
c = mx.nd.ones((5, 5), gpu_device)

a.copyto(c)
d = b + c
e = b.as_in_context(c.context) + c



import time
def do(x, n):
    return [mx.nd.dot(x, x) for i in range(n)]

def wait(x):
    for y in x:
        y.wait_to_read()

a = mx.nd.ones((500, 500))
b = mx.nd.ones((500, 500), gpu_device)

tic = time.time()
c = do(a, 50)
wait(c)
print(time.time() - tic)

tic = time.time()
c = do(b, 50)
wait(c)
print(time.time() - tic)

print('debug')



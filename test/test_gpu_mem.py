#!/usr/bin/python
from __future__ import division
from __future__ import print_function

import torch

from utils.gpu_memory import gpu_memory_log

dtype = torch.float
N, D_in, H, D_out = 64, 1000, 100, 10

device = torch.device("cuda")
x = torch.randn(N, D_in, device=device, dtype=dtype)
y = torch.randn(N, D_out, device=device, dtype=dtype)

w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)

learning_rate = 1e-6
gpu_memory_log()
for t in range(500):
    y_pred = x.mm(w1).clamp(min=0).mm(w2)
    loss = (y_pred - y).pow(2).sum()
    # print(t, loss.item())
    loss.backward()
    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad

        w1.grad.zero_()
        w2.grad.zero_()

print("正常情况：")
print(torch.cuda.memory_summary(device, True))

print("empty_cache后：")
torch.cuda.empty_cache()
print(torch.cuda.memory_summary(device, True))

gpu_memory_log("logs/gpu_mem.log")

print("分配了一个新的(100,100,100)前：")
print(torch.cuda.memory_allocated())

a = torch.zeros(100,100,100).cuda()
print("分配了一个新的(100,100,100)后：")
print(torch.cuda.memory_allocated())

del a
torch.cuda.synchronize()
print("del 删除了(100,100,100)后：")
print(torch.cuda.memory_allocated())

# python -m test.test_gpu_mem
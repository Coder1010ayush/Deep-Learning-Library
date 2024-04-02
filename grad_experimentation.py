import torch
from torch import autograd as agd
from torchviz import make_dot

"""

tensor([[-2.1419,  2.4966],
        [-0.7591,  5.5725]], grad_fn=<AddBackward0>)
arr1 grad is  tensor([[-0.3492, -0.6953],
        [-0.3492, -0.6953]])

arr2 grad is  tensor([[-0.0540, -0.0540],
        [-0.0128, -0.0128]])

grad of arr3  tensor([[0.5538, 0.5531],
        [0.5538, 0.5531]])

grad of arr4  tensor([[-0.4063, -0.4063],
        [ 0.9405,  0.9405]])

tensor([[ 0.4169, -0.8218],
        [-0.6329,  0.7705]], requires_grad=True)
tensor([[ 0.1586, -1.5555],
        [-1.6373, -1.1440]], requires_grad=True)
tensor([[-1.4126,  0.8049],
        [-0.2125,  2.9572]], requires_grad=True)
tensor([[ 2.7441, -0.5289],
        [ 0.4010,  1.8115]], requires_grad=True)


"""
torch.seed()
arr1 = torch.randn(size=(4,4),requires_grad=True)
arr2 = torch.randn(size=(4,4),requires_grad=True)



arr3 = torch.randn(size=(4,4),requires_grad=True)
arr4 = torch.randn(size=(4,4),requires_grad=True)


arr5 = torch.randn(size=(3,3),requires_grad=True)
arr6 = torch.randn(size=(3,3),requires_grad=True)


out1 = torch.matmul(arr1,arr2)
out1.retain_grad()
out2 = torch.matmul(arr3,arr4)
out2.retain_grad()

out = torch.add(out1,out2)
out.retain_grad()
loss = out.mean()
loss.retain_grad()
loss.backward()
print('arr1 grad is ',arr1.grad)
print()
print('arr2 grad is ',arr2.grad)
print()
print('grad of arr3 ',arr3.grad)
print()
print('grad of arr4 ',arr4.grad)
print()
params = {
    "arr1": arr1,
    "arr2": arr2,
    "arr3": arr3,
    "arr4": arr4,
    "out1": out1,
    "out2": out2,
    "out": out,
    "loss": loss
}
print(arr1)
print(arr2)
print(arr3)
print(arr4)
print()
print()
print()

print('loss is ',loss)
print('loss grad is ',loss.grad)


print()
print('out is ', out)
print('out grad is ', out.grad)

print()
print('out1 is ', out1)
print('out1 grad is ', out1.grad)

print()
print('out2 is ', out2)
print('out2 grad is ', out2.grad)

result = torch.matmul(out2.grad , arr3)
print('reasult is ', result)
make_dot(out, params=params).render("img/pytorch_computational_graph", format="png")

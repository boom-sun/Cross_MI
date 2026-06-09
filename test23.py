import torch
import torchvision
print("torch:",torch.__version__)
print("torchvision:",torchvision.__version__)
print('GPU:',torch.cuda.is_available())#cuda是否可用
print(torch.cuda.device_count())#返回GPU的数量
print(torch.version.cuda)
print(torch.backends.cudnn.version())
x = torch.rand(5, 3)
print(x)
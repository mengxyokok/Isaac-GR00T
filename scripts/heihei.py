import torch
import time

# 假设你的GPU有10GB显存，20%大约是2GB
# 这里分配一个大约2GB的张量（float32: 4字节）
GB = 1024 ** 3
target_bytes = int(2 * GB)
num_elements = target_bytes // 4  # float32

# 分配张量到GPU
x = torch.zeros(num_elements, dtype=torch.float32, device='cuda')
print(f"Allocated tensor of shape {x.shape} on GPU, using about 2GB memory.")

# 一直运行
while True:
    # 做一些无意义的操作，防止脚本退出
    x += 1
    time.sleep(1)
import pandas as pd

import os

folder_path="datasets/isaacsim_libero_object_joint_2025_06_09_n1/data/chunk-000"
# 读取文件夹中的Parquet文件
files = [f for f in os.listdir(folder_path) if f.endswith('.parquet')]
for file in files:
    # 读取Parquet文件
    df = pd.read_parquet(os.path.join(folder_path, file))
    # 修改列名为新的名称
    df.columns = ['observation.state', 'action', 'timestamp', 'frame_index', 'episode_index',
       'index', 'task_index']
    # 保存修改后的文件
    df.to_parquet(os.path.join(folder_path, file))

# # 读取Parquet文件
# df = pd.read_parquet('datasets/isaacsim_libero_object_joint_2025_06_09_n1/data/chunk-000/episode_000000.parquet')
#   # 修改列名为新的名称
# df.columns = ['observation.state', 'action', 'timestamp', 'frame_index', 'episode_index',
#        'index', 'task_index']
# # 保存修改后的文件
# df.to_parquet('datasets/isaacsim_libero_object_joint_2025_06_09_n1/data/chunk-000/episode_000000.parquet')

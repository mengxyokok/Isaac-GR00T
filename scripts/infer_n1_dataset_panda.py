from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.model.policy import Gr00tPolicy
from gr00t.data.embodiment_tags import EmbodimentTag
import numpy as np

data_config = DATA_CONFIG_MAP["bimanual_panda_gripper"]
modality_config = data_config.modality_config()
transforms = data_config.transform()

dataset = LeRobotSingleDataset(
    # dataset_path="demo_data/robot_sim.PickNPlace",
    dataset_path="dataset/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim/bimanual_panda_gripper.Threading",
    modality_configs=modality_config,
    transforms=None, 
    embodiment_tag=EmbodimentTag.NEW_EMBODIMENT, 
)

input=dataset[0]
# print keys and shape of each value

for key in input.keys():
    print(f"{key}: {np.array(input[key]).shape}")


policy = Gr00tPolicy(
    model_path="nvidia/GR00T-N1-2B",
    modality_config=modality_config,
    modality_transform=transforms,
    embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
    device="cuda"
)

output = policy.get_action(input)
# print keys and shape of each modality
for key in output.keys():
    print(f"{key}: {np.array(output[key]).shape}")


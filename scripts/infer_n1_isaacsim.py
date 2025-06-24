from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy import Gr00tPolicy
from gr00t.data.embodiment_tags import EmbodimentTag
import numpy as np

data_config = DATA_CONFIG_MAP["gr1_arms_only"]
modality_config = data_config.modality_config()
transforms = data_config.transform()

input = {
    "video.ego_view": np.random.randint(0, 256, (1, 256, 256, 3), dtype=np.uint8),
    "state.left_arm": np.random.rand(1, 7),
    "state.right_arm": np.random.rand(1, 7),
    "state.left_hand": np.random.rand(1, 6),
    "state.right_hand": np.random.rand(1, 6),
    "state.waist": np.random.rand(1, 3),
    "annotation.human.action.task_description": ["do your thing!"],
}

# print keys and shape of each value

for key in input.keys():
    print(f"{key}: {np.array(input[key]).shape}")


policy = Gr00tPolicy(
    # model_path="nvidia/GR00T-N1-2B",
    model_path="checkpoints/checkpoint-8500",
    modality_config=modality_config,
    modality_transform=transforms,
    embodiment_tag=EmbodimentTag.GR1,
    device="cuda"
)

output = policy.get_action(input)
# print keys and shape of each modality
for key in output.keys():
    print(f"{key}: {np.array(output[key]).shape}")


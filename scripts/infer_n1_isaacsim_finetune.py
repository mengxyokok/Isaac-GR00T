from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})


import isaacsim.core.utils.numpy.rotations as rot_utils
import isaacsim.core.utils.prims as prim_utils
import isaacsim.core.utils.stage as stage_utils
from isaacsim.sensors.camera import Camera
from isaacsim.core.prims import Articulation, SingleArticulation
from isaacsim.core.utils.types import ArticulationActions, ArticulationAction
from isaacsim.core.api import World,SimulationContext

from pxr import Gf, Sdf, Usd

import omni
import omni.usd
import omni.kit.commands


from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy import Gr00tPolicy
from gr00t.data.embodiment_tags import EmbodimentTag

import numpy as np
import cv2
import time

stage: Usd.Stage = stage_utils.get_current_stage()

scene_usd_path="/mnt/mxy/n1/scripts/empty.usd"
scene_prim_path="/World/scene"
robot_usd_path="/mnt/mxy/humanoidrobotlab/source/humanLab/data/Robots/FFTAI/GR1T1_fourier_hand_6dof/GR1T1_fourier_hand_6dof.usd"
robot_prim_path="/World/robot"
robot_articulation_path=f"{robot_prim_path}/root_joint"


if not stage.ResolveIdentifierToEditTarget(scene_usd_path):
    raise FileNotFoundError(f"USD file not found at path: '{scene_usd_path}'.")

if not prim_utils.is_prim_path_valid(scene_prim_path):
    # add prim as reference to stage
    prim_utils.create_prim(
        scene_prim_path,
        usd_path=scene_usd_path,

    )
    

if not stage.ResolveIdentifierToEditTarget(robot_usd_path):
    raise FileNotFoundError(f"USD file not found at path: '{robot_usd_path}'.")

if not prim_utils.is_prim_path_valid(robot_prim_path):
    # add prim as reference to stage
    prim_utils.create_prim(
        robot_prim_path,
        usd_path=robot_usd_path,
        translation=(0,-1,0.15),
        orientation=rot_utils.euler_angles_to_quats(np.array([0, 0, 90]), degrees=True),
        # scale=scale,
    )
    # Gf.Rotation(Gf.Vec3d(0, 0, 1), 90).GetQuaternion()



# todo: config camera
width, height = 1280, 800
pixel_size = 3e-3
f_stop = 200
focal_length = 0.119
focus_distance = 0.7
diagonal_fov = 145
fx = 396.08072698
fy = 395.9067434
cx = 635.47899214
cy = 393.72551834
distortion_coefficients = [0.34009344, 0.11343523, -0.18409054, 0.06066355]
horizontal_aperture = pixel_size * width / 10
vertical_aperture = pixel_size * height / 10
camera = Camera(
    prim_path=f"{robot_prim_path}/head_roll_link/Camera",
    frequency=30,
    resolution=(width, height),
    translation=np.array([0.1, 0.0, -0.02]),  # 1 meter away from the side of the cube
    orientation=rot_utils.euler_angles_to_quats(np.array([0, 45, 0]), degrees=True),
)
camera.initialize()
camera.set_clipping_range(0.05, 300)
camera.set_focal_length(focal_length)
camera.set_focus_distance(focus_distance)
camera.set_lens_aperture(f_stop)
camera.set_horizontal_aperture(horizontal_aperture)
camera.set_vertical_aperture(vertical_aperture)
camera.set_projection_type("fisheyePolynomial")
camera.set_kannala_brandt_properties(
    width, height, cx, cy, diagonal_fov, distortion_coefficients
)
camera.add_motion_vectors_to_frame()
camera.add_distance_to_image_plane_to_frame()


# todo: robot articulation
robot_articulation = SingleArticulation(robot_articulation_path)
# robot_articulation.initialize()
left_arm_joint_names = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_pitch_joint",
    "left_wrist_yaw_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint"
]

right_arm_joint_names = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_pitch_joint",
    "right_wrist_yaw_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint"
]

left_hand_joint_names = [
    "L_index_proximal_joint",
    "L_middle_proximal_joint",
    "L_pinky_proximal_joint",
    "L_ring_proximal_joint",
    "L_thumb_proximal_yaw_joint",
    "L_thumb_proximal_pitch_joint"
]

right_hand_joint_names = [
    "R_index_proximal_joint",
    "R_middle_proximal_joint",
    "R_pinky_proximal_joint",
    "R_ring_proximal_joint",
    "R_thumb_proximal_yaw_joint",
    "R_thumb_proximal_pitch_joint"
]

joint_names = left_arm_joint_names + right_arm_joint_names + left_hand_joint_names + right_hand_joint_names

data_config = DATA_CONFIG_MAP["gr1_arms_only"]  # gr1_arms_only or gr1_arms_waist or gr1_full_upper_body
modality_config = data_config.modality_config()
transforms = data_config.transform()

policy = Gr00tPolicy(
    model_path="checkpoints/checkpoint-8500",
    modality_config=modality_config,
    modality_transform=transforms,
    embodiment_tag=EmbodimentTag.GR1,
    device="cuda"
)

# def on_physics_step(step_size):
#     pass

world= World(stage_units_in_meters=1.0)
# world.add_physics_callback("physics_step", callback_fn=on_physics_step)
world.play()

# simulation_context = SimulationContext(physics_prim_path="/World/PhysicsScene")
# simulation_context.initialize_physics()
# simulation_context.add_physics_callback("physics_step", callback_fn=on_physics_step)
# simulation_context.play()



while simulation_app.is_running():
    # world.step(render=True)
    simulation_app.update()

    # todo: get camera rgb info
    print(camera.get_current_frame())
    img_rgb = camera.get_rgb()
    
    if(img_rgb.size==0):
        continue
    img_rgb_resize = cv2.resize(img_rgb, (256, 256)) 
    cv2.imwrite(f"/mnt/mxy/tmp/images/{time.time()}.jpg", img_rgb_resize)
    img_rgb_resize_uint8 = img_rgb_resize.astype(np.uint8)
    print(f"img_rgb_resize_uint8:{img_rgb_resize_uint8.shape}")


    if not robot_articulation.handles_initialized:
        robot_articulation.initialize()

    # todo: get robot state info
    all_state=robot_articulation.get_joint_positions()

    left_arm_joint_indices = [robot_articulation.get_dof_index(x) for x in left_arm_joint_names]
    right_arm_joint_indices = [robot_articulation.get_dof_index(x) for x in right_arm_joint_names]
    left_hand_joint_indices = [robot_articulation.get_dof_index(x) for x in left_hand_joint_names]
    right_hand_joint_indices = [robot_articulation.get_dof_index(x) for x in right_hand_joint_names]
    joint_indices= [robot_articulation.get_dof_index(x) for x in joint_names]

    left_arm_state = robot_articulation.get_joint_positions(left_arm_joint_indices)
    right_arm_state = robot_articulation.get_joint_positions(right_arm_joint_indices)
    left_hand_state = robot_articulation.get_joint_positions(left_hand_joint_indices)
    right_hand_state = robot_articulation.get_joint_positions(right_hand_joint_indices)
    print(f"left_arm_state:{left_arm_state.shape}")
    print(f"right_arm_state:{right_arm_state.shape}")
    print(f"left_hand_state:{left_hand_state.shape}")
    print(f"right_hand_state:{right_hand_state.shape}")


    # input = {
    #     "video.ego_view": np.random.randint(0, 256, (1, 256, 256, 3), dtype=np.uint8),
    #     "state.left_arm": np.random.rand(1, 7),
    #     "state.right_arm": np.random.rand(1, 7),
    #     "state.left_hand": np.random.rand(1, 6),
    #     "state.right_hand": np.random.rand(1, 6),
    #     "state.waist": np.random.rand(1, 3),
    #     "annotation.human.action.task_description": ["do your thing!"],
    # }
    
    input = {
        "video.ego_view": np.expand_dims(img_rgb_resize_uint8, axis=0),
        "state.left_arm": np.expand_dims(left_arm_state, axis=0),
        "state.right_arm": np.expand_dims(right_arm_state, axis=0),
        "state.left_hand": np.expand_dims(left_hand_state, axis=0),
        "state.right_hand": np.expand_dims(right_hand_state, axis=0),
        # "state.waist": np.random.rand(1, 3),
        "annotation.human.action.task_description": ["pick up the black bowl on the plate!"],
    }

    # - input: video.ego_view: (1, 256, 256, 3)
    # - input: state.left_arm: (1, 7)
    # - input: state.right_arm: (1, 7)
    # - input: state.left_hand: (1, 6)
    # - input: state.right_hand: (1, 6)
    # - input: state.waist: (1, 3)

    for key in input.keys():
        print(f"{key}: {np.array(input[key]).shape}")


    output = policy.get_action(input) 
    for key in output.keys():
        print(f"{key}: {np.array(output[key]).shape}")

    # - output: action.left_arm: (16, 7)
    # - output: action.right_arm: (16, 7)
    # - output: action.left_hand: (16, 6)
    # - output: action.right_hand: (16, 6)
    # - output: action.waist: (16, 3)

    # todo: control robot
    action = ArticulationAction(joint_positions=output["action.left_arm"][0], joint_indices=left_arm_joint_indices)
    robot_articulation.apply_action(action)
    action = ArticulationAction(joint_positions=output["action.right_arm"][0], joint_indices=right_arm_joint_indices)
    robot_articulation.apply_action(action)
    action = ArticulationAction(joint_positions=output["action.left_hand"][0], joint_indices=left_hand_joint_indices)
    robot_articulation.apply_action(action)
    action = ArticulationAction(joint_positions=output["action.right_hand"][0], joint_indices=right_hand_joint_indices)
    robot_articulation.apply_action(action)


    # action = ArticulationAction(joint_positions=right_arm_state+[0.001,0.001,0.001,0.001,0.001,0.001,0.001], joint_indices=right_arm_joint_indices)
    robot_articulation.apply_action(action)

    

simulation_app.close()


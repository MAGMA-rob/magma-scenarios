from typing import Literal, Union, Sequence, Optional
import sapien, torch
import numpy as np
from os.path import dirname, join, realpath

from mani_skill.utils.building import ArticulationBuilder
from mani_skill.utils.building import URDFLoader
from mani_skill.envs.scene import ManiSkillScene
from mani_skill.agents.base_agent import BaseAgent
from mani_skill.utils.common import to_numpy
from transforms3d.euler import euler2quat

from magma_core.base.data_structures import Trajectory, Point

APPROACHING = np.array([0, 0, -1])
CLOSING = [0, -1, 0]
SEUIL = 0.05

def compute_grasp_drop_trajectory(
        agent : BaseAgent, 
        obj_pose: Union[np.ndarray, torch.Tensor],
        drop_pose: Union[np.ndarray, torch.Tensor, sapien.Pose],
        approach_seuil : float = 0,
        drop_seuil : float = 0,
        grasp_approach_pose: Optional[sapien.Pose]=None,
        drop_approach_pose: Optional[sapien.Pose]=None,
        final_pose : Optional[sapien.Pose]=None
    ) -> Trajectory:
        """Compute a trajectory to take an object at position obj_pose and drop it to drop_pose"""
        
        if not grasp_approach_pose:
            _obj_pose = to_numpy(obj_pose)
            grasp_approach_pose = sapien.Pose(
                p=_obj_pose[:3] + [0,0,_obj_pose[2]+SEUIL+approach_seuil],
                q = [0,1,0,0]
            )
            
        poses = compute_grasp_trajectory(agent,obj_pose,approach_seuil,grasp_approach_pose)

        if not isinstance(drop_pose, sapien.Pose):
            _drop_pose = to_numpy(drop_pose)
            drop_pose = sapien.Pose(
                p= _drop_pose[:3]+[0,0,_drop_pose[2]+drop_seuil],
                q = _drop_pose[3:]
            )

        if not drop_approach_pose:
            drop_approach_pose = sapien.Pose(p=[drop_pose.p[0], drop_pose.p[1], drop_pose.p[2]+approach_seuil], q = drop_pose.q)

        poses.extend([drop_approach_pose, drop_pose, "OPEN", drop_approach_pose])

        if final_pose and isinstance(final_pose,sapien.Pose):
            poses.append(final_pose)

        return poses

def compute_grasp_trajectory(agent : BaseAgent, obj_pose: Union[np.ndarray, torch.Tensor], add_seuil : float = 0, move_pos: Optional[sapien.Pose]=None) -> Trajectory:
        """
        Compute a trajectory to grasp the object at position obj_pose with robot agent.
        The add_seuil allows to define a custom value above the object to avoid for exemple walls.
        """
        _obj_pose = to_numpy(obj_pose)
        if not move_pos:
            move_pos = sapien.Pose(
                    p=_obj_pose[:3] + [0,0,_obj_pose[2]+SEUIL+add_seuil],
                    q = [0,1,0,0]
                )

        grasp_pos = agent.build_grasp_pose(
            APPROACHING,
            CLOSING,
            _obj_pose[:3])

        return ["OPEN", move_pos, grasp_pos, "CLOSE", move_pos]


def compute_drop_trajectory(agent : BaseAgent, drop_pose: Union[np.ndarray, torch.Tensor, sapien.Pose], approach_pose : Union[np.ndarray, torch.Tensor, sapien.Pose, None]=None, approach_seuil : float = 0.0, drop_seuil : float = 0,final_pose : Optional[sapien.Pose]=None) -> Trajectory:
    if not isinstance(drop_pose, sapien.Pose):
        _drop_pose = to_numpy(drop_pose)
        drop_pose = sapien.Pose(
            p= _drop_pose[:3]+[0,0,_drop_pose[2]+drop_seuil],
            q = [0, 1, 0, 0]
        )

    if approach_pose is None:
        approach_pose = sapien.Pose(p=[drop_pose.p[0], drop_pose.p[1], drop_pose.p[2]+approach_seuil], q = drop_pose.q)
    elif not isinstance(approach_pose, sapien.Pose):
        _approach_pose = to_numpy(approach_pose)
        approach_pose = sapien.Pose(
            p= _approach_pose[:3]+[0,0,_approach_pose[2]+approach_seuil],
            q = [0, 1, 0, 0]
        )
    

    poses = [approach_pose, drop_pose, "OPEN", approach_pose]

    if final_pose and isinstance(final_pose,sapien.Pose):
        poses.append(final_pose)

    return poses

def compute_press_trajectory(
        obj_pos: np.ndarray,
        orientation : np.ndarray = euler2quat(0, np.deg2rad(90), 0),
        button_stroke : float = 0.0018,
        add_press_seuil : float = 0.02,
        final_pose : Optional[sapien.Pose] = None,
    ) -> Trajectory:
        """ Compute action sequence to push an object, we suppose that the wrench is open by default. """
        if final_pose is None:
            final_pose = sapien.Pose(
                p=[-0.1,0,0.4],
                q = [0,1,0,0]
                )
        elif not isinstance(final_pose, sapien.Pose):
            final_pose = to_numpy(final_pose)
            final_pose = sapien.Pose(
                p = final_pose[:3],
                q = final_pose[3:]
            )

        # poses to start and stop pushing the button
        start_push_pos = sapien.Pose(
            p=obj_pos + [-0.12 - add_press_seuil, 0, 0.05],
            q = orientation
            )
        end_push_pos = sapien.Pose(
            p=obj_pos + [-0.09 - add_press_seuil + button_stroke, 0, 0.05],
            q = orientation
            )

        # robot pose sequence to perform the task
        return ["CLOSE", start_push_pos, end_push_pos, start_push_pos, final_pose]

def sapien_to_tensor(pose : sapien.Pose, device = "cpu") -> torch.Tensor:
    """Convert a sapien pose to a 7 long pytorch tensor like the one used in 
    the _get_obs_extra() env method."""
    array = np.concatenate((pose.get_p(), pose.get_q()))
    return torch.from_numpy(array).to(device)


def make_urdf_loader(scene:ManiSkillScene, density= 1, scale = 1, is_fix=True) -> URDFLoader:
    """ Create a urdf loader with the given physics properties. """
    loader = scene.create_urdf_loader()
    loader.scale = scale
    loader.fix_root_link = is_fix
    loader.set_material(0.3,0,0) # joint friction
    loader.set_density(density)
    return loader

def get_asset_path() -> str:
    """ Get the asset root path. """
    dir_path = dirname(realpath(__file__))
    return join(dir_path, "assets")

def make_articulation_builder(asset_name:str, loader:URDFLoader)-> ArticulationBuilder:
    """ Make a articulation builder for an URDF asset."""
    urdf_path = join(get_asset_path(), f"{asset_name}/mobility.urdf")
    # the .parse function can also parse multiple articulations
    # actors and cameras but we only use the articulations
    articulation_builders = loader.parse(str(urdf_path))["articulation_builders"]
    builder = articulation_builders[0]
    builder.initial_pose = sapien.Pose(p=[0,0,0])
    return builder

def make_obj_builder(scene:ManiSkillScene,obj_path:str, color, scale):
    """ Make a builder for an .obj asset."""
    mesh_path = join(get_asset_path(), obj_path)
    scale_tup = (scale, scale, scale)
    builder = scene.create_actor_builder()
    builder.add_nonconvex_collision_from_file(filename=mesh_path, scale=scale_tup)
    builder.add_visual_from_file(filename=mesh_path, material=color, scale=scale_tup)
    return builder
from mani_skill.utils.building.articulation_builder import ArticulationBuilder
from mani_skill.envs.scene import ManiSkillScene
from magma_scenarios.utils import make_urdf_loader, make_articulation_builder, make_obj_builder
from sapien import Pose

# see https://maniskill.readthedocs.io/en/latest/user_guide/tutorials/custom_tasks/loading_objects.html

def create_cardboard_box_builder(scene:ManiSkillScene) -> ArticulationBuilder:
    """ Create a builder to build cardboard box from a SAPIEN urdf file."""
    loader = make_urdf_loader(scene, scale=0.21)
    builder = make_articulation_builder(asset_name="box-100154", loader=loader)
    return builder

def create_trashcan(scene: ManiSkillScene, name="trashcan", add_collision=True):
    """ Create a create_trashcan from a SAPIEN urdf file."""
    loader = make_urdf_loader(scene, scale=0.3, is_fix=True, density=1)
    builder = make_articulation_builder(asset_name="trashcan", loader=loader)
    if not add_collision:
        for link_builder in builder.link_builders:
            link_builder.collision_records = []
    return builder.build(name=name)

def create_lamp(scene:ManiSkillScene, name="lamp"):
    """ Create a create_lamp from a SAPIEN urdf file."""
    loader = make_urdf_loader(scene, scale=0.2, is_fix=True, density=1)
    builder = make_articulation_builder(asset_name="lamp", loader=loader)
    return builder.build(name=name)

def create_switch(scene:ManiSkillScene, name="switch"):
    """ Create a switch from a SAPIEN urdf file."""
    loader = make_urdf_loader(scene, scale=0.08, is_fix=True, density=50)
    builder = make_articulation_builder(asset_name="switch", loader=loader)
    return builder.build(name=name)

def create_pen(scene:ManiSkillScene, name="pen"):
    """ Create a pen from a SAPIEN urdf file."""
    loader = make_urdf_loader(scene, scale=0.08, is_fix=False, density=50)
    builder = make_articulation_builder(asset_name="pen-101712", loader=loader)
    return builder.build(name=name)

def create_jar(scene:ManiSkillScene, name="jar"):
    """ Create a jar / bottle from a SAPIEN urdf file."""
    loader = make_urdf_loader(scene, scale=0.06, is_fix=False, density=10)
    builder = make_articulation_builder(asset_name="jar-4427", loader=loader)
    return builder.build(name=name)

def create_water_bottle(scene:ManiSkillScene, name="water_bottle"):
    """ Create a water bottle from a SAPIEN urdf file."""
    loader = make_urdf_loader(scene, scale=0.09, is_fix=False, density=10)
    builder = make_articulation_builder(asset_name="water-3822", loader=loader)
    return builder.build(name=name)


def create_wash_machine(scene:ManiSkillScene, name="washing_machine"):
    """ Create a water bottle from a SAPIEN urdf file."""
    loader = make_urdf_loader(scene, scale=0.5, is_fix=True, density=1)
    builder = make_articulation_builder(asset_name="washmachine-103781", loader=loader)
    return builder.build(name=name)

def create_coffee_maker(scene:ManiSkillScene, name="coffee_maker"):
    """ Create a coffee maker from a SAPIEN urdf file."""
    loader = make_urdf_loader(scene, scale=0.2, is_fix=True, density=1)
    builder = make_articulation_builder(asset_name="coffee_maker_103057", loader=loader)
    return builder.build(name=name)

def create_donut(scene:ManiSkillScene, name="donut", pose=Pose(p=[0, 0, 0]), color=(0.7,0.47,0), scale=0.02):
    """ Create a custom donut from .obj file."""
    builder = make_obj_builder(scene=scene, obj_path="donut/torus.obj", color=color, scale=scale)
    builder.set_initial_pose(pose)
    return builder.build_dynamic(name=name)

def create_soap(scene:ManiSkillScene, name="soap"):
    """ Create a soap container from a SAPIEN urdf file."""
    loader = make_urdf_loader(scene, scale=0.1, is_fix=False, density=50)
    builder = make_articulation_builder(asset_name="3398-soap", loader=loader)
    return builder.build(name=name)

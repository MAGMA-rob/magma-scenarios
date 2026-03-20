from mani_skill.utils.building.articulation_builder import ArticulationBuilder
from mani_skill.envs.scene import ManiSkillScene
from magma_scenarios.utils import make_urdf_loader, make_articulation_builder

# see https://maniskill.readthedocs.io/en/latest/user_guide/tutorials/custom_tasks/loading_objects.html

def create_cardboard_box_builder(scene:ManiSkillScene) -> ArticulationBuilder:
    """ Create a builder to build cardboard box from a SAPIEN urdf file."""
    loader = make_urdf_loader(scene, scale=0.21)
    builder = make_articulation_builder(asset_name="box-100154", loader=loader)
    return builder

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
"""Public components provided by this scenario."""

from magma_scenarios.manifest import ScenarioManifest

SCENARIO = ScenarioManifest(
    id='standalone_environments',
    environments={
        "SimpleAssemblyTask": (
            "magma_scenarios.envs.simple_assembly_task_env.simple_assembly_task_env"
        ),
        'SortCubesIndustrial-v1': 'magma_scenarios.envs.sort_cube_envs.sort_industrial',
        'SortCubesRGB-v1': 'magma_scenarios.envs.sort_cube_envs.rgb_env',
        'SortObject-v1': 'magma_scenarios.envs.sort_objects_env.simple_sort_object',
        'SortObject-v2': 'magma_scenarios.envs.sort_objects_env.complex_sort_object',
        'TrayPickingBase-v1': 'magma_scenarios.envs.tray_picking.tray_picking_env',
    },
)

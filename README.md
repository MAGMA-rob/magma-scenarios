# MAGMA Scenarios

Official robotic task scenarios and the shared scenario registry for MAGMA-GEN
and MAGMA-BENCH. The package provides simulation environments, task definitions,
presets, skills, and functions for discovering and loading scenarios from
installed Python packages.

You can add your own scenarios in a separate package without modifying this
repository or the generation pipeline.

## Installation

Python 3.12 is required. Simulation installation has been checked on Linux x86_64.

```bash
python -m pip install "magma_scenarios==2.0.0"
```

This installs `magma_core[simulation]` and the simulation dependencies, including
ManiSkill, SAPIEN, and PyTorch. Version 2.0.0 is a stable release of this package;
it accepts MAGMA Core starting at `2.0.0b1`, including subsequent stable v2
releases. GPU rendering requires suitable system drivers.

To install the release directly from GitHub:

```bash
python -m pip install "magma_scenarios @ git+https://github.com/MAGMA-rob/magma-scenarios.git@v2.0.0"
```

This installs the scenario code from GitHub and resolves dependencies from PyPI.
For development, clone this repository and run `python -m pip install -e .`.

## Discover and load scenarios

```bash
magma-scenarios list
magma-scenarios show press_button
```

The same registry is available in Python:

```python
from magma_scenarios import get_scenario, list_scenarios, load_definition, load_preset

print(list_scenarios())
manifest = get_scenario("press_button")
definition_type = load_definition("press_button.Definition")
preset_type = load_preset("press_button.ButtonPressOrdered")
```

Listing scenarios and reading manifests do not start a simulator or register
simulation environments. Loading a definition, preset, or skill registers its
scenario's environments and returns the component class; it does not create an
environment. Benchmarks can register an environment by ID with
`register_environment("PressButtonBasic-v1")`.

Included scenarios cover button pressing, sorting, delivery, table cleaning,
laundry, coffee preparation, and packaging. Use `magma-scenarios list` to see
all installed scenario providers.

## Add your own scenarios

Please refer to the official [documentation site](https://magma-rob.github.io/docs/intro)

## Development and support

Use `magma-scenarios --help` for all commands. `test-tools` and `test-requests`
provide interactive checks for scenario implementations; see their `--help`
for configuration options.

Report bugs through [GitHub Issues](https://github.com/MAGMA-rob/magma-scenarios/issues)
or contact [Loan Bernat](mailto:l.bernat@sileane.com).

## Authors and license

Loan Bernat, with contributions from Abdelbasset Houdass (internship).
Licensed under the [BSD 2-Clause License](https://github.com/MAGMA-rob/magma-scenarios/blob/main/LICENSE).

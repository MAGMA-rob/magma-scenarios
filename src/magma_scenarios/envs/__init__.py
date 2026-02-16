# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

import pkgutil
import importlib
import pathlib

# Path of the current package
package_path = pathlib.Path(__file__).parent

# Iterate through submodules and subpackages
for module in pkgutil.iter_modules([str(package_path)]):
    if module.ispkg:
        importlib.import_module(f"{__name__}.{module.name}")
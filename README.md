# MAGMA Scenarios

This repository contains a python package that store task implementation for use in [MAGMA-GEN](https://github.com/MAGMA-s/magma-gen) and [MAGMA-BENCH](https://github.com/MAGMA-s/magma-bench)

---

## Getting started

You can find in this package the task definition used in MAGMA. 

- **benchmark** : contains benchmark task instance. Until september 2026, full benchmark is kept private. Only tasks used in the MAGMA-GEN paper are available.
- **envs** : contains different maniskill3 environments which supports the different scenarios.
- **scenarios** : contains per-scenario task definition (Tool API, Stage and potential preset)

---
## How to use

This repository must be cloned and installed in the same environment as the rest of MAGMA packages. magma_gen and magma-bench will import it to instanciate tasks.

For development, install it from the cloned source in editable mode:

```bash
git clone https://github.com/MAGMA-rob/magma-scenarios.git
cd magma-scenarios
pip install "git+https://github.com/MAGMA-rob/magma-core.git@v0.1.0"
pip install -e .
```

Editable installation keeps the package linked to this checkout, so changes to scenarios,
tasks, configs, and assets are immediately visible to MAGMA-GEN.

---
## How to contribute

You can create your own scenario and open a merge request.

---

## Support
You can contact me at l.bernat@sileane.com

## Authors and acknowledgment
Loan BERNAT (l.bernat@sileane.com)
Abdelbasset HOUDASS (internship)

## License
BSD 2 clauses

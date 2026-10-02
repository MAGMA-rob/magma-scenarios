# Changelog

All notable changes to MAGMA Scenarios are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [2.1.0] - 2026-10-02

### Changed

- Clarified task instructions, permanent rules, tool descriptions, and their
  randomized variants to reduce ambiguity in zero-shot evaluation, with the
  aim of improving open-source model performance.
- Expanded advanced-delivery guidance for returns, robot access, transport,
  package preparation, valid placement targets, and shipping labels.
- Made delivery-box, packaging, warehouse-sorting, and table-cleaning permanent
  rules more explicit, and clarified color-sorting detection and errors.
- Kept the existing MAGMA Core dependency range, accepting v2 beta and stable
  releases starting at `2.0.0b1`.

### Fixed

- Removed end-of-stage resets from hall-sorting delivery stages to preserve
  the environment between progressive placement goals.
- Adjusted two-robot sorting stage budgets to give damaged-object sorting more
  tolerance and set an explicit tool-call limit for recipe stages.

## [2.0.2] - 2026-09-18

### Added

- Added default human-render camera views to the warehouse-sorting and
  table-cleaning environments.

### Changed

- Simplified the warehouse-sorting layout with compact fixed containers and
  collision-safe randomized reference-object positions.
- Made the table-cleaning trashcan visual collision-free while retaining a
  small hidden catcher surface, and lowered its drop approach for more reliable
  disposal.
- Removed the physical lamp and switch from the single-robot table-cleaning
  environment while retaining `set_light` as an inert randomized noise tool.

### Fixed

- Hardened warehouse-sorting take and put validation with exact object and area
  names, gripper-only object detection, consistent area thresholds, and more
  reachable lift and drop trajectories.
- Corrected drop trajectory height calculation so the configured offset is not
  added twice when the target is already elevated.
- Improved two-robot table cleaning with a closer table robot, consistent scene
  location detection, explicit storage access rules, and shared take/put access
  to the drying zone for clean dishware.
- Restored the table-cleaning trace-planning exports required to load the
  `AdvancedCleanTable` preset.

## [2.0.1] - 2026-09-17

### Fixed

- Corrected the press-button environment height and push trajectory so buttons
  spawned from definitions can be reached and pressed consistently.
- Added a final raised pose to the laundry grasp and drop trajectories so the
  robot lifts items clear of surrounding laundry.

## [2.0.0] - 2026-09-14

### Added

- Published the stable v2 scenario registry, environments, definitions,
  presets, skills, and testing commands.

[Unreleased]: https://github.com/MAGMA-rob/magma-scenarios/compare/v2.1.0...HEAD
[2.1.0]: https://github.com/MAGMA-rob/magma-scenarios/compare/v2.0.2...v2.1.0
[2.0.2]: https://github.com/MAGMA-rob/magma-scenarios/compare/v2.0.1...v2.0.2
[2.0.1]: https://github.com/MAGMA-rob/magma-scenarios/compare/v2.0.0...v2.0.1
[2.0.0]: https://github.com/MAGMA-rob/magma-scenarios/releases/tag/v2.0.0

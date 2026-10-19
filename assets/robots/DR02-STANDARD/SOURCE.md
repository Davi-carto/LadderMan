# DR02 Standard model

Source: <https://github.com/DeepRoboticsLab/deep_robotics_model>

Upstream commit: `549eb9ac08ab0674ad388202f0a276a1b140833b`

Imported paths:

- `DR02/mjcf/standard/DR02_std.xml`
- `DR02/urdf/standard/DR02-std.urdf`
- `DR02/usd/standard/DR02-std.usd`
- `DR02/usd/standard/configuration/`
- `DR02/mjcf/standard/meshes/`

The upstream MJCF and URDF directories contain byte-identical copies of the
same 22 mesh files. This directory keeps one shared `meshes/` copy because both
model files already reference that relative location. The upstream BSD
3-Clause license is retained in `LICENSE.txt`.

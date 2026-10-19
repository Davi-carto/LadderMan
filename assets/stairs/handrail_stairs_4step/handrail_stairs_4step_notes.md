# Handrail Stairs 4-Step

IsaacSim USD asset for a temporary handrail-stair data collection task.

## Geometry

- Stairs ascend along `+X`.
- Width is along `Y`, up is `Z`.
- Step height: `0.20 m`.
- Step depth: `0.40 m` for the first three steps.
- Top platform depth: `1.00 m`.
- Step width/length across `Y`: `1.20 m`.
- Step count: `4`.
- Handrail side: `+Y`.
- Post diameter: `0.08 m`.
- Post length: `1.20 m`, measured from the corresponding tread/platform surface.
- Top rail diameter: `0.08 m`.

The first four post bases are placed on the four tread centers at `Y = 0.64 m`, just outside the stair edge. Each post is vertical and starts from the corresponding stair surface. The sloped top rail connects the first four post tops. Three extra vertical posts are placed on the top platform and connected by a horizontal top rail.

## File

- `handrail_stairs_4step.usda`: primitive-based USD with collision APIs.

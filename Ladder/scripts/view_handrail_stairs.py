"""Load and view a static stair/handrail USD asset in IsaacSim."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from isaaclab.app import AppLauncher


DEFAULT_ASSET = Path(__file__).resolve().parents[2] / "assets" / "stairs" / "handrail_stairs_4step" / "handrail_stairs_4step.usda"

parser = argparse.ArgumentParser(description="View the handrail stair USD asset.")
parser.add_argument("--asset", type=str, default=str(DEFAULT_ASSET), help="USD/USDA asset path to load.")
parser.add_argument("--prim_path", type=str, default="/World/HandrailStairs", help="Prim path for the loaded asset.")
parser.add_argument("--x", type=float, default=0.0, help="Asset X translation.")
parser.add_argument("--y", type=float, default=0.0, help="Asset Y translation.")
parser.add_argument("--z", type=float, default=0.0, help="Asset Z translation.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils


def main():
    asset_path = os.path.abspath(args_cli.asset)
    if not os.path.isfile(asset_path):
        raise FileNotFoundError(f"Asset not found: {asset_path}")

    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([2.8, -2.4, 1.7], [0.8, 0.1, 0.5])

    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)

    light_cfg = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    asset_cfg = sim_utils.UsdFileCfg(usd_path=asset_path)
    asset_cfg.func(args_cli.prim_path, asset_cfg, translation=(args_cli.x, args_cli.y, args_cli.z))

    sim.reset()
    print(f"[INFO] Loaded asset: {asset_path}")
    print(f"[INFO] Prim path: {args_cli.prim_path}")
    print("[INFO] Close the IsaacSim window or press Ctrl+C in the terminal to stop.")

    while simulation_app.is_running():
        sim.step()


if __name__ == "__main__":
    main()
    simulation_app.close()

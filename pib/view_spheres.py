# pib/view_spheres.py
import argparse
import time
from pathlib import Path

from curobo.robot_builder import RobotBuilder


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="Inspect fitted collision spheres in Viser")
    ap.add_argument("--config", default=str(here / "pib.yml"))
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--meshes", action="store_true", help="also show the collision meshes")
    ap.add_argument("--no-spheres", action="store_true", help="hide the spheres")
    args = ap.parse_args()

    assert Path(args.config).is_file(), f"config not found: {args.config} (run pib_builder.py first)"

    builder = RobotBuilder.from_config(args.config)
    print(f"Loaded {args.config}: {builder.num_spheres} spheres")

    config = builder.build()
    viewer = builder.visualize(          # keep the reference alive, or the viewer stops
        config,
        port=args.port,
        show_meshes=args.meshes,
        show_spheres=not args.no_spheres,
    )

    print(f"Viewer at http://localhost:{args.port}  (Ctrl+C to stop)")
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Stopping viewer")


if __name__ == "__main__":
    main()
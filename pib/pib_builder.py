# pib/pib_builder.py
from pathlib import Path
from curobo.robot_builder import RobotBuilder


def main():
    here = Path(__file__).resolve().parent
    urdf_path = here / "robot" / "pib_planning.urdf"
    asset_path = here / "robot"          # meshes/root.obj -> robot/meshes/root.obj

    assert urdf_path.is_file(), f"URDF not found: {urdf_path}"
    assert (asset_path / "meshes").is_dir(), "run prep_urdf.py first"

    builder = RobotBuilder(str(urdf_path), str(asset_path))
    builder.fit_collision_spheres(sphere_density=2.0)
    builder.compute_collision_matrix(num_samples=2000)
    config = builder.build()
    builder.save(config, str(here / "pib.yml"))


if __name__ == "__main__":
    main()
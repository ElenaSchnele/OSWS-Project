from pathlib import Path
from curobo.robot_builder import RobotBuilder


def main():
    here = Path(__file__).resolve().parent

    urdf_path = here / "URDF" / "pib.control.urdf"
    asset_path = here.parent  # contains pib/meshes/...

    assert urdf_path.is_file(), f"URDF not found: {urdf_path}"
    assert asset_path.is_dir(), f"Asset dir not found: {asset_path}"

    builder = RobotBuilder(str(urdf_path), str(asset_path))
    builder.fit_collision_spheres(sphere_density=2.0)
    builder.compute_collision_matrix(num_samples=2000)
    config = builder.build()
    builder.save(config, str(here / "pib.yml"))


if __name__ == "__main__":
    main()
# pib/prep_urdf.py
import math
from collections import defaultdict
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import trimesh
from scipy import ndimage
from skimage import measure

HERE = Path(__file__).resolve().parent
SRC = HERE / "URDF" / "pib.urdf"
SRC_ASSETS = HERE.parent                # package://pib/meshes/X.glb -> SRC_ASSETS/pib/meshes/X.glb
OUT_URDF = HERE / "robot" / "pib_planning.urdf"
OUT_MESHES = HERE / "robot" / "meshes"

ACTIVE = {
    "rotation_inner_shoulder_1", "rotation_shoulder_left", "central_rotator_rotation",
    "rotation_elbow", "rotate_forearm", "rotation_wrist_right",
    "rotation_inner_shoulder", "rotation_shoulder_left_1", "central_rotator_rotation_1",
    "rotation_elbow_1", "rotate_forearm_1", "rotation_wrist",
    "gearing_rotation__new_body_", "nodding_rotation",
}
CONT_LIMIT = math.pi                    # placeholder for continuous / fixed-in-export joints

# Joints that are not planned with but are baked at a fixed angle (rad) instead of zero.
# The back lids are exported open (angle 0, range [-pi/2, 0]); the closed pose is assumed to be -pi/2.
# If a lid ends up pointing the wrong way in the viewer, flip the sign.
BAKE_POSE = {"upper_door_rotation": -math.pi / 2, "lower_door_rotation": -math.pi / 2}

# Geometry of these links is dropped (match = text before the first "_").
# Starting list, review it against the real robot.
DROP = {
    "a05", "a06", "a81", "a82", "a83",                       # head internals
    "b25", "b27", "b47", "b48", "b49", "b55", "b57", "b58",  # connectors, hinges, relay, logo
    "b62", "b63", "b64", "b65", "b66", "b67",                # knobs, stickers
    "c15", "c26", "c27", "c28", "c31", "c32", "c33", "c34",  # elbow/arm internals
    "c41", "c51", "c52", "c60l", "c60r", "c73", "c74", "c77",
}
DROP_EXACT = {"b", "p", "dot", "i_major", "i_minor"}

# Links to keep as zero-geometry frames (tool frames), even if merged into another body.
KEEP_FRAMES = {"d01_r_palm_right", "d01_l_palm_left"}

# Every body is turned into one closed, solid, watertight mesh. The value is the closing radius (m):
# gaps and holes narrower than 2*radius are sealed (speaker holes, seams), the inside is filled, and
# concavities wider than that stay. Larger radius = more sealing but more rounding of inner corners.
SOLIDIFY = {
    "root": 0.02,               # torso + neck
    "a81_motor_shaft": 0.012,   # head
}
SOLIDIFY_DEFAULT = 0.006        # all other bodies: just merge into one clean solid
MAX_FACES = 8000                # per body, keeps collision queries and sphere fitting fast


def dropped(name):
    return name.split("_")[0] in DROP or name in DROP_EXACT


def tf(origin):
    xyz, rpy = [0.0] * 3, [0.0] * 3
    if origin is not None:
        xyz = [float(v) for v in origin.get("xyz", "0 0 0").split()]
        rpy = [float(v) for v in origin.get("rpy", "0 0 0").split()]
    r, p, y = rpy
    Rx = np.array([[1, 0, 0], [0, math.cos(r), -math.sin(r)], [0, math.sin(r), math.cos(r)]])
    Ry = np.array([[math.cos(p), 0, math.sin(p)], [0, 1, 0], [-math.sin(p), 0, math.cos(p)]])
    Rz = np.array([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]])
    T = np.eye(4)
    T[:3, :3] = Rz @ Ry @ Rx
    T[:3, 3] = xyz
    return T


def axis_rot(axis_xyz, q):
    """4x4 rotation by q (rad) about the given axis (Rodrigues)."""
    a = np.array([float(v) for v in axis_xyz.split()])
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    T = np.eye(4)
    T[:3, :3] = np.eye(3) + math.sin(q) * K + (1 - math.cos(q)) * (K @ K)
    return T


def rpy_of(R):
    p = math.asin(max(-1.0, min(1.0, -R[2, 0])))
    if abs(math.cos(p)) > 1e-6:
        return math.atan2(R[2, 1], R[2, 2]), p, math.atan2(R[1, 0], R[0, 0])
    return math.atan2(-R[1, 2], R[1, 1]), p, 0.0


def set_origin(parent_el, T):
    o = ET.SubElement(parent_el, "origin")
    o.set("xyz", " ".join(f"{v:.9g}" for v in T[:3, 3]))
    o.set("rpy", " ".join(f"{v:.9g}" for v in rpy_of(T[:3, :3])))


def add_inertial(link):
    # dummy values: cuRobo ignores them, Isaac Sim's importer wants them present
    i = ET.SubElement(link, "inertial")
    ET.SubElement(i, "origin", xyz="0 0 0", rpy="0 0 0")
    ET.SubElement(i, "mass", value="1.0")
    ET.SubElement(i, "inertia", ixx="0.001", ixy="0", ixz="0",
                  iyy="0.001", iyz="0", izz="0.001")


def solidify(mesh, close_radius, pitch=None, smooth_iters=3, max_faces=MAX_FACES, max_cells=160):
    """Closed, solid, watertight copy of `mesh`.

    Gaps and holes narrower than ~2*close_radius are sealed and the enclosed interior is filled.
    Concavities wider than that are kept (inner corners get rounded by ~close_radius).
    The result is biased slightly outward, which is the conservative direction for collisions.
    """
    if pitch is None:
        pitch = max(close_radius / 4.0, float(mesh.extents.max()) / max_cells)
    r_vox = close_radius / pitch
    pad = int(np.ceil(r_vox)) + 3

    surf = mesh.voxelized(pitch)                        # also works on open meshes / triangle soups
    occ = np.pad(surf.matrix, pad)
    dil = ndimage.distance_transform_edt(~occ) <= r_vox + 0.5    # closing = dilate, fill, erode
    filled = ndimage.binary_fill_holes(dil)
    solid = ndimage.distance_transform_edt(filled) > r_vox + 0.75

    verts, faces, _, _ = measure.marching_cubes(solid.astype(np.float32), level=0.5)
    verts = trimesh.transformations.transform_points(verts - pad, surf.transform)
    out = trimesh.Trimesh(verts, faces, process=True)
    if smooth_iters:
        trimesh.smoothing.filter_taubin(out, iterations=smooth_iters)
    if max_faces:
        for target in (max_faces, 2 * max_faces, 4 * max_faces):   # keep the first watertight decimation
            if len(out.faces) <= target:
                break
            dec = out.simplify_quadric_decimation(face_count=target)
            if dec.is_watertight:
                out = dec
                break
    trimesh.repair.fix_normals(out)
    return out


def main():
    robot = ET.parse(SRC).getroot()
    links = {l.get("name"): l for l in robot.findall("link")}
    joints = robot.findall("joint")
    parent_joint = {j.find("child").get("link"): j for j in joints}
    children = defaultdict(list)
    for j in joints:
        children[j.find("parent").get("link")].append(j)
    root = next(n for n in links if n not in parent_joint)

    # world poses at zero configuration (BAKE_POSE joints at their baked angle), parents before children
    T_world, order, stack = {root: np.eye(4)}, [], [root]
    while stack:
        n = stack.pop()
        order.append(n)
        for j in children[n]:
            c = j.find("child").get("link")
            T = T_world[n] @ tf(j.find("origin"))
            q = BAKE_POSE.get(j.get("name"), 0.0)
            if q:
                T = T @ axis_rot(j.find("axis").get("xyz"), q)
            T_world[c] = T
            stack.append(c)

    # rigid components: a link belongs to the body of the nearest active joint above it
    top = {}
    for n in order:
        j = parent_joint.get(n)
        if j is None or j.get("name") in ACTIVE:
            top[n] = n
        else:
            top[n] = top[j.find("parent").get("link")]
    bodies = defaultdict(list)
    for n, t in top.items():
        bodies[t].append(n)

    # bake one solid mesh per body
    cache = {}
    OUT_MESHES.mkdir(parents=True, exist_ok=True)
    OUT_URDF.parent.mkdir(parents=True, exist_ok=True)
    has_mesh, report = {}, {}
    for t, members in bodies.items():
        parts = []
        for n in members:
            if dropped(n):
                continue
            for v in links[n].findall("visual"):
                mesh_el = v.find("geometry/mesh")
                if mesh_el is None:
                    continue
                path = SRC_ASSETS / mesh_el.get("filename").replace("package://", "")
                assert path.is_file(), f"missing mesh: {path}"
                if path not in cache:
                    cache[path] = trimesh.load(str(path), force="mesh")
                m = cache[path].copy()
                scale = [float(s) for s in mesh_el.get("scale", "1 1 1").split()]
                m.apply_scale(scale)
                m.apply_transform(np.linalg.inv(T_world[t]) @ T_world[n] @ tf(v.find("origin")))
                parts.append(m)
        has_mesh[t] = bool(parts)
        if parts:
            raw = trimesh.util.concatenate(parts)
            mesh = solidify(raw, SOLIDIFY.get(t, SOLIDIFY_DEFAULT))
            mesh.export(OUT_MESHES / f"{t}.obj")
            hull_vol = raw.convex_hull.volume
            report[t] = (len(mesh.faces), mesh.is_watertight, mesh.volume / hull_vol if hull_vol > 0 else 0.0)

    # write the merged URDF
    out = ET.Element("robot", name="pib_planning")
    for t in bodies:
        l = ET.SubElement(out, "link", name=t)
        add_inertial(l)
        if has_mesh[t]:
            g = ET.SubElement(ET.SubElement(l, "visual"), "geometry")
            ET.SubElement(g, "mesh", filename=f"meshes/{t}.obj")

    for name in sorted(KEEP_FRAMES):
        if name in top and top[name] != name:
            l = ET.SubElement(out, "link", name=name)
            add_inertial(l)
            j = ET.SubElement(out, "joint", name=f"{name}_frame", type="fixed")
            set_origin(j, np.linalg.inv(T_world[top[name]]) @ T_world[name])
            ET.SubElement(j, "parent", link=top[name])
            ET.SubElement(j, "child", link=name)

    for j in joints:
        if j.get("name") not in ACTIVE:
            continue
        parent = j.find("parent").get("link")
        src_type = j.get("type")
        new = ET.SubElement(out, "joint", name=j.get("name"),
                            type="revolute" if src_type in ("continuous", "fixed") else src_type)
        set_origin(new, np.linalg.inv(T_world[top[parent]]) @ T_world[parent] @ tf(j.find("origin")))
        ET.SubElement(new, "parent", link=top[parent])
        ET.SubElement(new, "child", link=j.find("child").get("link"))
        ET.SubElement(new, "axis", xyz=j.find("axis").get("xyz"))
        lim = ET.SubElement(new, "limit", effort="1", velocity="1")
        old = j.find("limit")
        if src_type in ("continuous", "fixed") or old is None:
            lim.set("lower", str(-CONT_LIMIT))
            lim.set("upper", str(CONT_LIMIT))
        else:
            lim.set("lower", old.get("lower"))
            lim.set("upper", old.get("upper"))

    ET.indent(out)
    ET.ElementTree(out).write(OUT_URDF, encoding="utf-8", xml_declaration=True)

    print(f"{len(bodies)} bodies, {sum(has_mesh.values())} with geometry, "
          f"{sum(1 for j in joints if j.get('name') in ACTIVE)} active joints")
    print(f"  {'body':40s} {'links':>5s} {'faces':>6s} {'watertight':>10s} {'vol/hull':>8s}")
    for t, members in bodies.items():
        f, w, r = report.get(t, (0, False, 0.0))
        print(f"  {t:40s} {len(members):5d} {f:6d} {str(w):>10s} {r:8.2f}")
    print("vol/hull near 1.0 = a filled solid; much lower (< ~0.5) on the torso or head = still hollow/open.")


if __name__ == "__main__":
    main()
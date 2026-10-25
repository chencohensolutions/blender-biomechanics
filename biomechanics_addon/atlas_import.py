"""Import muscles and ligaments from a Z-Anatomy atlas collection as strings.

Z-Anatomy (CC BY-SA 4.0) models every muscle's bony attachments as small
surface meshes named `<Muscle>.o<n><side>` (origins) and `<Muscle>.e<n><side>`
(insertions), and ligaments as meshes in its joints file. After the atlas has
been registered onto the scan (see DEVELOPMENT_PLAN.md, Atlas Completion), this
module turns those into the add-on's strings:

- Muscles: one string per origin/insertion pairing, each end snapped to the
  nearest vertex of the nearest scene bone, grouped per (muscle, side).
- Ligaments: a ligament mesh's two ends along its longest axis, snapped to two
  different bones, as a tension-only string.

Strings are created in both the Original and Rigid string collections through
the same machinery as the built-in atlases, so hover, colors, export, and the
chain solver treat them alike. Rebuilding the built-in strings recreates those
collections, so re-run the import afterwards.
"""

import re

import bpy  # pyright: ignore[reportMissingImports]
import numpy as np
from mathutils import Matrix  # pyright: ignore[reportMissingImports]
from mathutils.kdtree import KDTree  # pyright: ignore[reportMissingImports]

from .alignment import base_object_name, local_vertices, world_vertices
from .muscle_atlas import GROUPS
from .muscle_strings import (
    BUILT_FROM_PROPERTY,
    STRING_KIND_PROPERTY,
    STRING_SETS,
    _create_string,
    _mesh_by_base_name,
    clear_caches,
    muscle_collections,
    string_info,
)

GROUPS.update(
    {
        "atlas_muscle": ("Atlas muscles", (0.8, 0.35, 0.3, 1.0)),
        "atlas_ligament": ("Atlas ligaments", (0.75, 0.85, 0.6, 1.0)),
    }
)

REGION = re.compile(r"^(?P<name>.+?)\.(?P<kind>[oe])(?P<index>\d*)(?P<side>[lr])$")
LIGAMENT_WORDS = ("ligament", "membrane", "capsule", "retinaculum", "tendon")
ATLAS_SOURCE_PROPERTY = "biomechanics_atlas_source"
SUBSAMPLE = 4
SNAP_LIMIT_MM = 25.0
LIGAMENT_COMPLIANCE = 0.3
LIGAMENT_SLACK = 0.02


class BoneIndex:
    """Nearest scene-bone vertex lookup over one bone collection."""

    def __init__(self, collection):
        self.entries = []
        points = []
        for obj in collection.objects:
            if obj.type != "MESH" or string_info(obj) is not None or obj.hide_get():
                continue
            vertices = world_vertices(obj)
            for index in range(0, len(vertices), SUBSAMPLE):
                self.entries.append((obj, index))
                points.append(vertices[index])
        self.tree = KDTree(len(points))
        for position, point in enumerate(points):
            self.tree.insert(point, position)
        self.tree.balance()
        self._local = {}

    def nearest(self, point, limit=SNAP_LIMIT_MM / 1000.0):
        """(bone object, exact nearest vertex index) or None when nothing is within limit."""
        location, position, distance = self.tree.find(point)
        if position is None or distance > limit:
            return None
        obj, _coarse = self.entries[position]
        vertices = self._local.get(obj.name)
        if vertices is None:
            vertices = self._local[obj.name] = world_vertices(obj)
        index = int(np.argmin(np.linalg.norm(vertices - np.asarray(point), axis=1)))
        return obj, index


def atlas_regions(atlas):
    """{(muscle name, side): {'o': [(index, obj)], 'e': [(index, obj)]}}."""
    regions = {}
    for obj in atlas.objects:
        if obj.type != "MESH":
            continue
        match = REGION.match(obj.name)
        if not match:
            continue
        key = (match.group("name"), "L" if match.group("side") == "l" else "R")
        index = int(match.group("index") or 0)
        regions.setdefault(key, {"o": [], "e": []})[match.group("kind")].append((index, obj))
    for lists in regions.values():
        lists["o"].sort()
        lists["e"].sort()
    return regions


def _centroid(obj, transform):
    vertices = world_vertices(obj)
    matrix = np.array(transform, dtype=np.float64)
    return (vertices @ matrix[:3, :3].T + matrix[:3, 3]).mean(axis=0)


def pair_regions(origins, insertions):
    """Pair origin and insertion regions: index-matched, fanned, or nearest."""
    if len(origins) == len(insertions):
        return list(zip(origins, insertions))
    if len(origins) == 1:
        return [(origins[0], insertion) for insertion in insertions]
    if len(insertions) == 1:
        return [(origin, insertions[0]) for origin in origins]
    pairs = []
    for origin in origins:
        nearest = min(insertions, key=lambda insertion: np.linalg.norm(insertion[1] - origin[1]))
        pairs.append((origin, nearest))
    return pairs


def _twin_attachments(attachments, meshes):
    twins = []
    for bone, index in attachments:
        twin = meshes.get(base_object_name(bone.name))
        if not twin or index >= len(twin[0].data.vertices):
            return None
        twins.append((twin[0], index))
    return twins


def _string_collection(scene, bone_collection, kind):
    existing = muscle_collections(scene, kind).get(bone_collection.name)
    if existing is not None:
        return existing
    collection = bpy.data.collections.new(f"{bone_collection.name} {STRING_SETS[kind][0]}")
    collection[BUILT_FROM_PROPERTY] = bone_collection.name
    collection[STRING_KIND_PROPERTY] = kind
    scene.collection.children.link(collection)
    return collection


def _remove_previous_import(scene, kind):
    for collection in muscle_collections(scene, kind).values():
        for obj in list(collection.objects):
            if obj.get(ATLAS_SOURCE_PROPERTY):
                data = obj.data
                bpy.data.objects.remove(obj, do_unlink=True)
                if data is not None and data.users == 0:
                    bpy.data.curves.remove(data)


def existing_muscle_labels(scene):
    labels = set()
    for collection in muscle_collections(scene, "muscle").values():
        for obj in collection.objects:
            info = string_info(obj)
            if info is not None and not obj.get(ATLAS_SOURCE_PROPERTY):
                labels.add(_normalize(info["label"]))
    return labels


def _normalize(label):
    label = label.lower().replace(" muscle", "").replace("musculus ", "")
    return re.sub(r"[^a-z]+", " ", label).strip()


def region_transform_lookup(atlas, extra_transforms):
    """Map each atlas bone object to the extra transform applied to it (femur ICP, neck shift)."""
    bones = []
    for obj in atlas.objects:
        if obj.type != "MESH" or REGION.match(obj.name) or obj.name.startswith("("):
            continue
        bones.append(obj)
    centroids = np.array([world_vertices(obj).mean(axis=0) for obj in bones]) if bones else np.zeros((0, 3))

    def lookup(point):
        if len(bones) == 0:
            return Matrix.Identity(4)
        nearest = bones[int(np.argmin(np.linalg.norm(centroids - point, axis=1)))]
        for predicate, transform in extra_transforms:
            if predicate(nearest.name):
                return transform
        return Matrix.Identity(4)

    return lookup


def import_atlas_muscles(scene, atlas, extra_transforms=(), skip_existing=True):
    """Create muscle strings from atlas attachment regions. Returns (created, skipped reasons)."""
    source = scene.biomechanics_source_collection
    target = scene.biomechanics_target_collection
    clear_caches()
    _remove_previous_import(scene, "muscle")
    index = BoneIndex(source)
    target_meshes = _mesh_by_base_name(target)
    lookup = region_transform_lookup(atlas, extra_transforms)
    known = existing_muscle_labels(scene) if skip_existing else set()
    source_strings = _string_collection(scene, source, "muscle")
    target_strings = _string_collection(scene, target, "muscle")

    created = 0
    skipped = []
    for (name, side), lists in sorted(atlas_regions(atlas).items()):
        if not lists["o"] or not lists["e"]:
            continue
        if _normalize(name) in known:
            continue
        origins = [(idx, _centroid(obj, lookup(world_vertices(obj).mean(axis=0)))) for idx, obj in lists["o"]]
        insertions = [(idx, _centroid(obj, lookup(world_vertices(obj).mean(axis=0)))) for idx, obj in lists["e"]]
        pairs = pair_regions(origins, insertions)
        strings = []
        for (o_index, o_point), (i_index, i_point) in pairs:
            start = index.nearest(o_point)
            end = index.nearest(i_point)
            if start is None or end is None:
                skipped.append(f"{name} {side}: attachment not near a bone")
                continue
            if start[0] is end[0]:
                skipped.append(f"{name} {side}: both ends on {start[0].name}")
                continue
            strings.append((f"o{o_index} -> e{i_index}", [start, end]))
        key = _normalize(name).replace(" ", "_")
        for position, (string_name, attachments) in enumerate(strings):
            string = {
                "muscle": f"atlas_{key}",
                "label": name,
                "group": "atlas_muscle",
                "side": side,
                "index": position,
                "count": len(strings),
                "name": string_name,
                "kind": "muscle",
            }
            twins = _twin_attachments(attachments, target_meshes)
            obj = _create_string(source_strings, source, string, attachments, scene)
            obj[ATLAS_SOURCE_PROPERTY] = "Z-Anatomy (CC BY-SA 4.0)"
            if twins is not None:
                twin = _create_string(target_strings, target, string, twins, scene)
                twin[ATLAS_SOURCE_PROPERTY] = "Z-Anatomy (CC BY-SA 4.0)"
            created += 1
    return created, skipped


def ligament_ends(obj, transform):
    """Two end points of a ligament mesh along its longest axis."""
    matrix = np.array(transform, dtype=np.float64)
    vertices = world_vertices(obj) @ matrix[:3, :3].T + matrix[:3, 3]
    center = vertices.mean(axis=0)
    _u, _s, vt = np.linalg.svd(vertices - center, full_matrices=False)
    axis = vt[0]
    projections = (vertices - center) @ axis
    low = vertices[projections <= np.percentile(projections, 5)].mean(axis=0)
    high = vertices[projections >= np.percentile(projections, 95)].mean(axis=0)
    return low, high


def import_atlas_ligaments(scene, joints, extra_transforms=()):
    """Create tension-only ligament strings from ligament meshes of the joints collection."""
    source = scene.biomechanics_source_collection
    target = scene.biomechanics_target_collection
    clear_caches()
    _remove_previous_import(scene, "ligament")
    index = BoneIndex(source)
    target_meshes = _mesh_by_base_name(target)
    atlas = bpy.data.collections.get("atlas")
    lookup = region_transform_lookup(atlas, extra_transforms) if atlas else (lambda _point: Matrix.Identity(4))
    source_strings = _string_collection(scene, source, "ligament")
    target_strings = _string_collection(scene, target, "ligament")
    created = 0
    skipped = []
    groups = {}
    for obj in joints.objects:
        if obj.type != "MESH" or not any(word in obj.name.lower() for word in LIGAMENT_WORDS):
            continue
        match = re.match(r"^(?P<name>.+?)(?:\.(?P<side>[lr]))?$", obj.name)
        name, side = match.group("name"), {"l": "L", "r": "R", None: "M"}[match.group("side")]
        transform = lookup(world_vertices(obj).mean(axis=0))
        low, high = ligament_ends(obj, transform)
        start, end = index.nearest(low), index.nearest(high)
        if start is None or end is None:
            skipped.append(f"{obj.name}: end not near a bone")
            continue
        if start[0] is end[0]:
            skipped.append(f"{obj.name}: both ends on {start[0].name}")
            continue
        groups.setdefault((name, side), []).append([start, end])
    for (name, side), attachments_list in sorted(groups.items()):
        key = _normalize(name).replace(" ", "_")
        for position, attachments in enumerate(attachments_list):
            string = {
                "muscle": f"atlas_{key}",
                "label": name,
                "group": "atlas_ligament",
                "side": side,
                "index": position,
                "count": len(attachments_list),
                "name": f"part {position + 1}",
                "kind": "ligament",
                "compliance": LIGAMENT_COMPLIANCE,
                "unilateral": True,
                "slack": LIGAMENT_SLACK,
                "mode": "tension",
            }
            twins = _twin_attachments(attachments, target_meshes)
            obj = _create_string(source_strings, source, string, attachments, scene)
            obj[ATLAS_SOURCE_PROPERTY] = "Z-Anatomy (CC BY-SA 4.0)"
            if twins is not None:
                twin = _create_string(target_strings, target, string, twins, scene)
                twin[ATLAS_SOURCE_PROPERTY] = "Z-Anatomy (CC BY-SA 4.0)"
            created += 1
    return created, skipped


def extra_transforms_from_session():
    """Per-region transforms recorded by the registration scripts (femur ICP, neck shift)."""
    transforms = []
    fits = bpy.app.driver_namespace.get("femur_fit", {})
    leg_words = ("Femur", "Patella", "Tibia", "Fibula", "Calcaneus", "Talus", "Navicular", "Cuboid", "cuneiform", "metatarsal", "of foot", "Sesamoid bones of foot")
    for side_key, suffix in (("L", ".l"), ("R", ".r")):
        if side_key in fits:
            matrix = Matrix(fits[side_key])
            transforms.append((lambda n, suffix=suffix: n.endswith(suffix) and any(w in n for w in leg_words), matrix))
    neck = bpy.app.driver_namespace.get("neck_offset")
    if neck is not None:
        neck_words = ("Vertebra C", "Atlas (C1)", "Axis (C2)", "Mandible", "Hyoid", "bone", "tooth", "incisor", "canine", "premolar", "Vomer", "Sinus", "cells of ethmoid", "septal cartilage")
        skull_exclusions = ("Hip bone", "Capitate", "Hamate", "Lunate", "Pisiform", "Scaphoid", "Trapezium", "Trapezoid", "Triquetrum", "metacarpal", "metatarsal", "Navicular", "Cuboid", "cuneiform")
        transforms.append(
            (lambda n: any(w in n for w in neck_words) and not any(w in n for w in skull_exclusions) and not n.endswith(("rib.l", "rib.r")), Matrix.Translation(neck))
        )
    return transforms


class BIOMECHANICS_OT_import_atlas_muscles(bpy.types.Operator):
    """Create muscle strings from the Z-Anatomy attachment areas in the 'atlas' collection"""

    bl_idname = "biomechanics.import_atlas_muscles"
    bl_label = "Import Atlas Muscles"
    bl_options = {"REGISTER", "UNDO"}

    skip_existing: bpy.props.BoolProperty(name="Skip muscles already modeled", default=True)

    @classmethod
    def poll(cls, context):
        return bpy.data.collections.get("atlas") is not None and context.scene.biomechanics_source_collection is not None

    def execute(self, context):
        created, skipped = import_atlas_muscles(
            context.scene, bpy.data.collections["atlas"], extra_transforms_from_session(), self.skip_existing
        )
        message = f"{created} atlas muscle strings"
        if skipped:
            message += f"; skipped {len(skipped)}: {skipped[0]}"
        self.report({"WARNING"} if skipped else {"INFO"}, message)
        return {"FINISHED"}


class BIOMECHANICS_OT_import_atlas_ligaments(bpy.types.Operator):
    """Create ligament strings from the Z-Anatomy ligament meshes in the 'atlas joints' collection"""

    bl_idname = "biomechanics.import_atlas_ligaments"
    bl_label = "Import Atlas Ligaments"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return bpy.data.collections.get("atlas joints") is not None and context.scene.biomechanics_source_collection is not None

    def execute(self, context):
        created, skipped = import_atlas_ligaments(context.scene, bpy.data.collections["atlas joints"], extra_transforms_from_session())
        message = f"{created} atlas ligament strings"
        if skipped:
            message += f"; skipped {len(skipped)}: {skipped[0]}"
        self.report({"WARNING"} if skipped else {"INFO"}, message)
        return {"FINISHED"}

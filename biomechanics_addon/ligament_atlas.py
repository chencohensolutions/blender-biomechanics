"""Ligament and intervertebral disc atlas for the trunk, pelvis, and hip (bpy-free).

Ligaments use the same string model as muscles (`muscle_atlas.py`): a string is a
list of `(bone, landmark)` or `(bone, landmark, side)` points, one string per
bony attachment, and both Original and Rigid collections get the same strings.
Each entry also carries the mechanical data the chain solver needs:

- `compliance`: XPBD compliance (stretch per unit force, arbitrary units); lower
  is stiffer.
- `unilateral`: True for ligaments, which resist only tension; False for disc
  springs, which resist compression and tension.
- `slack`: strain (fraction of rest length) a ligament may stretch before it
  resists, so a chain stays "a bit stretchy".

A disc between two vertebral bodies is a bundle of springs between the facing
endplates: one central spring (the nucleus, stiff) and a ring of peripheral
springs (the annulus, softer). Rest lengths always come from the Original pose.

Landmark specs are `(direction, bands)` or `(direction, bands, mode)`. Bands may
use axis "M", the normalized distance from the midline, which does not depend on
side. Mode "centroid" picks the vertex nearest the centroid of the banded region
instead of the extreme vertex along the direction. Joint surfaces use inline dict
specs with `near`: the banded region is first cut down to the vertices within
`tolerance_mm` of the bone's closest approach to the named neighbor, which finds a
contact face (endplate, facet, auricular surface) no extreme point can.
"""

import math

from .muscle_atlas import GROUPS, LANDMARKS, LUMBAR, MIDLINE, SPINE, THORACIC

RING_POINTS = 6

ADJACENT_VERTEBRAE = tuple(zip(SPINE, SPINE[1:]))  # (upper, lower) from T01-T02 to L4-L5

GROUPS.update(
    {
        "ligament_spine": ("Spinal ligaments", (0.95, 0.95, 0.6, 1.0)),
        "ligament_pelvis": ("Pelvic ligaments", (0.7, 0.95, 0.6, 1.0)),
        "ligament_hip": ("Hip ligaments", (0.6, 0.9, 0.9, 1.0)),
        "ligament_costal": ("Costovertebral ligaments", (0.95, 0.75, 0.6, 1.0)),
        "costal_cartilage": ("Costal cartilage", (0.85, 0.9, 0.95, 1.0)),
        "disc_nucleus": ("Disc nucleus", (0.95, 0.4, 0.4, 1.0)),
        "disc_strut": ("Disc strut", (0.6, 0.1, 0.1, 1.0)),
        "disc_annulus": ("Disc annulus", (0.95, 0.65, 0.5, 1.0)),
        "disc_fibers": ("Disc oblique fibers", (0.9, 0.5, 0.3, 1.0)),
    }
)

# Stiffness classes: (compliance, unilateral, slack). "disc_strut" is special-cased as a
# rigid compression-only element (mode "compression") in `_entry`. XPBD compliance is relative to
# a body inverse mass of about 1: values near 1 stretch readily, near 0.05 are stiff.
MECHANICS = {
    "disc_nucleus": (0.02, False, 0.0),
    "disc_strut": (0.0, False, 0.0),
    # Oblique annulus lamellae: tension-only, no slack; they give the disc its shear
    # and torsion stiffness (vertical rim springs alone let vertebrae slide sideways).
    "disc_fibers": (0.05, True, 0.0),
    "disc_annulus": (0.08, False, 0.0),
    "longitudinal": (0.3, True, 0.02),
    "flavum": (3.0, True, 0.05),
    "interspinous": (2.0, True, 0.04),
    "supraspinous": (1.0, True, 0.03),
    "intertransverse": (4.0, True, 0.05),
    "capsular": (1.5, True, 0.03),
    "iliolumbar": (0.3, True, 0.02),
    "sacroiliac": (0.1, True, 0.01),
    "sacropelvic": (0.3, True, 0.02),
    "pubic": (0.1, True, 0.01),
    "hip_capsule": (0.3, True, 0.03),
    "costal": (0.5, True, 0.02),
    # Cartilage bars from the bony rib ends to the sternum: stiff and bilateral, so
    # the sternum rides on the rib cage instead of floating when the spine bends.
    "cartilage": (0.05, False, 0.0),
}


def near(direction, bands, near_bone, tolerance_mm=3.0, mode="extreme"):
    """Inline landmark spec for a contact region facing near_bone."""
    return {"direction": direction, "bands": bands, "near": near_bone, "tolerance_mm": tolerance_mm, "mode": mode}


def nearest_to(other_bone, other_landmark, bands=None, side=None):
    """Inline landmark spec: the vertex (within bands) closest to another bone's landmark."""
    spec = {"direction": CENTER, "bands": bands or {}, "nearest_to": [other_bone, other_landmark]}
    if side is not None:
        spec["nearest_to"].append(side)
    return spec


BODY_BANDS = {"A": (0.55, 1.0), "M": (0.0, 0.35)}
FACET_BANDS = {"A": (0.05, 0.55), "L": (0.45, 0.9)}
CENTER = (0.0, 0.0, 0.0)


def _ring_direction(index):
    angle = 2.0 * math.pi * index / RING_POINTS
    return (math.cos(angle), math.sin(angle), 0.0)


def _vertebra_landmarks():
    body = {"A": (0.55, 1.0), "M": (0.0, 0.35)}
    landmarks = {
        "body_anterior_superior": ((1.0, 0.0, 0.2), {"A": (0.55, 1.0), "M": (0.0, 0.12), "S": (0.6, 1.0)}),
        "body_anterior_inferior": ((1.0, 0.0, -0.2), {"A": (0.55, 1.0), "M": (0.0, 0.12), "S": (0.0, 0.4)}),
        "body_posterior_superior": ((-1.0, 0.0, 0.1), {"A": (0.5, 0.78), "M": (0.0, 0.1), "S": (0.55, 0.95)}),
        "body_posterior_inferior": ((-1.0, 0.0, -0.1), {"A": (0.5, 0.78), "M": (0.0, 0.1), "S": (0.05, 0.45)}),
        "body_posterolateral": ((-0.4, 1.0, 0.0), {"A": (0.5, 0.8), "S": (0.3, 0.8)}),
        "lamina_superior": ((-0.6, 0.0, 1.0), {"A": (0.1, 0.45), "L": (0.5, 0.7)}),
        "lamina_inferior": ((-0.6, 0.0, -1.0), {"A": (0.1, 0.45), "L": (0.5, 0.7)}),
        "spinous_superior": ((-0.4, 0.0, 1.0), {"A": (0.0, 0.3), "M": (0.0, 0.12)}),
        "spinous_inferior": ((-0.4, 0.0, -1.0), {"A": (0.0, 0.3), "M": (0.0, 0.12)}),
        "articular_inferior": ((-0.3, 0.0, -1.0), {"A": (0.15, 0.55), "L": (0.6, 0.8)}),
    }
    return landmarks


LIGAMENT_LANDMARKS = {
    "vertebra": _vertebra_landmarks(),
    "sacrum": {
        "promontory": ((0.8, 0.0, 0.6), {"M": (0.0, 0.15)}),
        "posterior_superior": ((-0.6, 0.0, 0.8), {"M": (0.0, 0.15), "A": (0.3, 0.7)}),
        "lateral_upper": ((0.0, 1.0, 0.3), {"S": (0.6, 0.9)}),
        "posterolateral_lower": ((-0.5, 0.8, -0.3), {"S": (0.15, 0.45)}),
    },
    "ilium": {
        "iliac_tuberosity": ((-0.7, -0.7, 0.2), {"S": (0.55, 0.8)}),
        "aiis": ((1.0, 0.0, 0.2), {"S": (0.45, 0.6)}),
    },
    "femur": {
        "neck_anterior": ((1.0, 0.0, 0.0), {"S": (0.6, 0.9), "L": (0.3, 0.7)}),
        "neck_posterior": ((-1.0, 0.0, 0.0), {"S": (0.6, 0.9), "L": (0.3, 0.7)}),
        "neck_inferior": ((0.3, -0.5, -0.8), {"S": (0.5, 0.75), "L": (0.2, 0.6)}),
    },
}

# Contact-surface landmarks, parameterized by the neighboring bone.
# The endplate region is every body vertex within this distance of the closest approach
# to the neighboring vertebra. A wedge-shaped disc varies several mm across the plate, so
# a tight band would catch only a sliver and the rim points would collapse together.
ENDPLATE_TOLERANCE_MM = 8.0


def endplate(toward, direction=CENTER, bands=None):
    bands = bands or BODY_BANDS
    return near(direction, bands, toward, ENDPLATE_TOLERANCE_MM, "centroid" if direction == CENTER else "extreme")


def facet(toward):
    return near(CENTER, FACET_BANDS, toward, 4.0, "centroid")


def sacral_endplate(direction=CENTER):
    return near(
        direction, {"A": (0.4, 1.0), "M": (0.0, 0.4)}, "L5", ENDPLATE_TOLERANCE_MM, "centroid" if direction == CENTER else "extreme"
    )


def auricular(bone, other, part):
    bands = {"S": (0.5, 0.95) if part == "upper" else (0.2, 0.5)} if bone == "sacrum" else {"S": (0.3, 0.8)}
    if part == "anterior":
        return near((1.0, 0.0, 0.0), {"S": (0.3, 0.8)}, other, 4.0)
    if bone != "sacrum":
        bands = {"S": (0.55, 0.8) if part == "upper" else (0.3, 0.55)}
    return near(CENTER, bands, other, 3.0, "centroid")


def acetabulum(side, part):
    femur = f"femur {side}"
    bands = {"S": (0.2, 0.6), "L": (0.5, 1.0)}
    if part == "center":
        return near(CENTER, bands, femur, 3.0, "centroid")
    return near((1.0, 0.3, 0.0) if part == "anterior" else (-1.0, 0.3, 0.0), bands, femur, 6.0)


def femoral_head(side):
    return near(CENTER, {"S": (0.6, 1.0)}, f"ilium {side}", 3.0, "centroid")


def rib_head(level, side):
    return near(CENTER, {"L": (0.0, 0.3)}, f"T{level:02d}", 3.0, "centroid")


def rib_tubercle(level, side):
    return near(CENTER, {"L": (0.08, 0.4)}, f"T{level:02d}", 4.0, "centroid")


def costal_facet(level, side):
    return near(CENTER, {"A": (0.4, 0.9), "L": (0.4, 1.0)}, f"T{level:02d} rib {side}", 3.0, "centroid")


def transverse_costal_facet(level, side):
    return near(CENTER, {"L": (0.6, 1.0)}, f"T{level:02d} rib {side}", 3.0, "centroid")

for _kind, _landmarks in LIGAMENT_LANDMARKS.items():
    LANDMARKS[_kind].update(_landmarks)



def _entry(key, label, group, mechanics, side, index, count, name, points):
    compliance, unilateral, slack = MECHANICS[mechanics]
    return {
        "muscle": key,
        "label": label,
        "group": group,
        "side": side,
        "index": index,
        "count": count,
        "name": name,
        "points": tuple(points),
        "kind": "disc" if group.startswith("disc") else "ligament",
        "compliance": compliance,
        "unilateral": unilateral,
        "slack": slack,
        "mode": "compression" if mechanics == "disc_strut" else ("tension" if unilateral else "both"),
    }


def _bundle(key, label, group, mechanics, side, strings):
    """strings: [(name, points)] -> entries sharing key/side."""
    return [
        _entry(key, label, group, mechanics, side, index, len(strings), name, points)
        for index, (name, points) in enumerate(strings)
    ]


def _rib(level, side):
    return f"T{level:02d} rib {side}"


def _segments_with_sacrum():
    """(upper, lower, lower landmarks) for every disc level including L5-S1."""
    for upper, lower in ADJACENT_VERTEBRAE:
        yield upper, lower, "vertebra"
    yield "L5", "sacrum", "sacrum"


def disc_strings():
    entries = []
    for upper, lower, lower_kind in _segments_with_sacrum():
        if lower_kind == "sacrum":
            lower_center = (lower, sacral_endplate())
            lower_rim = sacral_endplate
        else:
            lower_center = (lower, endplate(upper))
            lower_rim = lambda direction: endplate(upper, direction)  # noqa: E731
        key = f"disc_{upper}_{lower}".lower().replace(" ", "_")
        label = f"Disc {upper}-{lower if lower != 'sacrum' else 'S1'}"
        entries += _bundle(
            key + "_nucleus", label + " nucleus", "disc_nucleus", "disc_nucleus", MIDLINE,
            [("center", ((upper, endplate(lower)), lower_center))],
        )
        # The disc cannot get shorter than its rest height: the column carries load axially.
        entries += _bundle(
            key + "_strut", label + " strut", "disc_strut", "disc_strut", MIDLINE,
            [("center", ((upper, endplate(lower)), lower_center))],
        )
        ring = []
        fibers = []
        for index in range(RING_POINTS):
            # Six distinct extreme points on each endplate: rim i above faces rim i below,
            # and the oblique fibers join rim i above to rims i +/- 1 below.
            upper_rim = endplate(lower, _ring_direction(index))
            ring.append((f"rim {index}", ((upper, upper_rim), (lower, lower_rim(_ring_direction(index))))))
            for step in (-1, 1):
                neighbor = (index + step) % RING_POINTS
                fibers.append((f"rim {index} -> {neighbor}", ((upper, upper_rim), (lower, lower_rim(_ring_direction(neighbor))))))
        entries += _bundle(key + "_annulus", label + " annulus", "disc_annulus", "disc_annulus", MIDLINE, ring)
        entries += _bundle(key + "_fibers", label + " oblique fibers", "disc_fibers", "disc_fibers", MIDLINE, fibers)
    return entries


def spinal_ligament_strings():
    entries = []
    segments = list(_segments_with_sacrum())
    longitudinal_anterior = []
    longitudinal_posterior = []
    supraspinous = []
    interspinous = []
    for upper, lower, lower_kind in segments:
        lower_label = "S1" if lower_kind == "sacrum" else lower
        if lower_kind == "sacrum":
            longitudinal_anterior.append((f"{upper}-{lower_label}", ((upper, "body_anterior_inferior"), ("sacrum", "promontory"))))
            longitudinal_posterior.append((f"{upper}-{lower_label}", ((upper, "body_posterior_inferior"), ("sacrum", nearest_to(upper, "body_posterior_inferior", {"A": (0.3, 0.8), "M": (0.0, 0.2)})))))
            interspinous.append((f"{upper}-{lower_label}", ((upper, "spinous_inferior"), ("sacrum", "posterior_superior"))))
            supraspinous.append((f"{upper}-{lower_label}", ((upper, "spinous"), ("sacrum", "posterior_superior"))))
        else:
            longitudinal_anterior.append((f"{upper}-{lower}", ((upper, "body_anterior_inferior"), (lower, nearest_to(upper, "body_anterior_inferior", {"A": (0.55, 1.0)})))))
            longitudinal_posterior.append((f"{upper}-{lower}", ((upper, "body_posterior_inferior"), (lower, nearest_to(upper, "body_posterior_inferior", {"A": (0.45, 0.8), "M": (0.0, 0.2)})))))
            interspinous.append((f"{upper}-{lower}", ((upper, "spinous_inferior"), (lower, "spinous_superior"))))
            supraspinous.append((f"{upper}-{lower}", ((upper, "spinous"), (lower, "spinous"))))
    entries += _bundle("anterior_longitudinal", "Anterior longitudinal ligament", "ligament_spine", "longitudinal", MIDLINE, longitudinal_anterior)
    entries += _bundle("posterior_longitudinal", "Posterior longitudinal ligament", "ligament_spine", "longitudinal", MIDLINE, longitudinal_posterior)
    entries += _bundle("interspinous", "Interspinous ligament", "ligament_spine", "interspinous", MIDLINE, interspinous)
    entries += _bundle("supraspinous", "Supraspinous ligament", "ligament_spine", "supraspinous", MIDLINE, supraspinous)

    for side in ("L", "R"):
        flavum = []
        intertransverse = []
        capsular = []
        for upper, lower, lower_kind in segments:
            if lower_kind == "sacrum":
                flavum.append((f"{upper}-S1", ((upper, "lamina_inferior"), ("sacrum", nearest_to(upper, "lamina_inferior", {"A": (0.0, 0.6)})))))
                capsular.append((f"{upper}-S1", ((upper, facet("sacrum")), ("sacrum", nearest_to(upper, facet("sacrum"), {"A": (0.0, 0.6), "M": (0.1, 0.5)})))))
                intertransverse.append((f"{upper}-S1", ((upper, "transverse"), ("sacrum", "lateral_upper"))))
            else:
                flavum.append((f"{upper}-{lower}", ((upper, "lamina_inferior"), (lower, nearest_to(upper, "lamina_inferior", {"A": (0.05, 0.5)})))))
                capsular.append((f"{upper}-{lower}", ((upper, facet(lower)), (lower, nearest_to(upper, facet(lower), FACET_BANDS)))))
                intertransverse.append((f"{upper}-{lower}", ((upper, "transverse"), (lower, "transverse"))))
        entries += _bundle("ligamentum_flavum", "Ligamentum flavum", "ligament_spine", "flavum", side, flavum)
        entries += _bundle("facet_capsule", "Facet joint capsule", "ligament_spine", "capsular", side, capsular)
        entries += _bundle("intertransverse", "Intertransverse ligament", "ligament_spine", "intertransverse", side, intertransverse)
        entries += _bundle(
            "iliolumbar", "Iliolumbar ligament", "ligament_pelvis", "iliolumbar", side,
            [
                ("L5", (("L5", "transverse"), (f"ilium {side}", "iliac_tuberosity"))),
                ("L4", (("L4", "transverse"), (f"ilium {side}", "crest_posterior"))),
            ],
        )
    return entries


def pelvic_ligament_strings():
    entries = []
    for side in ("L", "R"):
        ilium = f"ilium {side}"
        entries += _bundle(
            "sacroiliac_anterior", "Anterior sacroiliac ligament", "ligament_pelvis", "sacroiliac", side,
            [("upper", (("sacrum", "ala"), (ilium, auricular("ilium", "sacrum", "anterior"))))],
        )
        entries += _bundle(
            "sacroiliac_interosseous", "Interosseous sacroiliac ligament", "ligament_pelvis", "sacroiliac", side,
            [
                ("upper", (("sacrum", auricular("sacrum", ilium, "upper")), (ilium, auricular("ilium", "sacrum", "upper")))),
                ("lower", (("sacrum", auricular("sacrum", ilium, "lower")), (ilium, auricular("ilium", "sacrum", "lower")))),
            ],
        )
        entries += _bundle(
            "sacroiliac_posterior", "Posterior sacroiliac ligament", "ligament_pelvis", "sacroiliac", side,
            [("short", (("sacrum", "posterolateral"), (ilium, "psis"))), ("long", (("sacrum", "posterolateral_lower"), (ilium, "psis")))],
        )
        entries += _bundle(
            "sacrotuberous", "Sacrotuberous ligament", "ligament_pelvis", "sacropelvic", side,
            [("sacrum", (("sacrum", "posterolateral_lower"), (ilium, "ischial_tuberosity"))), ("PSIS", ((ilium, "psis"), (ilium, "ischial_tuberosity")))],
        )
        entries += _bundle(
            "sacrospinous", "Sacrospinous ligament", "ligament_pelvis", "sacropelvic", side,
            [("sacrum", (("sacrum", "inferolateral"), (ilium, "ischial_spine")))],
        )
        femur = f"femur {side}"
        entries += _bundle(
            "iliofemoral", "Iliofemoral ligament", "ligament_hip", "hip_capsule", side,
            [("lateral band", ((ilium, "aiis"), (femur, "neck_anterior"))), ("medial band", ((ilium, acetabulum(side, "anterior")), (femur, "neck_inferior")))],
        )
        entries += _bundle(
            "pubofemoral", "Pubofemoral ligament", "ligament_hip", "hip_capsule", side,
            [("eminence", ((ilium, "iliopubic_eminence"), (femur, "neck_inferior")))],
        )
        entries += _bundle(
            "ischiofemoral", "Ischiofemoral ligament", "ligament_hip", "hip_capsule", side,
            [("posterior rim", ((ilium, acetabulum(side, "posterior")), (femur, "neck_posterior")))],
        )
        entries += _bundle(
            "ligamentum_teres", "Ligament of the femoral head", "ligament_hip", "hip_capsule", side,
            [("fovea", ((ilium, acetabulum(side, "center")), (femur, femoral_head(side))))],
        )
    entries += _bundle(
        "pubic_symphysis", "Pubic symphysis ligaments", "ligament_pelvis", "pubic", MIDLINE,
        [
            ("superior", (("ilium L", "pubic_symphysis", "L"), ("ilium R", "pubic_symphysis", "R"))),
            ("arcuate", (("ilium L", "pubic_body_inferior", "L"), ("ilium R", "pubic_body_inferior", "R"))),
        ],
    )
    return entries


def costal_ligament_strings():
    entries = []
    for side in ("L", "R"):
        radiate = []
        costotransverse = []
        for level in range(1, 13):
            rib = _rib(level, side)
            vertebra = f"T{level:02d}"
            radiate.append((f"rib {level}", ((rib, rib_head(level, side)), (vertebra, nearest_to(rib, rib_head(level, side))))))
            tubercle = rib_tubercle(level, side)
            costotransverse.append((f"rib {level}", ((rib, tubercle), (vertebra, nearest_to(rib, tubercle, {"A": (0.0, 0.7)})))))
        entries += _bundle("radiate", "Radiate (costovertebral) ligament", "ligament_costal", "costal", side, radiate)
        entries += _bundle("costotransverse", "Costotransverse ligament", "ligament_costal", "costal", side, costotransverse)
    return entries


def costal_cartilage_strings():
    """Ribs 1-7 to the sternum; ribs 8-10 to the cartilage (bony tip) of the rib above."""
    entries = []
    for side in ("L", "R"):
        sternal = []
        for level in range(1, 8):
            rib = _rib(level, side)
            sternal.append((f"rib {level}", ((rib, "anterior"), ("strenum", nearest_to(rib, "anterior")))))
        entries += _bundle("costal_cartilage_sternal", "Costal cartilage (sternal)", "costal_cartilage", "cartilage", side, sternal)
        interchondral = [
            (f"rib {level}", ((_rib(level, side), "anterior"), (_rib(level - 1, side), nearest_to(_rib(level, side), "anterior"))))
            for level in range(8, 11)
        ]
        entries += _bundle("costal_cartilage_interchondral", "Costal cartilage (interchondral)", "costal_cartilage", "cartilage", side, interchondral)
    return entries


def ligament_strings():
    """Every ligament and disc string, in build order."""
    return (
        disc_strings()
        + spinal_ligament_strings()
        + pelvic_ligament_strings()
        + costal_ligament_strings()
        + costal_cartilage_strings()
    )

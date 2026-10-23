"""Muscle attachment atlas for the trunk, pelvis, and proximal femur (bpy-free).

The scene skeleton contains T01-T12, L1-L5, ribs 1-12, the sternum, sacrum, both
hip bones (named `ilium`), and the proximal femurs. Only muscles whose origin and
insertion both lie on those bones are listed; see MUSCLE_ATTACHMENTS.md for the
muscles that were left out and why.

Each muscle produces one string per bony attachment. A string is a polyline of
attachment references `(bone, landmark)` from origin to insertion, optionally
through via points that keep it on the anatomical side of a bone.

Landmarks are geometric approximations: the extreme vertex of a bone along a
direction in the body frame (anterior, lateral, superior), optionally restricted to
bands of the bone's normalized extent along axes A, L, or S (0 = posterior,
medial/side-opposite, or inferior; 1 = anterior, lateral, or superior). "Lateral" points toward the
string's side; midline muscles use no lateral component.
"""

THORACIC = tuple(f"T{level:02d}" for level in range(1, 13))
LUMBAR = tuple(f"L{level}" for level in range(1, 6))
SPINE = THORACIC + LUMBAR

SIDES = ("L", "R")
MIDLINE = "M"

# landmark key -> (direction (anterior, lateral, superior), bands {axis: (low, high)})
LANDMARKS = {
    "vertebra": {
        "spinous": ((-1.0, 0.0, -0.3), {}),
        "transverse": ((-0.3, 1.0, 0.0), {}),
        "lamina": ((-1.0, 0.0, 0.0), {"L": (0.55, 0.65)}),
        "articular": ((-0.5, 0.0, 1.0), {"L": (0.6, 0.8)}),
        "body_anterolateral": ((0.7, 0.7, 0.0), {}),
    },
    "rib": {
        "tubercle": ((-1.0, 0.0, 0.0), {"L": (0.0, 0.3)}),
        "angle": ((-1.0, 0.25, 0.0), {}),
        "lateral": ((0.0, 1.0, 0.0), {}),
        "anterolateral": ((0.6, 0.8, 0.0), {}),
        "anterior": ((1.0, 0.0, 0.0), {}),
    },
    "sternum": {
        "manubrium": ((0.0, 0.0, 1.0), {}),
        "xiphoid": ((0.0, 0.0, -1.0), {}),
        "body_posterior_lower": ((-1.0, 0.0, 0.0), {"S": (0.1, 0.4)}),
    },
    "sacrum": {
        "posterior": ((-1.0, 0.0, 0.0), {}),
        "posterolateral": ((-0.7, 0.7, 0.0), {"S": (0.3, 0.8)}),
        "anterolateral": ((0.6, 0.8, 0.0), {"S": (0.3, 0.7)}),
        "ala": ((0.5, 0.8, 0.5), {"S": (0.7, 1.0)}),
        "inferolateral": ((0.0, 1.0, -0.3), {"S": (0.15, 0.4)}),
        "apex": ((0.0, 0.0, -1.0), {}),
    },
    "ilium": {
        "crest_top": ((0.0, 0.2, 1.0), {}),
        "crest_posterior": ((-1.0, 0.0, 0.5), {"S": (0.7, 1.0)}),
        "crest_anterior": ((0.5, 0.5, 0.7), {}),
        "asis": ((0.8, 0.0, 0.6), {}),
        "psis": ((-0.8, -0.3, 0.5), {}),
        "gluteal_posterior": ((-1.0, 0.4, 0.0), {"S": (0.55, 0.85)}),
        "gluteal_middle": ((0.0, 1.0, 0.2), {"S": (0.65, 0.95)}),
        "gluteal_anterior_low": ((0.2, 1.0, 0.0), {"S": (0.5, 0.65)}),
        "iliac_fossa": ((0.0, -1.0, 0.0), {"S": (0.72, 0.9), "A": (0.4, 0.7)}),
        "iliopubic_eminence": ((1.0, 0.0, 0.5), {"S": (0.3, 0.45), "L": (0.4, 0.55)}),
        "pecten_pubis": ((0.6, 0.0, 0.6), {"S": (0.25, 0.45), "L": (0.15, 0.35)}),
        "pubic_tubercle": ((1.0, 0.0, -0.2), {"S": (0.1, 0.4), "L": (0.1, 0.25)}),
        "pubic_symphysis": ((0.3, -1.0, 0.0), {"S": (0.05, 0.35)}),
        "pubic_body_inferior": ((0.3, -0.6, -0.8), {}),
        "ischial_spine": ((-0.5, -0.8, 0.0), {"S": (0.25, 0.5)}),
        "ischial_tuberosity": ((-0.5, 0.2, -0.8), {}),
    },
    "femur": {
        "greater_trochanter": ((0.0, 1.0, 0.3), {"S": (0.7, 1.0)}),
        "trochanter_superior": ((0.0, 0.6, 0.8), {"S": (0.75, 1.0)}),
        "trochanter_anterior": ((0.6, 0.8, 0.2), {"S": (0.7, 1.0)}),
        "trochanteric_fossa": ((-0.8, 0.2, 0.5), {"S": (0.7, 0.95)}),
        "intertrochanteric_crest": ((-1.0, 0.3, 0.0), {"S": (0.55, 0.8)}),
        "lesser_trochanter": ((-0.6, -0.8, 0.0), {"S": (0.3, 0.6), "L": (0.3, 0.6)}),
        "pectineal_line": ((-1.0, -0.3, 0.0), {"S": (0.15, 0.35), "L": (0.3, 0.6)}),
        "gluteal_tuberosity": ((-0.7, 0.7, 0.0), {"S": (0.25, 0.5)}),
        "linea_aspera_proximal": ((-1.0, 0.0, 0.0), {"S": (0.0, 0.2)}),
    },
}

# group key -> (label, RGBA viewport color)
GROUPS = {
    "erector_spinae": ("Erector spinae", (0.85, 0.2, 0.2, 1.0)),
    "transversospinalis": ("Transversospinalis", (0.95, 0.55, 0.15, 1.0)),
    "deep_back": ("Deep segmental back", (0.75, 0.6, 0.3, 1.0)),
    "abdominal": ("Abdominal wall", (0.95, 0.85, 0.2, 1.0)),
    "thoracic": ("Thoracic wall", (0.9, 0.4, 0.7, 1.0)),
    "hip_flexor": ("Hip flexors", (0.3, 0.8, 0.35, 1.0)),
    "gluteal": ("Gluteal", (0.25, 0.5, 0.95, 1.0)),
    "deep_hip": ("Deep hip rotators and adductors", (0.2, 0.8, 0.85, 1.0)),
    "pelvic_floor": ("Pelvic floor", (0.6, 0.35, 0.9, 1.0)),
}


def bone_kind(bone):
    if bone in SPINE:
        return "vertebra"
    if " rib " in f" {bone} ":
        return "rib"
    for kind in ("sternum", "sacrum", "ilium", "femur"):
        if bone.startswith(kind) or (kind == "sternum" and bone == "strenum"):
            return kind
    raise ValueError(f"Unknown bone kind for {bone}")


def _rib(level, side):
    return f"T{level:02d} rib {side}"


def _string(*refs, via=()):
    """refs: origin and insertion; via: points between them."""
    origin, insertion = refs
    return (origin,) + tuple(via) + (insertion,)


def _paired_muscles():
    """Yield (key, label, group, builder(side) -> [(label, points)])."""

    def iliocostalis_lumborum(side):
        return [
            (f"{_rib(r, side)}", _string((f"ilium {side}", "crest_posterior"), (_rib(r, side), "angle")))
            for r in range(7, 13)
        ]

    def iliocostalis_thoracis(side):
        return [
            (f"{_rib(r, side)} -> {_rib(r - 6, side)}", _string((_rib(r, side), "angle"), (_rib(r - 6, side), "angle")))
            for r in range(7, 13)
        ]

    def longissimus_thoracis(side):
        return [(f"{v} transverse", _string(("sacrum", "posterolateral"), (v, "transverse"))) for v in SPINE]

    def spinalis_thoracis(side):
        origins = {1: "L2", 2: "L2", 3: "L1", 4: "L1", 5: "T12", 6: "T12", 7: "T11", 8: "T11"}
        return [
            (f"{origin} -> T{insertion:02d}", _string((origin, "spinous"), (f"T{insertion:02d}", "spinous")))
            for insertion, origin in origins.items()
        ]

    def semispinalis_thoracis(side):
        return [
            (f"T{o:02d} -> T{o - 6:02d}", _string((f"T{o:02d}", "transverse"), (f"T{o - 6:02d}", "spinous")))
            for o in range(7, 11)
        ]

    def multifidus(side):
        strings = []
        for index in range(3, len(SPINE)):
            origin = SPINE[index]
            landmark = "articular" if origin in LUMBAR else "transverse"
            strings.append((f"{origin} -> {SPINE[index - 3]}", _string((origin, landmark), (SPINE[index - 3], "spinous"))))
        strings.append(("sacrum -> L4", _string(("sacrum", "posterolateral"), ("L4", "spinous"))))
        return strings

    def rotatores_brevis(side):
        return [
            (f"{THORACIC[i]} -> {THORACIC[i - 1]}", _string((THORACIC[i], "transverse"), (THORACIC[i - 1], "lamina")))
            for i in range(1, 12)
        ]

    def rotatores_longus(side):
        return [
            (f"{THORACIC[i]} -> {THORACIC[i - 2]}", _string((THORACIC[i], "transverse"), (THORACIC[i - 2], "spinous")))
            for i in range(2, 12)
        ]

    def intertransversarii_lumborum(side):
        levels = ("T12",) + LUMBAR
        return [
            (f"{levels[i]} -> {levels[i + 1]}", _string((levels[i], "transverse"), (levels[i + 1], "transverse")))
            for i in range(len(levels) - 1)
        ]

    def quadratus_lumborum(side):
        crest = (f"ilium {side}", "crest_posterior")
        strings = [("rib 12", _string(crest, (_rib(12, side), "angle")))]
        strings += [(f"{v} transverse", _string(crest, (v, "transverse"))) for v in LUMBAR[:4]]
        return strings

    def rectus_abdominis(side):
        pubis = (f"ilium {side}", "pubic_tubercle")
        strings = [("xiphoid", _string(pubis, ("strenum", "xiphoid")))]
        strings += [(f"rib {r} cartilage", _string(pubis, (_rib(r, side), "anterior"))) for r in (5, 6, 7)]
        return strings

    def external_oblique(side):
        strings = []
        for r in range(5, 13):
            rib_landmark = "anterolateral" if r < 10 else "lateral"
            insertion = (f"ilium {side}", "pubic_tubercle" if r < 9 else "crest_anterior")
            strings.append((f"rib {r}", _string((_rib(r, side), rib_landmark), insertion)))
        return strings

    def internal_oblique(side):
        crest = (f"ilium {side}", "crest_anterior")
        strings = [(f"rib {r}", _string(crest, (_rib(r, side), "anterior"))) for r in (10, 11, 12)]
        strings.append(("pubic crest", _string((f"ilium {side}", "asis"), (f"ilium {side}", "pubic_tubercle"))))
        return strings

    def external_intercostals(side):
        return [
            (f"rib {r} -> {r + 1}", _string((_rib(r, side), "lateral"), (_rib(r + 1, side), "lateral")))
            for r in range(1, 12)
        ]

    def internal_intercostals(side):
        return [
            (f"rib {r} -> {r + 1}", _string((_rib(r, side), "anterolateral"), (_rib(r + 1, side), "anterolateral")))
            for r in range(1, 12)
        ]

    def levatores_costarum(side):
        return [
            (f"T{r:02d} -> rib {r + 1}", _string((f"T{r:02d}", "transverse"), (_rib(r + 1, side), "tubercle")))
            for r in range(1, 12)
        ]

    def serratus_posterior_superior(side):
        origins = {2: "T01", 3: "T01", 4: "T02", 5: "T03"}
        return [
            (f"{origin} -> rib {r}", _string((origin, "spinous"), (_rib(r, side), "angle")))
            for r, origin in origins.items()
        ]

    def serratus_posterior_inferior(side):
        origins = {9: "T11", 10: "T12", 11: "L1", 12: "L2"}
        return [
            (f"{origin} -> rib {r}", _string((origin, "spinous"), (_rib(r, side), "angle")))
            for r, origin in origins.items()
        ]

    def transversus_thoracis(side):
        return [
            (f"rib {r}", _string(("strenum", "body_posterior_lower"), (_rib(r, side), "anterior")))
            for r in range(2, 7)
        ]

    def subcostales(side):
        return [
            (f"rib {r} -> {r + 2}", _string((_rib(r, side), "angle"), (_rib(r + 2, side), "angle")))
            for r in range(8, 11)
        ]

    def psoas_major(side):
        eminence = (f"ilium {side}", "iliopubic_eminence")
        trochanter = (f"femur {side}", "lesser_trochanter")
        return [
            (f"{v}", _string((v, "body_anterolateral"), trochanter, via=(eminence,)))
            for v in ("T12",) + LUMBAR
        ]

    def psoas_minor(side):
        return [(v, _string((v, "body_anterolateral"), (f"ilium {side}", "pecten_pubis"))) for v in ("T12", "L1")]

    def iliacus(side):
        eminence = (f"ilium {side}", "iliopubic_eminence")
        trochanter = (f"femur {side}", "lesser_trochanter")
        return [
            ("iliac fossa", _string((f"ilium {side}", "iliac_fossa"), trochanter, via=(eminence,))),
            ("sacral ala", _string(("sacrum", "ala"), trochanter, via=(eminence,))),
        ]

    def gluteus_maximus(side):
        tuberosity = (f"femur {side}", "gluteal_tuberosity")
        return [
            ("posterior ilium", _string((f"ilium {side}", "psis"), tuberosity)),
            ("sacrum", _string(("sacrum", "posterolateral"), tuberosity)),
            ("coccyx", _string(("sacrum", "inferolateral"), tuberosity)),
        ]

    def gluteus_medius(side):
        trochanter = (f"femur {side}", "greater_trochanter")
        return [
            ("anterior", _string((f"ilium {side}", "crest_anterior"), trochanter)),
            ("middle", _string((f"ilium {side}", "gluteal_middle"), trochanter)),
            ("posterior", _string((f"ilium {side}", "gluteal_posterior"), trochanter)),
        ]

    def gluteus_minimus(side):
        return [("outer ilium", _string((f"ilium {side}", "gluteal_anterior_low"), (f"femur {side}", "trochanter_anterior")))]

    def piriformis(side):
        return [("anterior sacrum", _string(("sacrum", "anterolateral"), (f"femur {side}", "trochanter_superior")))]

    def gemellus_superior(side):
        return [("ischial spine", _string((f"ilium {side}", "ischial_spine"), (f"femur {side}", "trochanteric_fossa")))]

    def gemellus_inferior(side):
        return [
            ("ischial tuberosity", _string((f"ilium {side}", "ischial_tuberosity"), (f"femur {side}", "trochanteric_fossa")))
        ]

    def quadratus_femoris(side):
        return [
            (
                "ischial tuberosity",
                _string((f"ilium {side}", "ischial_tuberosity"), (f"femur {side}", "intertrochanteric_crest")),
            )
        ]

    def pectineus(side):
        return [("pecten pubis", _string((f"ilium {side}", "pecten_pubis"), (f"femur {side}", "pectineal_line")))]

    def adductor_brevis(side):
        return [
            ("pubic body", _string((f"ilium {side}", "pubic_body_inferior"), (f"femur {side}", "linea_aspera_proximal")))
        ]

    def coccygeus(side):
        return [("ischial spine", _string((f"ilium {side}", "ischial_spine"), ("sacrum", "inferolateral")))]

    def levator_ani(side):
        return [
            ("iliococcygeus", _string((f"ilium {side}", "ischial_spine"), ("sacrum", "apex"))),
            ("pubococcygeus", _string((f"ilium {side}", "pubic_symphysis"), ("sacrum", "apex"))),
        ]

    yield "iliocostalis_lumborum", "Iliocostalis lumborum", "erector_spinae", iliocostalis_lumborum
    yield "iliocostalis_thoracis", "Iliocostalis thoracis", "erector_spinae", iliocostalis_thoracis
    yield "longissimus_thoracis", "Longissimus thoracis", "erector_spinae", longissimus_thoracis
    yield "spinalis_thoracis", "Spinalis thoracis", "erector_spinae", spinalis_thoracis
    yield "semispinalis_thoracis", "Semispinalis thoracis", "transversospinalis", semispinalis_thoracis
    yield "multifidus", "Multifidus", "transversospinalis", multifidus
    yield "rotatores_brevis", "Rotatores brevis", "transversospinalis", rotatores_brevis
    yield "rotatores_longus", "Rotatores longus", "transversospinalis", rotatores_longus
    yield "intertransversarii_lumborum", "Intertransversarii lumborum", "deep_back", intertransversarii_lumborum
    yield "quadratus_lumborum", "Quadratus lumborum", "deep_back", quadratus_lumborum
    yield "rectus_abdominis", "Rectus abdominis", "abdominal", rectus_abdominis
    yield "external_oblique", "External oblique", "abdominal", external_oblique
    yield "internal_oblique", "Internal oblique", "abdominal", internal_oblique
    yield "external_intercostals", "External intercostals", "thoracic", external_intercostals
    yield "internal_intercostals", "Internal intercostals", "thoracic", internal_intercostals
    yield "levatores_costarum", "Levatores costarum", "thoracic", levatores_costarum
    yield "serratus_posterior_superior", "Serratus posterior superior", "thoracic", serratus_posterior_superior
    yield "serratus_posterior_inferior", "Serratus posterior inferior", "thoracic", serratus_posterior_inferior
    yield "transversus_thoracis", "Transversus thoracis", "thoracic", transversus_thoracis
    yield "subcostales", "Subcostales", "thoracic", subcostales
    yield "psoas_major", "Psoas major", "hip_flexor", psoas_major
    yield "psoas_minor", "Psoas minor", "hip_flexor", psoas_minor
    yield "iliacus", "Iliacus", "hip_flexor", iliacus
    yield "gluteus_maximus", "Gluteus maximus", "gluteal", gluteus_maximus
    yield "gluteus_medius", "Gluteus medius", "gluteal", gluteus_medius
    yield "gluteus_minimus", "Gluteus minimus", "gluteal", gluteus_minimus
    yield "piriformis", "Piriformis", "deep_hip", piriformis
    yield "gemellus_superior", "Gemellus superior", "deep_hip", gemellus_superior
    yield "gemellus_inferior", "Gemellus inferior", "deep_hip", gemellus_inferior
    yield "quadratus_femoris", "Quadratus femoris", "deep_hip", quadratus_femoris
    yield "pectineus", "Pectineus", "deep_hip", pectineus
    yield "adductor_brevis", "Adductor brevis", "deep_hip", adductor_brevis
    yield "coccygeus", "Coccygeus", "pelvic_floor", coccygeus
    yield "levator_ani", "Levator ani", "pelvic_floor", levator_ani


def _midline_muscles():
    def interspinales_lumborum(_side):
        levels = ("T12",) + LUMBAR
        return [
            (f"{levels[i]} -> {levels[i + 1]}", _string((levels[i], "spinous"), (levels[i + 1], "spinous")))
            for i in range(len(levels) - 1)
        ]

    yield "interspinales_lumborum", "Interspinales lumborum", "deep_back", interspinales_lumborum


def muscle_strings():
    """Return every string as a dict with muscle, label, group, side, index, count, name, and points."""
    strings = []
    for sides, muscles in ((SIDES, _paired_muscles()), ((MIDLINE,), _midline_muscles())):
        for key, label, group, builder in muscles:
            for side in sides:
                built = builder(side)
                for index, (string_label, points) in enumerate(built):
                    strings.append(
                        {
                            "muscle": key,
                            "label": label,
                            "group": group,
                            "side": side,
                            "index": index,
                            "count": len(built),
                            "name": string_label,
                            "points": points,
                        }
                    )
    return strings


def string_key(string):
    return (string["muscle"], string["side"], string["index"])


def landmark_spec(bone, landmark):
    kind = bone_kind(bone)
    try:
        return LANDMARKS[kind][landmark]
    except KeyError:
        raise KeyError(f"No landmark {landmark!r} for {kind} bone {bone}") from None

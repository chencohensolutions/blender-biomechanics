# Muscle Attachment Reference

Reference for the **Muscle Strings** feature (`biomechanics_addon/muscle_atlas.py`).
The scene skeleton contains T01-T12, L1-L5, ribs 1-12 on both sides, the sternum
(`strenum`), sacrum (with coccyx), both hip bones (`ilium L/R`, which include ischium
and pubis), and the proximal femurs. Only muscles whose origin **and** insertion are
both on those bones are modeled. Each string connects bones of the same collection.

## String Model

- One string per bony attachment. A muscle with insertions on six ribs has six
  strings from its origin landmark to each insertion. A later "fabric" model can add
  more strings per muscle without changing measurement or comparison.
- Attachment points are landmark vertices found on Original bones: the extreme
  vertex along a body-frame direction (anterior / lateral / superior), optionally
  limited to a band of the bone. The body frame comes from the skeleton itself
  (sacrum -> T01 is superior; right femur -> left femur is left).
- Rigid strings reuse the same vertex indices on the paired Rigid meshes, so both
  collections measure the same anatomical points.
- Length is the polyline length through the attachment vertices in world space, in
  millimeters. Via points (psoas and iliacus over the iliopubic eminence) keep a
  string on the anatomical side of the pelvic brim. Strings do not wrap around bone
  surfaces otherwise; a straight segment can pass through bone.
- Landmarks are geometric approximations of anatomical sites, not segmentations.
  Treat absolute lengths as approximate; Original-versus-Rigid differences use
  identical points and are the reliable measurement.

## Modeled Muscles

35 muscles; 361 strings per collection (34 bilateral muscles plus interspinales on
the midline).

| Group | Muscle | Anatomical origin -> insertion | Strings per side |
|---|---|---|---:|
| Erector spinae | Iliocostalis lumborum | posterior iliac crest -> angles of ribs 7-12 | 6 |
| | Iliocostalis thoracis | angles of ribs 7-12 -> angles of ribs 1-6 | 6 |
| | Longissimus thoracis | sacrum / lumbar fascia -> transverse processes T1-L5 | 17 |
| | Spinalis thoracis | spinous processes T11-L2 -> spinous processes T1-T8 | 8 |
| Transversospinalis | Semispinalis thoracis | transverse processes T7-T10 -> spinous processes T1-T4 (C6-C7 absent) | 4 |
| | Multifidus | thoracic transverse / lumbar mammillary processes, sacrum -> spinous process 3 levels up | 15 |
| | Rotatores brevis | transverse process -> lamina one level up (T1-T12) | 11 |
| | Rotatores longus | transverse process -> spinous process two levels up | 10 |
| Deep segmental back | Interspinales lumborum (midline) | spinous process -> next spinous process, T12-L5 | 5 |
| | Intertransversarii lumborum | transverse process -> next transverse process, T12-L5 | 5 |
| | Quadratus lumborum | posterior iliac crest -> rib 12, transverse processes L1-L4 | 5 |
| Abdominal wall | Rectus abdominis | pubic crest -> xiphoid, costal cartilages 5-7 | 4 |
| | External oblique | outer ribs 5-12 -> iliac crest (ribs 9-12), pubic tubercle via aponeurosis (ribs 5-8) | 8 |
| | Internal oblique | anterior iliac crest -> ribs 10-12; ASIS -> pubic crest (conjoint tendon) | 4 |
| Thoracic wall | External intercostals | rib n -> rib n+1, lateral | 11 |
| | Internal intercostals | rib n -> rib n+1, anterolateral | 11 |
| | Levatores costarum | transverse process T1-T11 -> rib below, near tubercle (C7 absent) | 11 |
| | Serratus posterior superior | spinous processes T1-T3 -> ribs 2-5 (C7 absent) | 4 |
| | Serratus posterior inferior | spinous processes T11-L2 -> ribs 9-12 | 4 |
| | Transversus thoracis | posterior lower sternum -> costal cartilages 2-6 | 5 |
| | Subcostales | rib angle -> rib angle two below, ribs 8-12 | 3 |
| Hip flexors | Psoas major | bodies T12-L5 -> lesser trochanter, via iliopubic eminence | 6 |
| | Psoas minor | bodies T12-L1 -> pecten pubis | 2 |
| | Iliacus | iliac fossa, sacral ala -> lesser trochanter, via iliopubic eminence | 2 |
| Gluteal | Gluteus maximus | posterior ilium, sacrum, coccyx -> gluteal tuberosity | 3 |
| | Gluteus medius | outer ilium (anterior, middle, posterior) -> greater trochanter | 3 |
| | Gluteus minimus | lower outer ilium -> anterior greater trochanter | 1 |
| Deep hip | Piriformis | anterior sacrum -> superior greater trochanter | 1 |
| | Gemellus superior | ischial spine -> trochanteric fossa | 1 |
| | Gemellus inferior | ischial tuberosity -> trochanteric fossa | 1 |
| | Quadratus femoris | ischial tuberosity -> intertrochanteric crest | 1 |
| | Pectineus | pecten pubis -> pectineal line of femur | 1 |
| | Adductor brevis | inferior pubic body -> proximal linea aspera (lowest posterior point of the partial femur) | 1 |
| Pelvic floor | Coccygeus | ischial spine -> lateral lower sacrum / coccyx | 1 |
| | Levator ani | ischial spine (iliococcygeus), pubis (pubococcygeus) -> coccyx | 2 |

## Muscles Not Modeled

| Reason | Muscles |
|---|---|
| Attach to the skull, cervical spine, scapula, clavicle, or humerus (not in the scene) | trapezius, latissimus dorsi, rhomboids, levator scapulae, serratus anterior, pectoralis major and minor, subclavius, scalenes, sternocleidomastoid, splenius, semispinalis cervicis/capitis, longissimus cervicis/capitis, iliocostalis cervicis, spinalis cervicis, longus colli |
| Insert on the tibia, fibula, patella, or distal femur (not in the scene) | rectus femoris, sartorius, tensor fasciae latae (iliotibial tract), gracilis, semitendinosus, semimembranosus, biceps femoris, adductor longus, adductor magnus, vastus group |
| No bony insertion (tendon, membrane, or aponeurosis only) | diaphragm (central tendon), transversus abdominis (linea alba), pyramidalis, obturator internus and externus (obturator membrane origin cannot be located by extreme-point landmarks) |
| Limb, head, and neck muscles | all remaining muscles of the arms, hands, legs, feet, face, jaw, tongue, larynx, and pharynx |

Adding bones (for example the scapula or cervical spine) and their landmarks in
`LANDMARKS` lets these muscles be added as further entries in `muscle_atlas.py`.

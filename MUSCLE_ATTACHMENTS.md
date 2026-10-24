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
  millimeters. Via points (psoas and iliacus over the iliopubic eminence,
  obturator internus around the lesser sciatic notch) keep a string on its
  anatomical path. Strings do not wrap around bone
  surfaces otherwise; a straight segment can pass through bone.
- Optional sheets fill triangles between consecutive strings of a muscle (in the
  order listed below), so area change can be seen as well as fiber length change.
  Fan-shaped muscles (one origin, several insertions) give skinny triangles whose
  area is sensitive to the angle between neighboring strings.
- Landmarks are geometric approximations of anatomical sites, not segmentations.
  Treat absolute lengths as approximate; Original-versus-Rigid differences use
  identical points and are the reliable measurement.

## Modeled Muscles

44 muscles; 441 strings per collection (42 bilateral muscles plus interspinales
lumborum and thoracis on the midline).

| Group | Muscle | Anatomical origin -> insertion | Strings per side |
|---|---|---|---:|
| Erector spinae | Iliocostalis lumborum | posterior iliac crest -> angles of ribs 7-12 | 6 |
| | Iliocostalis lumborum (lumbar part) | posterior iliac crest / PSIS -> transverse processes L1-L4 | 4 |
| | Iliocostalis thoracis | angles of ribs 7-12 -> angles of ribs 1-6 | 6 |
| | Longissimus thoracis | sacrum / lumbar fascia -> transverse processes T1-L5, ribs 3-12 near the tubercle | 27 |
| | Spinalis thoracis | spinous processes T11-L2 -> spinous processes T1-T8 | 8 |
| Transversospinalis | Semispinalis thoracis | transverse processes T7-T10 -> spinous processes T1-T4 (C6-C7 absent) | 4 |
| | Multifidus | thoracic transverse / lumbar mammillary processes, sacrum -> spinous process 3 levels up | 15 |
| | Rotatores brevis | transverse process -> lamina one level up (T1-T12) | 11 |
| | Rotatores longus | transverse process -> spinous process two levels up | 10 |
| | Rotatores lumborum | transverse process -> lamina one level up (T12-L5) | 5 |
| Deep segmental back | Interspinales lumborum (midline) | spinous process -> next spinous process, T12-L5 | 5 |
| | Interspinales thoracis (midline) | spinous processes T1-T2 and T11-T12 | 2 |
| | Intertransversarii lumborum | transverse process -> next transverse process, T12-L5 | 5 |
| | Intertransversarii thoracis | transverse processes T10-T11, T11-T12 | 2 |
| | Quadratus lumborum | posterior iliac crest -> rib 12, transverse processes L1-L4 | 5 |
| Abdominal wall | Rectus abdominis | pubic crest -> xiphoid, costal cartilages 5-7 | 4 |
| | External oblique | outer ribs 5-12 -> iliac crest (ribs 9-12), pubic tubercle via aponeurosis (ribs 5-8) | 8 |
| | Internal oblique | anterior iliac crest -> ribs 10-12; ASIS -> pubic crest (conjoint tendon) | 4 |
| | Transversus abdominis | iliac crest, ASIS -> pecten pubis (conjoint tendon); fibers to the linea alba have no bone | 2 |
| Thoracic wall | External intercostals | rib n -> rib n+1, lateral | 11 |
| | Internal intercostals | rib n -> rib n+1, anterolateral | 11 |
| | Innermost intercostals | rib n -> rib n+1, posterolateral (ribs 2-12) | 10 |
| | Levatores costarum breves | transverse process T1-T11 -> rib below, near tubercle (C7 absent) | 11 |
| | Levatores costarum longi | transverse process T7-T10 -> rib two below, near tubercle | 4 |
| | Serratus posterior superior | spinous processes T1-T3 -> ribs 2-5 (C7 absent) | 4 |
| | Serratus posterior inferior | spinous processes T11-L2 -> ribs 9-12 | 4 |
| | Transversus thoracis | posterior lower sternum -> costal cartilages 2-6 | 5 |
| | Subcostales | rib angle -> rib angle two below, ribs 8-12 | 3 |
| Hip flexors | Psoas major | bodies T12-L5 -> lesser trochanter, via iliopubic eminence | 6 |
| | Psoas minor | bodies T12-L1 -> pecten pubis | 2 |
| | Iliacus | iliac fossa, sacral ala -> lesser trochanter, via iliopubic eminence | 2 |
| Gluteal | Gluteus maximus | posterior ilium, sacrum, coccyx -> gluteal tuberosity (IT-tract part not modeled) | 3 |
| | Gluteus medius | outer ilium (anterior, middle, posterior) -> greater trochanter | 3 |
| | Gluteus minimus | lower outer ilium -> anterior greater trochanter | 1 |
| Deep hip | Piriformis | anterior sacrum -> superior greater trochanter | 1 |
| | Gemellus superior | ischial spine -> trochanteric fossa | 1 |
| | Gemellus inferior | ischial tuberosity -> trochanteric fossa | 1 |
| | Obturator internus | inner obturator margin -> trochanteric fossa, via lesser sciatic notch | 1 |
| | Obturator externus | outer obturator margin -> trochanteric fossa | 1 |
| | Quadratus femoris | ischial tuberosity -> intertrochanteric crest | 1 |
| | Pectineus | pecten pubis -> pectineal line of femur | 1 |
| | Adductor brevis | inferior pubic body -> proximal linea aspera (lowest posterior point of the partial femur) | 1 |
| Pelvic floor | Coccygeus | ischial spine -> lateral lower sacrum / coccyx | 1 |
| | Levator ani | ischial spine (iliococcygeus), pubis (pubococcygeus) -> coccyx | 2 |

## Full Muscle Catalog

Named skeletal muscles of the body by region (paired muscles listed once). Status:

- **Modeled**: both ends are on bones in the scene; strings are built.
- **Missing bone**: at least one attachment is on a bone not in the scene (named).
  These are not placed; no substitute attachment points are invented.
- **No bony end**: one end is soft tissue only (tendon, raphe, membrane, skin,
  organ), so a bone-to-bone string cannot represent it.

### Head

| Muscles | Status |
|---|---|
| Occipitofrontalis, temporoparietalis, procerus, nasalis, depressor septi nasi, orbicularis oculi, corrugator supercilii, depressor supercilii, levator labii superioris alaeque nasi, levator labii superioris, zygomaticus major and minor, levator anguli oris, risorius, orbicularis oris, depressor anguli oris, depressor labii inferioris, mentalis, buccinator, auricularis anterior/superior/posterior | Missing bone (skull, mandible) |
| Masseter, temporalis, medial pterygoid, lateral pterygoid | Missing bone (skull, mandible) |
| Superior, inferior, medial, lateral rectus; superior and inferior oblique; levator palpebrae superioris | Missing bone (skull, orbit) |
| Genioglossus, hyoglossus, styloglossus, palatoglossus; superior and inferior longitudinal, transverse, and vertical tongue muscles | Missing bone (mandible, hyoid, skull) / no bony end |
| Tensor veli palatini, levator veli palatini, musculus uvulae, palatopharyngeus | Missing bone (skull) / no bony end |
| Superior, middle, inferior pharyngeal constrictors; stylopharyngeus, salpingopharyngeus | Missing bone (skull, hyoid) / no bony end |
| Cricothyroid, posterior and lateral cricoarytenoid, transverse and oblique arytenoid, thyroarytenoid, vocalis, aryepiglotticus | No bony end (laryngeal cartilages) |
| Tensor tympani, stapedius | Missing bone (middle ear ossicles) |

### Neck

| Muscles | Status |
|---|---|
| Platysma | No bony end (skin) and missing bone (mandible) |
| Sternocleidomastoid | Missing bone (skull, clavicle); sternal head origin is on the manubrium |
| Digastric, stylohyoid, mylohyoid, geniohyoid | Missing bone (mandible, skull, hyoid) |
| Sternohyoid, sternothyroid | Missing bone (hyoid, thyroid cartilage); origin on the manubrium |
| Thyrohyoid, omohyoid | Missing bone (hyoid, scapula) |
| Anterior, middle, posterior scalenes; scalenus minimus | Missing bone (cervical vertebrae); insertions on ribs 1-2 |
| Longus colli | Missing bone (cervical vertebrae); part arises from T1-T3 bodies |
| Longus capitis, rectus capitis anterior, rectus capitis lateralis | Missing bone (skull, atlas) |
| Rectus capitis posterior major and minor, obliquus capitis superior and inferior | Missing bone (skull, C1-C2) |

### Back

| Muscles | Status |
|---|---|
| Trapezius | Missing bone (skull, clavicle, scapula); origin on C7-T12 spinous processes |
| Latissimus dorsi | Missing bone (humerus); origin on T7-L5, sacrum, iliac crest, ribs 9-12 |
| Rhomboid major and minor, levator scapulae | Missing bone (scapula, cervical vertebrae) |
| Serratus posterior superior and inferior | Modeled |
| Splenius capitis, splenius cervicis | Missing bone (skull, cervical vertebrae) |
| Iliocostalis lumborum (thoracic and lumbar parts), iliocostalis thoracis | Modeled |
| Iliocostalis cervicis | Missing bone (cervical vertebrae); origin on ribs 3-6 |
| Longissimus thoracis | Modeled |
| Longissimus cervicis, longissimus capitis | Missing bone (cervical vertebrae, skull) |
| Spinalis thoracis | Modeled |
| Spinalis cervicis, spinalis capitis | Missing bone (cervical vertebrae, skull) |
| Semispinalis thoracis | Modeled (insertions on C6-C7 absent) |
| Semispinalis cervicis, semispinalis capitis | Missing bone (cervical vertebrae, skull) |
| Multifidus | Modeled (thoracic, lumbar, sacral parts); cervical part missing bone |
| Rotatores thoracis (brevis, longus), rotatores lumborum | Modeled |
| Rotatores cervicis | Missing bone (cervical vertebrae) |
| Interspinales lumborum, interspinales thoracis | Modeled |
| Interspinales cervicis | Missing bone (cervical vertebrae) |
| Intertransversarii lumborum, intertransversarii thoracis | Modeled |
| Intertransversarii cervicis (anterior, posterior) | Missing bone (cervical vertebrae) |
| Levatores costarum breves and longi | Modeled (C7 origin absent) |

### Thorax

| Muscles | Status |
|---|---|
| External, internal, and innermost intercostals; subcostales; transversus thoracis | Modeled |
| Diaphragm | No bony end (central tendon); origins on xiphoid, ribs 7-12, L1-L3 |
| Pectoralis major | Missing bone (humerus, clavicle); origin on sternum and costal cartilages |
| Pectoralis minor | Missing bone (scapula); origin on ribs 3-5 |
| Subclavius | Missing bone (clavicle); origin on rib 1 |
| Serratus anterior | Missing bone (scapula); origin on ribs 1-8 |

### Abdomen and Posterior Abdominal Wall

| Muscles | Status |
|---|---|
| Rectus abdominis, external oblique, internal oblique, transversus abdominis | Modeled (aponeurotic fibers to the linea alba have no bony end) |
| Pyramidalis | No bony end (linea alba) |
| Cremaster | No bony end (inguinal ligament, spermatic cord) |
| Quadratus lumborum, psoas major, psoas minor, iliacus | Modeled |

### Pelvic Floor and Perineum

| Muscles | Status |
|---|---|
| Coccygeus; levator ani (iliococcygeus, pubococcygeus) | Modeled |
| Puborectalis | No bony end (anorectal sling) |
| External anal sphincter, external urethral sphincter, compressor urethrae, sphincter urethrovaginalis | No bony end |
| Deep and superficial transverse perineal, bulbospongiosus, ischiocavernosus | No bony end (perineal body, erectile tissue) |

### Hip and Gluteal Region

| Muscles | Status |
|---|---|
| Gluteus maximus, medius, minimus; piriformis; gemellus superior and inferior; obturator internus and externus; quadratus femoris | Modeled |
| Tensor fasciae latae | Missing bone (tibia, via iliotibial tract) |

### Thigh

| Muscles | Status |
|---|---|
| Pectineus, adductor brevis | Modeled (adductor brevis to the lowest point of the partial femur) |
| Adductor longus, adductor magnus | Missing bone (middle and distal femur not in the model) |
| Gracilis, sartorius | Missing bone (tibia) |
| Rectus femoris | Missing bone (patella, tibia); origin on the AIIS |
| Vastus lateralis, vastus medialis, vastus intermedius, articularis genus | Missing bone (patella, tibia, distal femur) |
| Biceps femoris (long and short heads), semitendinosus, semimembranosus | Missing bone (tibia, fibula, distal femur); long heads arise from the ischial tuberosity |

### Shoulder and Arm

| Muscles | Status |
|---|---|
| Deltoid, supraspinatus, infraspinatus, teres minor, teres major, subscapularis, coracobrachialis | Missing bone (scapula, clavicle, humerus) |
| Biceps brachii, brachialis, triceps brachii, anconeus | Missing bone (scapula, humerus, radius, ulna) |

### Forearm and Hand

| Muscles | Status |
|---|---|
| Pronator teres, flexor carpi radialis, palmaris longus, flexor carpi ulnaris, flexor digitorum superficialis, flexor digitorum profundus, flexor pollicis longus, pronator quadratus | Missing bone (forearm and hand) |
| Brachioradialis, extensor carpi radialis longus and brevis, extensor digitorum, extensor digiti minimi, extensor carpi ulnaris, supinator, abductor pollicis longus, extensor pollicis brevis and longus, extensor indicis | Missing bone (forearm and hand) |
| Abductor pollicis brevis, flexor pollicis brevis, opponens pollicis, adductor pollicis; abductor digiti minimi, flexor digiti minimi brevis, opponens digiti minimi, palmaris brevis; lumbricals, palmar and dorsal interossei | Missing bone (hand) |

### Leg and Foot

| Muscles | Status |
|---|---|
| Tibialis anterior, extensor hallucis longus, extensor digitorum longus, fibularis tertius | Missing bone (tibia, fibula, foot) |
| Fibularis longus and brevis | Missing bone (fibula, foot) |
| Gastrocnemius, soleus, plantaris, popliteus, flexor hallucis longus, flexor digitorum longus, tibialis posterior | Missing bone (distal femur, tibia, fibula, foot) |
| Extensor digitorum brevis, extensor hallucis brevis; abductor hallucis, flexor digitorum brevis, abductor digiti minimi, quadratus plantae, lumbricals, flexor hallucis brevis, adductor hallucis, flexor digiti minimi brevis, plantar and dorsal interossei | Missing bone (foot) |

Adding bones (for example the scapula, cervical spine, or the rest of the femur) with
their landmarks in `LANDMARKS` lets the "Missing bone" muscles become entries in
`muscle_atlas.py`.

# Biomechanics Add-on Development Plan

This document is the handoff for future coding sessions. It describes the current
behavior, every UI control, the important implementation rules, and the checks that
must continue to pass.

## Goal

Compare two collections containing identical anatomical mesh objects:

- **Original**: the unmodified source model.
- **Rigid**: the same model after its objects were independently moved or rotated.

The user selects one mesh in Original and aligns the entire Original collection so
that mesh coincides with its matching Rigid mesh. The remaining objects then show
their displacement relative to the aligned pair.

## User Workflow

1. Open **3D Viewport > Sidebar > Biomechanics**.
2. Choose the **Original** and **Rigid** collections.
3. Select one mesh object from Original, or its pair in Rigid.
4. Choose a comparison collection: **None**, **Original**, or **Rigid**.
5. Optionally use **Mesh Wireframe**, **Overlay Opacity**, and **Overlay On Top**.
6. Select **Align to Source**.
7. Inspect how nearby original and rigid meshes differ relative to the aligned pair.

## UI Controls

### Original

Collection picker stored in `Scene.biomechanics_source_collection`.

- Contains the model that will be moved by **Align to Source**.
- Defaults to the collection named `Collection` when available.
- A valid selected alignment object must be a mesh directly in this collection, or a
  Rigid mesh, which stands for its unique Original pair.
- Changing the collection refreshes and safely restores the comparison display.

### Rigid

Collection picker stored in `Scene.biomechanics_target_collection`.

- Contains the comparison model and paired mesh objects.
- Defaults to the collection named `rigid` when available.
- Changing the collection refreshes and safely restores the comparison display.

### Comparison Display: None / Original / Rigid

Expanded enum stored in `Scene.biomechanics_transparency_mode`.

- **None** restores both collections to their previous display settings.
- **Original** applies the comparison display to Original.
- **Rigid** applies the comparison display to Rigid.
- Switching directly between Original and Rigid restores the previous collection
  before changing the next collection.
- The mesh pair used for alignment is excluded from the comparison display because
  both paired meshes occupy the same location after alignment.

The property name still contains `transparency` for compatibility with the current
implementation, although the comparison can now be transparent or wireframe.

### Mesh Wireframe

Toggle stored in `Scene.biomechanics_overlay_wireframe`.

- Off: the comparison collection uses the selected opacity.
- On: comparison meshes use `display_type = "WIRE"` instead of transparency.
- The opacity slider is disabled while wireframe is active.
- The selected aligned pair remains in its original display mode.
- Switching this toggle must restore prior alpha, transparency, display type, and
  in-front state before applying the new mode.

### Overlay Opacity

Factor slider stored in `Scene.biomechanics_overlay_opacity`.

- Range: 0.0 to 1.0, displayed as 0% to 100%.
- Default: 0.25 (25% opacity).
- Updates the active transparent comparison collection immediately.
- Disabled when Comparison Display is None or Mesh Wireframe is enabled.
- Transparent mode temporarily changes solid viewports to Object Color mode because
  Blender only displays `Object.color` alpha in that mode.

### Overlay On Top

Toggle stored in `Scene.biomechanics_overlay_on_top`.

- Default: enabled.
- On: comparison meshes use `show_in_front = True`.
- Off: comparison meshes retain their original `show_in_front` values.
- Works with both transparent and wireframe comparison modes.
- Disabled in the UI while Comparison Display is None.
- The selected aligned pair is excluded.

### Align to Source

Operator: `BIOMECHANICS_OT_align_to_source`.

Blender ID: `biomechanics.align_to_source`.

Behavior:

1. Validate that Original and Rigid are selected and different.
2. Validate that the active object is a mesh in Original or Rigid. A Rigid mesh is
   replaced by its unique Original pair (same suffix-removal rule); missing or
   ambiguous Original pairs are rejected.
3. Pair it with one Rigid mesh by removing one Blender numeric suffix such as
   `.001` or `.002` and comparing the remaining names.
4. Reject missing or ambiguous pairs.
5. Validate that paired meshes have equal vertex and polygon counts and at least
   three vertices.
6. Read corresponding vertices in world space.
7. Use a Kabsch/SVD best-fit rigid transform to calculate rotation and translation.
8. Reject meshes whose fitted vertex RMS error exceeds
   `max(target_size * 1e-4, 1e-6)`.
9. Apply the transform to every mesh object in Original as one rigid group.
10. Preserve object scale and all relative spacing in Original.
11. Refresh the comparison display and exclude the newly aligned pair.
12. Report the moved object count. The operator supports Blender Undo.

### Point Correspondence

Panel: **Point Correspondence**. Module: `biomechanics_addon/correspondence.py`.

- **Hover Correspondence** (`biomechanics.toggle_correspondence_hover`) toggles a
  pass-through modal operator. Under the mouse it ray-casts the scene; for a hit on
  an Original or Rigid mesh it finds the unique pair by `base_object_name`, maps the
  hit to the pair with `poly_3d_calc` weights over the shared polygon vertex indices,
  and draws both points, a connecting line, and the world distance in millimeters.
  It changes no scene data. Topology-changing modifiers break the face-index mapping.
- **Find Max Distance Point** (`biomechanics.find_max_correspondence_distance`)
  compares every Original mesh with its Rigid pair vertex-by-vertex in world space,
  ranks pairs by their largest vertex distance, marks the overall maximum in the
  viewport, and lists the top five pairs. Results are in-memory only and cleared by
  the X button, file load, or unregister.
- One `POST_PIXEL` draw handler serves both overlays; it is added on register and
  removed on unregister.
- **Distance Heatmap** (`biomechanics.toggle_distance_heatmap`, module
  `distance_heatmap.py`) writes a temporary point-domain `FLOAT_COLOR` attribute
  `biomechanics_correspondence_distance` to both meshes of every pair (blue near ->
  red far) and switches solid viewports to Attribute color (`color_type = "VERTEX"`).
  **All Bones** scales red to the global maximum; **Per Bone** (`Scene.biomechanics_heatmap_scale`)
  scales to each pair's own maximum, never below 0.01 mm so an aligned pair's
  float noise (~1e-4 mm) stays blue. A depsgraph handler marks a mesh dirty only when
  its `matrix_world` actually differs from the cached one (bone-parented Rigid meshes
  report `is_updated_transform` on every evaluation, so the flag alone causes a
  refresh loop), and a 0.1 s timer recomputes only those pairs from cached local
  vertices (all pairs when the global maximum shifts by more than 10%). Off, unregister, and file load remove the attribute, restore each
  mesh's previous active/render color attribute from `_biomechanics_previous_*`
  mesh properties, and restore viewport color modes. Workbench honors object alpha
  only in Object color mode (verified by render), so while the heatmap is on and
  Mesh Wireframe is off, `refresh_transparency_overlay` hides comparison meshes with
  `hide_set(True)` (prior state in `_biomechanics_previous_hidden`) and leaves the
  viewport color mode to the heatmap. A `POST_VIEW` handler in `distance_heatmap.py`
  redraws those meshes with a custom headlight GPU shader in heatmap colors at
  Overlay Opacity, honoring Overlay On Top (depth test off). Hidden comparison meshes
  are not hit by Hover Correspondence ray casts. Turning the heatmap off reapplies the
  normal transparent comparison display.

### Muscle Strings

Panel: **Muscle Strings**. Modules: `biomechanics_addon/muscle_atlas.py` (bpy-free
attachment data) and `biomechanics_addon/muscle_strings.py`. Anatomical reference and
the list of muscles left out: [MUSCLE_ATTACHMENTS.md](MUSCLE_ATTACHMENTS.md).

- **Build Muscle Strings** (`biomechanics.build_muscle_strings`) detects landmark
  vertices on the Original bones in a body frame derived from the skeleton (sacrum,
  T01, both femurs), then creates one string per muscle attachment in a new
  collection `<collection> muscles` for Original and again for Rigid, reusing the
  same vertex indices on the paired Rigid meshes. Every string connects bones of one
  collection. Rebuilding replaces only collections tagged with
  `biomechanics_muscle_bones`. Supports Undo.
- Strings are 3D poly curves (0.6 mm bevel, material per muscle group) with one Hook
  modifier per control point to its bone, so they stretch when bones move. Each
  string stores muscle, side, index, and `[bone, vertex index]` attachments as JSON in
  `biomechanics_muscle_string`.
- Length is always recomputed from the attachment vertices in world space (polyline
  through via points), in millimeters through the scene unit scale, never from the
  evaluated curve.
- **Hover Muscle Lengths** (`biomechanics.toggle_muscle_hover`) is a pass-through
  modal operator. It projects every visible string to the region and picks the
  nearest within 8 px (scaled by UI scale); the `POST_PIXEL` overlay highlights it
  and shows its length and the matching Original/Rigid string's length and delta.
  Visibility is checked through object and layer-collection flags because
  `Object.visible_get()` returns False from a 3D-view context in Blender 5.x.
- **Length Change Colors** (`biomechanics.toggle_muscle_delta_colors`) colors each
  Original/Rigid twin pair by Rigid minus Original length on a diverging ramp: red
  shorter, gray unchanged, blue longer (sheets: red smaller, blue larger), with a
  21-bin set of shared
  `Biomechanics Muscle Delta NN` materials. **Percent** (strain, default) or **mm**
  is chosen by `Scene.biomechanics_muscle_delta_unit`; `Scene.biomechanics_muscle_delta_range`
  (default 10) maps to full color. It sets each string's material slot to
  `link = "OBJECT"` and `Object.color` (for Object color mode), so the curve data
  keeps its group material; off, Remove, file load, and unregister restore
  `link = "DATA"` and the group color. A 0.25 s timer recolors while active, but only
  when a signature of every Original/Rigid mesh `matrix_world` changes, and it writes
  only changed slots and material colors (any write tags a depsgraph update and
  stalls viewport navigation). Bone objects, attachment vertex coordinates, and
  parsed string/sheet JSON are cached because `bpy.data.objects.get` is slow with
  ~1,500 objects; caches clear on build, file load, and reload. A legend is drawn above the
  distance heatmap legend. Solid viewports in Attribute color mode (Distance
  Heatmap on) do not show string colors.
- **Muscle Sheets** (`biomechanics.toggle_muscle_sheets`, module `muscle_sheets.py`)
  fills triangles between consecutive strings (atlas index order) of each muscle and
  side, in both muscle collections; muscles with one string get no sheet. Vertices
  are the shared attachment vertices, hooked per bone, so sheets stretch with the
  strings. Each sheet stores its attachments and faces in `biomechanics_muscle_sheet`;
  twin sheets have identical face lists, so face i pairs with face i. On is derived
  from sheet objects existing; Build Muscle Strings rebuilds sheets if they existed.
- With Length Change Colors on, each sheet face takes a material slot by its area
  change in percent (slot 0 = group color, then the 21 delta bins, then unpaired)
  scaled by `Scene.biomechanics_muscle_sheet_range` (default 20 %), and the sheet's
  `Object.color` is its total area change, because Object color mode (used by the
  transparent Comparison Display) cannot show per-face colors. A second legend shows
  the sheet scale. Hover falls back to a scene ray cast and reports a sheet
  triangle's areas; string and sheet hover always report Rigid minus Original.
- **Export Muscle Lengths** (`biomechanics.export_muscle_lengths`) writes one CSV row
  per string with both lengths, delta in mm, and delta in percent.
- Landmarks are approximations (extreme vertex along a direction within optional
  bone bands). Absolute lengths are approximate; Original/Rigid deltas use identical
  points. Strings do not wrap bone surfaces except at explicit via points.

### Ligaments & Discs

Module `biomechanics_addon/ligament_atlas.py` (bpy-free); built by
**Build Ligaments & Discs** (`biomechanics.build_muscle_strings` with `kind="ligament"`)
into `<collection> ligaments`, tagged `biomechanics_string_kind = "ligament"`. The
muscle hover, length-change colors, and CSV export cover both kinds; sheets are
muscle-only. Each string stores `compliance`, `unilateral`, and `slack` for the
chain solver. Discs are a central nucleus spring (stiff) plus a 6-point annulus
ring between the facing endplates; ligaments resist tension only.

Landmark resolution (`muscle_strings._LandmarkResolver`) gained: sequential bands
(each normalized over the vertices the earlier bands kept), axis `M` (distance from
the midline), mode `centroid`, inline dict specs with `near` (vertices within
`tolerance_mm` of the closest approach to a named bone: endplates, facets, auricular
surfaces, costal facets, acetabulum) and `nearest_to` (closest vertex to another
bone's landmark). Per-point side overrides allow cross-midline strings (pubic
symphysis).

### Bone Chain

Panel: **Bone Chain**. Modules: `distance_field.py` (bpy-free grid, trilinear
sampling, flood-fill sign), `bone_fields.py` (BVH-based field computation, disk
cache `<Field Cache>/<fingerprint>_<voxel>mm.npz`, surface samples),
`chain_solver.py` (bpy-free XPBD), `chain_interaction.py` (session, target Empty,
handlers, operators).

- **Precompute Collision Fields** computes 1 mm signed distance fields (auto-coarsened
  above 2.5 M cells) in each bone's local frame; the exterior is flood-filled through
  cells more than 1.75 voxels from the surface and grown back, so segmentation holes
  stay closed. Verified on L5: 99.8 % sign agreement with ray parity, max distance
  error 0.46 mm, 2 k-point query with gradient in ~2 ms. Blender blocks during
  computation (about 3 s per vertebra, 20 s for the sacrum/ilia).
- **Start Chain Interaction** needs a Rigid vertebra active. Free bodies are the
  vertebrae above **Base Bone** up to **Top Bone**; the base is fixed. **Base Bone**
  `femurs` (the default) fixes both femurs instead and frees the pelvis: sacrum and
  both hip bones become bodies with hip, sacroiliac, and pubic-symphysis contact
  pairs, so the whole trunk stands on the femoral heads (22 bodies, 540 constraints,
  449 muscle strings in 86 groups, ~120 ms per solve). The pelvis is held together
  and on the femurs by near-rigid zero-length ball joints (compliance 1e-4,
  projected alone after the pair groups): hips at femoral head centers fitted by a
  least-squares sphere to the femur vertices within 6 mm of the acetabulum (radius
  ~22 mm), sacroiliac joints at the interosseous ligament centers, and the pubic
  symphysis at its ligament center; the hip, SI, and pubic contact pairs are dropped
  because contacts fight the joints. Fully rigid joints oscillate at 5 sweeps. Even
  so the trunk's weight sinks the pelvis through the hips (sacrum -42 mm with rigid
  hips alone, -72 mm over 150 solves with all joints), so **Pelvis Hold** (default
  0.005) is a soft 6-DOF pose constraint on the sacrum standing in for the hip and
  balance muscles: with it the sacrum stays within 0.1 mm and 0.4 deg over 150
  solves while the spine above still sags (T01 -57 mm), which is the controller's job.
  The hold is exported in the chain package (`holds`) so training sees it too. Constraints are
  every ligament/disc string whose bones are all in the chain, with rest lengths from
  the pose at start (**Rest Lengths from Original** switches to Original lengths,
  which makes the Rigid pose drift toward the Original equilibrium). Adjacent bones
  form collision pairs with 1,200 surface samples each; a pair's margin is capped
  at its rest clearance so touching bones are not pushed apart at start. Non-chain bones follow
  rigidly when **Attached Bones Follow** is on: ribs and costal cartilages their
  vertebra, skull/mandible/hyoid/C1-C7 T01, clavicles the sternum, scapulae/arms/hands
  T04; bones on the fixed femurs stay.
- **Move Bones** (`biomechanics.toggle_move_bones`) is the primary control: a free
  session with no handle. The depsgraph handler compares every chain bone in the
  update set with the matrix the solver last wrote (`ChainSession.written`); a bone
  that differs was moved by the user, becomes the handle (`set_handle`, which frees
  a pinned top bone while it is held and rebuilds the solver arrays), its matrix
  becomes the pose target, and the solve plus settle run. Toggling off keeps the
  pose. Blender's own undo steps (from the transform operators) restore earlier poses.
- **Start with Target Arrows** is the older explicit mode with a target Empty.
- The target Empty is 12 cm arrows drawn in front; **Select Target** reselects it.
  Grabbing the handle vertebra itself also works: when its matrix differs from the
  one the solver last wrote, the target is moved onto it and the solve runs.
- The handle is pulled toward `Biomechanics Chain Target` by a pose constraint
  (**Handle Compliance**, 1 mm / 0.02 rad per sweep step limit, which prevents
  tunneling). A depsgraph handler solves when the target moves, and a 30 ms timer
  keeps solving while bodies still move more than 0.01 mm (at most 60 solves), so a
  held target settles to equilibrium. **Stop, Keep Pose** / **Stop, Reset**.
- Solver: batched XPBD. Distance constraints are grouped by bone pair and the
  groups are projected in sequence along the chain (direction alternating each
  sweep); contacts are projected per pair the same way. Averaging a body's
  corrections over all of its constraints instead was tried and lets a steady load
  (gravity) win every round. `substeps` rounds of `iterations` sweeps; collision
  samples screened once per substep within band + maximum pose travel + gravity;
  up to 4 deepest contacts per pair direction. Measured: 17 free bones, 289
  constraints, 17 pairs, 5 x 2 sweeps: about 70-90 ms per solve, penetration
  0.2-0.5 mm, zero drift at rest without gravity.
- **Gravity** (toggle + mm/step, default 0.1): every free body is displaced by that
  amount along world -Z at the start of each substep (the XPBD position prediction
  under constant acceleration); ligaments, discs, and contacts carry it. Starting a
  session kicks off settling immediately, and changing gravity restarts it.
  **Pin Top Bone** (default on) fixes the top bone so the chain hangs between base
  and top; unpinned, the thoracic column has nothing above it and droops like a
  cantilever (118 mm at T01 with 0.3 mm/step), which is what a passive spine without
  muscle tone does. With the defaults the mid-thoracic spine sags about 4 mm
  (ligament slack plus disc give) and settles in about 50 solves (under 5 s); lumbar
  bones move under 1 mm, and dragging L3 20 mm forward still moves it 15 mm.
- Solver details that matter: unilateral multipliers are clamped (tension <= 0,
  compression >= 0) rather than reset when slack, and multipliers are warm-started
  across solves (full memory), so a steady load is met by the forces that balanced it
  last round. Within a pair batch each constraint's XPBD step is divided by the
  number of constraints moving the body and that divided value is accumulated as the
  multiplier, so it equals the force actually applied; plain averaging made a
  compliant spring act n times softer. A straight 12-body test stack now holds 0.4
  mm under 0.3 mm/step gravity; a leaning stack still buckles, which is physical (a
  ligamentous spine without muscles buckles under ~90 N).
- **Discs** are a stiff nucleus spring, a rigid compression-only center strut (the
  column carries load axially), six rim springs, and twelve oblique tension-only
  annulus fibers (rim i to rim i +/- 1), which give shear and torsion stiffness.
- **Muscles** are active elements: every muscle string of the Rigid muscles
  collection whose ends map to chain bodies becomes a `MuscleElement`; rib
  attachments ride on their vertebra, and ilium/femur/sternum attachments pull against
  fixed anchor bodies. Each solver round a string pulls its ends together by
  `activation * Muscle Strength` (default 0.05 mm/step per string, forces add up,
  never averaged). Activations are per (muscle, side) group (56 groups, 405 strings
  for the sacrum-T01 chain). **Muscle Tone** sets them uniformly; uniform tone or
  "all extensors" does not hold the spine up (co-contraction compresses, extension on
  a kyphotic thoracic spine tips the rib cage back), so posture is a control problem.
- **Export Chain Package** writes `chain.json` + `chain.npz` (bodies, constraints,
  muscles, pair samples and margins, field file paths) that `chain_package.build_solver`
  turns back into a ChainSolver without bpy. Headless reproduction is exact (T01
  -22.5 mm after 40 passive solves both in Blender and headless, ~70 ms per solve).
- `tools/train_posture.py` optimizes a controller against the package with an
  antithetic evolution strategy over worker processes: `--mode static` (one
  activation vector) or `--mode policy` (tanh network on the observation: per free
  body, offset from rest in cm and rotation vector). Score = top-bone height gain (cm)
  - 0.3 drift - 0.2 strain - penetration over 1 mm - effort. Writes
  `posture_policy.npz`; `posture_policy.py` (bpy-free) holds observation, network,
  and save/load shared with Blender. An episode whose top bone drops by `--fail-cm`
  (default 5 cm) or more ends at once as a failure scored -10 plus 2 x (fraction of
  steps survived), always below any non-failing score. The Bone Chain panel shows the
  top bone's height change and flags FAILED below -5 cm.
- **Posture Controller** toggle (needs an active session) loads the policy file and
  sets activations from the live chain state before every solve; Muscle Tone is
  disabled while it is on. The panel shows mean activation, policy generation, and
  score. **Follow Training** (default on) polls the policy file every second and,
  when the trainer has written a newer best (it saves atomically after every
  generation), restores the chain to its start pose, rebuilds the session with the
  new controller, and lets it play out, so training can be watched live on the bones.
- **Force Colors** toggle tints every muscle string red by its activation and every
  ligament/disc string blue by the force it carries (|multiplier|, scaled to a slowly
  decaying running maximum), refreshed after each solve (about +90 ms per solve for
  ~900 strings). Off, stop, and file load restore the group colors or the length-change
  colors. **Reset Rigid to Original** aligns every Rigid bone exactly onto its
  Original twin (per-bone Kabsch fit), which is the neutral pose.
- Training setup notes: from the neutral pose the passive chain holds at gravity 0.15
  (T01 -0.4 mm after 80 solves), sags 5.5 mm at 0.3 and 16.5 mm at 0.5, and the solver
  diverges at 0.8. Every hand-set activation pattern lowers T01 (muscles mostly
  compress a curved column), so the trainer uses `--perturb-mm/--perturb-deg`
  (random pushes of three bodies at episode start, averaged over `--episodes`) and
  the policy must recover height. The displaced Rigid pose that the file was saved
  with was far less stable than the neutral pose; most of the collapse chased earlier
  came from that pose plus the solver issues above.
- Learned dynamics surrogate (option 4, `tools/train_posture_surrogate.py`, PyTorch in
  `.venv`): `collect` rolls the real solver under smooth random activations and random
  pushes (`PostureEnv` from `train_posture.py`, 40 steps, gravity 0.3) and stores
  `(observation, activations) -> next observation` transitions in
  `dynamics_dataset.npz`; `fit` trains an MLP 217 -> 256 x 3 (SiLU) -> 126 that
  predicts the normalized state delta (`dynamics_model.npz`, with the normalization
  stats); `optimize` trains the posture policy (same 126 -> 32 -> 91 network and file
  as the ES trainer) by Adam through 40 unrolled surrogate steps: mean top-bone height
  capped at `--height-cap` 1 cm, minus 0.2 x activation^2, minus 10 x a soft fail
  indicator below -5 cm; a quarter of each batch starts at rest, the rest from recorded
  non-failed states. The result is scored in the real solver and written to
  `posture_policy.npz` only if it beats the stored score (`--overwrite` to force).
  `iterate` repeats collect (half the episodes under the latest surrogate policy plus
  noise, appended to the dataset), warm-started fit, optimize, evaluate.
- Surrogate findings: held-out per-step error 0.32 mm but 4-12 mm drift over 40
  open-loop steps, and the policy exploits it. With random data only, the surrogate
  predicted +9 cm (impossible) while the real solver scored -8 (failing); with the
  height cap the surrogate predicted +0.7 cm, real -8.2; four on-policy rounds
  (dataset 61,600 transitions, fit error flat at 0.42 normalized / 0.31 mm per step)
  scored -6.6, -7.4, -8.3, -7.4 in the real solver: not improving. The ES policy
  (-4.83, generation 6) remains the best real-solver result and is what
  `posture_policy.npz` holds; `posture_policy_latest.npz` is the last surrogate policy.
  Likely next levers: an ensemble of dynamics models with a disagreement penalty in
  the objective (keeps the policy where the models agree), a longer input history or
  the contact multipliers as hidden state, and training on multi-step rollouts.
- `tools/surrogate_dashboard.py --package <dir>` serves `surrogate_dashboard.html` at
  http://localhost:8765: the policy network with every weight as a line (blue positive,
  red negative, opacity and width by magnitude, threshold slider), node fills from a
  forward pass on the neutral or a pushed state, output activation bars, hover values,
  and charts of the surrogate objective, the dynamics fit error, and the real-solver
  score per round. The trainer writes `surrogate_live.json` atomically (every fit epoch,
  every `--live-every` 10 optimize iterations); the page polls it once a second.
  `.claude/launch.json` has a `surrogate-dashboard` entry for the desktop app preview.
- Settle loop: `SETTLE_THRESHOLD` 0.03 mm of maximum body movement per solve, at most
  `MAX_SETTLE_SOLVES` (150) per kick, 30 ms apart. Each solve blocks the UI for its
  duration, so settling is visibly choppy; the neural surrogate is the planned fix.


### Atlas Completion (scene data, not add-on code)

The scan contains T01-L5, ribs, sternum, sacrum, hip bones, and proximal femurs.
The rest of the skeleton was added from **Z-Anatomy** (`SkeletalSystem100.fbx`,
GitHub LluisV/Z-Anatomy, PC-Version branch), licensed **CC BY-SA 4.0**: derived
bones in this scene carry a `biomechanics_atlas_source` custom property naming the
atlas parts, and any distribution of the scene must keep the attribution and the
share-alike terms. Procedure (ad hoc scripts in this session, not yet an operator):

1. Import the FBX into an `atlas` collection (2,926 objects; the real bones are the
   ones without landmark/attachment suffixes `.i .j .el .er .ol .or`).
2. Similarity fit (Umeyama, scale 0.943) from atlas to scan bone centroids over
   T1-T12, L1-L5, sacrum, both hip bones: RMS 10.5 mm, max 17.9 mm.
3. Femurs: rigid ICP of each atlas femur onto the scan's proximal femur using the
   cached distance field (projection to surface, 40 iterations): 1.9 / 2.1 mm RMS on
   the overlap. The extended femur = scan vertices + atlas vertices below the scan
   cut (z of the scan's lowest femur vertex + 4 mm), joined into one mesh named
   `femur L` / `femur R` (and `.002` twins); the scan-only meshes are kept hidden as
   `femur L partial` etc. Fields and strings must be rebuilt for the new meshes.
4. Patella, tibia, fibula, and foot (27 bones joined) follow each femur's ICP
   transform. Head and neck (skull: 58 atlas parts joined; mandible; hyoid; C1-C7)
   follow the global fit shifted so the C7-T01 gap is 5 mm. Clavicles, scapulae,
   humeri, radii, ulnae, hands (27 bones joined), and costal cartilages 1-10 use the
   global fit. Every new bone exists in both Original and Rigid with identical
   geometry. The `atlas` collection is left hidden in the scene and can be deleted.
5. The scan's sternum is re-seated by rigid ICP onto the registered atlas sternum
   (manubrium + body + xiphoid), so it sits where the costal cartilages end.

Not done: per-region refinement of the shoulder girdle and the neck beyond the
C7 offset, cervical discs/ligaments, and strings for the new bones. The muscle and
ligament atlases were rebuilt after the femur replacement (610 ligament strings).

### CT Bone Extraction (tools/ct, not add-on code)

`tools/ct/` (repo `.venv`, Python 3.13; usage in README) rebuilds bone meshes from a
CT scan. Bone surfaces now come from `segment_nni.py` (learned promptable
segmentation with nnInteractive v1.0 on isotropic crops that `fuse_series.py` fuses
from the axial, coronal and sagittal series of each acquisition). The
previous method (`fuse_planes.py` + `extract_bone.py` threshold/watershed/heuristics +
`resegment_pairs.py` consistency pass) is DEPRECATED for bone surfaces; its invariants
below are tagged **Previous method** and kept for comparison only. Do not extend it.

Resolution truth (corrected 2026-11-01 after the user questioned it): the coronal and
sagittal series are tagged `DERIVED\SECONDARY\MPR`, but the console built them from
the thin 0.625 mm raw reconstruction, not from the exported 3 mm axial slices. Measured
on the data: along the body axis the real sagittal/coronal series have high-frequency
content 0.26 vs 0.14-0.19 for the 3 mm axial resampled onto the same grid, cortical
edge gradient 32 vs 19 HU/px, and the series are aligned to <0.1 mm. ORIGINAL axial
series: 404 (chest, 0.453 x 0.453 x 1 mm, kernel YB), 405/407 (2 mm), 401/501/601
(3 mm, kernel B); 502/503 are byte-identical duplicates of 602/603. Each series is
sharp in two directions and blurred across its slice thickness only, so the three
together resolve every direction twice. `fuse_series.py` fuses them in the Fourier
domain (weights exp(-2 (f_n t)^2) along each series' slice normal) onto a 0.6 mm
isotropic grid: abdomen and pelvis become ~0.6-0.7 mm isotropic instead of 0.68 x 0.68
x 3 mm; the chest is dominated by the 1 mm axial. The earlier statement here that
"fusion never added resolution" was wrong (the previous method's spatial-domain average
simply did not exploit it). Facet and costovertebral joint spaces (0.5-1 mm) remain at
voxel scale. For future scans, request a 0.6-1 mm bone-kernel axial reconstruction of
the whole body (collimation was 0.625 mm, so the raw data supports it).

Why the previous method was replaced: ~15 hand-tuned rules that broke each other;
hollow vertebral bodies where marrow fell below the fill threshold; rib heads left on
vertebrae; independent per-bone extraction that then needed a reconciliation pass.

Current method, `segment_nni.py` (nnInteractive v1.0, DKFZ / Helmholtz Imaging;
Isensee, Rokuss, Krämer et al. 2025, arXiv 2503.08373). Invariants:

- LICENCE: the nnInteractive weights are CC BY-NC-SA 4.0 (non-commercial); the Python
  backend is Apache-2.0. Weights: Hugging Face `nnInteractive/nnInteractive`, folder
  `nnInteractive_v1.0` (411 MB), in `~/.cache/nnInteractive/nnInteractive_v1.0`
  (default `--model-dir`). Package `nninteractive` 2.6.0 in the repo `.venv`, torch
  2.13 on MPS; the tool sets `PYTORCH_ENABLE_MPS_FALLBACK=1` (`adaptive_avg_pool3d` is
  missing on MPS).
- Input is an existing `pipeline.py` run folder (`run.json`, `series/`, `totalseg/`).
  Each bone uses the group whose ORIGINAL axial series contains its guide label
  (`--group-margin-mm` 5; `--group-override BONE=GROUP` forces one). Guide = the
  TotalSegmentator label, else the fitted-atlas label from `--atlas-labels`.
- Per bone: crop around the guide with `--margin-mm` 25. `--planes fused` (default):
  the crop is fused by `fuse_series.py` from the group's axial + coronal + sagittal
  series onto a `--fused-mm` 0.6 mm isotropic grid (about 1 s per bone; the group's
  label grid and all masks live on that grid). `--planes axial`: native axial voxels
  only (the ct2021-nni run). The coarse label is the initial segmentation; then
  `--pos-points` 2 positive prompts deep inside the bone and `--neg-points` 1 negative
  prompt inside each neighbouring bone's label near the bone (`--point-sep-mm` 20);
  keep the largest connected component and fill holes. Masks are cached in
  `RUN/nni_fused/masks/` (`RUN/nni/masks/` for `--planes axial`; `--force` redoes
  them, `--skip-segment` only re-resolves and re-meshes).
- Joint resolution: all bones of a group share one label grid; a voxel claimed by two
  bones goes to the bone whose uncontested core is nearer (`--resolve-with` adds
  existing masks of the same group). No pairwise reconciliation pass is needed.
  Neighbours segmented on different series (L2 on the 1 mm chest series, L3 on the
  3 mm abdomen series) are resolved across groups afterwards: where the finer series'
  bone exists, the coarser series' neighbour gives way (`carve_across_groups`).
- Mesh: 0.5 iso-surface of the Gaussian-smoothed mask (`--smooth-mm` 0.5;
  `--smooth-min-vox`, default 0, was 0.8 for the axial-only run so 3 mm slices did not
  stair-step; the fused grid needs none), then
  `--smooth-iters` 10 Taubin iterations; PLY/STL in Blender metres (LPS mm x 0.001,
  scanner origin kept) plus a JSON per bone in `RUN/nni_fused/meshes/`.
- Measured vs inferred: `RUN/nni_fused/qa/contacts.json` records every touching pair's
  interface and HU. Median HU above `--measured-max-hu` 450 = no visible joint space =
  boundary INFERRED by the model's anatomical prior; below it = measured. Never report
  an inferred boundary as a measured joint gap.
- Blender (`--blender`): import by bone name + `--suffix`, replacing the same-named
  object, into `--collection` (default `bones`), with `ct_run` (`--run-tag`, default
  `ct2021-nni-fused`), `ct_method`, `ct_group`, `ct_planes`, `ct_series`,
  `ct_voxel_mm`, `ct_inferred_boundaries`; `--replace` first removes other runs' objects from the
  collection. Pilots use `--collection bones_nni --suffix _nni` to keep the old bones.
- Bones are never moved, and the atlas never shapes a bone: it is used only for
  naming, ownership (fallback guide label), and QA.
- QA: `register_atlas.py RUN --meshes-dir RUN/nni_fused/meshes ... --out
  RUN/nni_fused/atlas_check --no-cartilage`, `check_overlaps.py RUN/nni_fused --exact
  --samples 3000 --out RUN/nni_fused/qa`,
  and `reference_check.py`, which scores one or more mesh folders against an
  ALREADY-FITTED atlas reference (`<bone>_atlas.ply`) without refitting, so methods are
  compared on one fixed yardstick: `mean_mm`, `p95_mm`, `far_frac` (CT surface >
  `--far-mm` 4 from its own reference), `misassigned_frac` (far from its own and within
  `--near-mm` 1.5 of a neighbour's reference), `missing_frac` (reference surface >
  `--far-mm` from the CT mesh: bone not captured), `volume_cm3`. The reference is a
  yardstick only.
- Running rules: about 60-130 s per bone on an Apple M5 (32 GB) (26 s initial
  refinement + ~6 s per point prompt), 55 bones in about 80 minutes. Full-grid bool
  labels are ~230 MB each on the chest grid, so at most 8 are cached. Never run
  `segment_nni.py` concurrently with other heavy jobs (an exact `check_overlaps.py` run
  was killed for memory while segmentation ran); run QA after segmentation finishes.
- Pilot evidence (T8, 1 mm chest, same fixed reference), previous -> new: volume 18.0
  -> 26.9 cm3 (old body hollow); missing reference surface 5.6% -> 1.5%; mean residual
  1.79 -> 1.75 mm; mask voxels below 0 HU 6.7% -> 4.4%; dense bone (> 300 HU) within
  2 mm outside the mask 24.9k -> 16.5k voxels. The new mask fits the data better on
  both sides while being solid; raw-slice contour overlays show it following the cortex
  and stopping at the disc and the rib head.
- State (2026-11-01): 55 bones in `spine/ct_bones.blend`, collection `bones`, run tag
  `ct2021-nni-fused`, all on fused 0.6 mm crops (chest 404+406, abdomen 601+602+603);
  `rib_right_11` forced to the chest group (`--group-override rib_right_11=Lung`);
  `rib_right_12`, L3-L5, sacrum, hips, and femurs come from the abdomen group. The
  axial-only run stays on disk as `RUN/nni` (tag `ct2021-nni`) for comparison. Fused
  vs axial-only: sacral foramina open, 3 mm terracing gone; dense bone left within
  1.2 mm outside the mask 1324 -> 397 mm3 (sacrum), 455 -> 253 mm3 (L3), 275 -> 42 mm3
  (left first rib); volumes within 5% except first ribs (-15%, a soft-tissue blob at
  the rib head dropped) and right 12th rib (-12%); contacts 23 measured / 35 inferred
  (25 / 33). Fixed atlas yardstick (`RUN/nni_fused/qa/reference_check.json`, 23
  bones): the runs are indistinguishable (mean residual within 0.1 mm for 20 of 23;
  left first rib 2.94 -> 1.51 mm; sacrum 4.10 -> 4.54 mm because open foramina add
  surface the solid atlas lacks, missing 25.9 -> 22.3%); the atlas fit is 1.1-4.5 mm
  off and cannot resolve sub-mm gains, so judge by the CT-based checks and overlays.
  `reference_check.py` on hips/sacrum needs one bone per call and <= 4000 samples or
  it is killed for memory. Known blemish: a faint seam down the left iliac wing; all three series
  cover the hip and none has a slice discontinuity there, so it is most likely an
  nnInteractive patch-tiling seam on the large hip crop, not a fusion artefact. Label grids of the fused run are not
  saved yet (`RUN/nni/labels` holds the axial-only ones).

Shared tools and the previous method: `pipeline.py` runs DICOM series groups -> TotalSegmentator guides ->
`fuse_planes.py` -> `extract_bone.py` -> `blender_bridge.py`, cached per run directory
with a `run.json` manifest and `--jobs N` bones in parallel (2-4 GB RAM each); it
forwards `--finest-only` to `fuse_planes.py`, and `--atlas-labels DIR` and
`--extract-args ...` (everything after it) to `extract_bone.py`; `--smooth-iters`
defaults to 5. `export_atlas_meshes.py` (headless Blender) exports the 277 Z-Anatomy
bones as PLY into `<segmentation>/atlas/meshes/`; `index_bp3d.py` indexes BodyParts3D
4.0 as an alternative atlas; `register_atlas.py` fits either to the CT bones
(`atlas/`, costal cartilages included); `align_groups.py` brings secondary
acquisition groups into the primary group's frame (`align/`); `compare_runs.py` diffs
two runs; `check_overlaps.py` is the second-pass QA of neighbouring bones (`qa/`);
`resegment_pairs.py` is the consistency pass that re-cuts flagged pairs
(`pipeline.py --resegment` runs both at the end of a run); `extract_discs.py`
segments the intervertebral discs (`discs/`). Never edit
`tools/ct/*.py` while `pipeline.py --jobs N` runs: the spawned workers re-import the modules and crash on
mismatched arguments; killing the main process leaves idle orphan multiprocessing
workers that must be killed too. Invariants:

- Blender metres = scanner LPS mm x 0.001, scanner origin kept (+X left, +Y posterior,
  +Z superior); one shared transform, so joint gaps are the scan's. The target `.blend`
  is metric with millimetre display; `scale_length` stays 1.0.
- `--match` groups are listed highest resolution first
  (`--match Lung --match Delayed`); a group is every series matching one of its
  `|`-separated substrings (all planes, several reconstructions per plane), and must
  be one co-registered acquisition. Each bone comes from the first group whose axial
  volume fully contains its guide label. `segment_nni.py` reads the groups from
  `run.json` but uses only each group's finest ORIGINAL axial series.
- **Previous method.** Fusion weights inputs by edge direction and slice thickness (`--thickness-power` 1);
  grids snap to a global lattice from the scanner origin, so neighbouring ROIs share
  voxels exactly. Every input of a group is fused by default (smoother surfaces,
  preferred); `--finest-only` keeps the finest series per plane and drops series
  thicker than `--max-thickness-ratio` 2.5x the finest (sharper edges, +27% gradient
  on T4, but noisier, rougher surfaces).
- **Previous method.** Surface = 250 HU iso-surface of the fused 0.45 mm volume, padded with air so it is
  closed. A 0.6 mm Gaussian on the fused volume before any thresholding removes the
  speckle of the sharp lung kernel (little spikes on chest bones) without moving a
  symmetric edge; `pipeline.py --denoise-mm auto` (default) applies it only to groups
  whose finest slices are <= 1.5 mm, not to the smooth-kernel pelvis. TotalSegmentator
  labels (1.5 mm) only seed the split where confident: a label is trusted within
  `--label-reach-mm` 4 of itself; bone outside the `--band-mm` 4 band around label
  boundaries follows the nearest label. `--split distance` (default) is a watershed on
  the distance transform, run first on the cortex above `--split-level` 500 HU so thin
  joints bridged by partial volume fall apart; the softer 250-500 HU shell is then
  grown onto the cores by a watershed on inverted intensity (fronts meet on the
  darkest path, the joint; joining the nearest core cut a thin seam through vertebral
  bodies whose lateral wall fell below 500 HU). `--rib-band-mm` 20 applies one-sidedly
  where a non-rib label is nearest and a rib label is the runner-up, since
  TotalSegmentator puts rib heads, necks, and tubercles on the vertebra (its rib label
  stops ~26 mm short of the head). Rib prior: from the medial end of each rib label a
  cylinder (`--rib-extend-mm` 30, `--rib-radius-mm` 6) is extended along the rib's
  local direction towards the spine; dense cortex (> `--split-level`) inside it that
  connects to the rib label and is not deeper than `--rib-core-erode-mm` 6 inside a
  vertebra label is marked as that rib. Alternatives: `--split hu` (inverted
  intensity) and `--split cascade` with `--levels` (900 -> 250 HU, merged pieces cut
  at the thinnest neck; worse on T4: ragged, grabbed neighbours). The target label is
  closed and hole-filled (`--guide-close-mm` 5) before the ROI (`--guide-margin-mm`
  8), since the sacrum label misses the S1 body. `extract_bone.py --target` accepts
  several bones, cut in one shared partition (joint extraction); one target behaves
  as before.
- **Previous method.** Solid fill: marrow > 100 HU (`--core-level`) inside an 8 mm silhouette
  (`--envelope-mm`) plus > 150 HU (`--fill-level`) within 2x `--solid-close-mm` (2 mm)
  of the cortex, closed and hole-filled; pockets < 60 HU and > 50 mm3 (canal, foramen
  contents) are carved back out. The per-bone JSON records `basins_file`, `basin_ids`,
  `target_id` for `check_overlaps.py`.
- **Previous method.** `pipeline.py` Blender objects carry `ct_group`, `ct_level_hu`, `ct_run` (`--run-tag`, default
  `<out name>@<time>`); objects in the collection from other runs are removed at the
  end of a run unless `--keep-old`. `--blender append|new` needs BlenderMCP on
  localhost:9876; otherwise meshes stay in `<run>/meshes/`.
- Atlas prior: `register_atlas.py RUN --meshes-dir ... --out RUN/atlas` fits every
  atlas bone to its CT bone (global similarity from matched centroids, sternum, skull,
  and costal cartilages excluded as composites; then a per-bone two-way similarity
  ICP with guards, a rigid retry, and a global fallback; partially scanned bones
  flagged) and writes `meshes/<bone>_atlas.ply`, `labels/<bone>.nii.gz` (0.9 mm), and
  `atlas_fit.json`. 2021 scan: 55 bones in 126 s; mean residual 1.3-2.3 mm on
  vertebrae, 1-3 mm on ribs, 3-4 mm on hips and sacrum.
  Previous method: `extract_bone.py --atlas-labels DIR` (via `pipeline.py --atlas-labels`) decides
  ownership from the fitted shapes: a bone owns its atlas shape plus whatever its
  coarse label covers that no other bone's atlas claims (coccyx stays with the sacrum,
  the rib head leaves the vertebra); `--atlas-skip` (default femur_left, femur_right,
  humerus_left, humerus_right, vertebrae_C6) keeps the coarse label for partially
  scanned bones.
  Surfaces come only from the CT iso-surface: the atlas never shapes or moves a bone,
  and the per-bone fit follows rotated or asymmetric vertebrae. Effect: rib bone left
  on T4 0.81 -> 0.09 cm3; L5-sacrum overlap 2.4 cm3 / 4.9 mm penetration -> 1 mm3 /
  none. `--blender` imports the globally fitted atlas into a hidden `atlas` collection.
- Costal cartilages: calcified costal cartilage exceeds 250 HU and TotalSegmentator
  labels it as rib (its `costal_cartilages` class was 0.5 cm3 on the 2021 scan), so
  the first ribs ran into the sternum. `register_atlas.py --cartilage` (default on)
  carries each fitted rib's transform onto the atlas costal cartilage of that rib
  (Z-Anatomy ribs 1-10, BodyParts3D ribs 1-7) and writes
  `labels/costal_cartilage_{side}_{n}.nii.gz`; `--reuse-fit` adds them to an existing
  fit without refitting bones. With fitted-atlas labels, `pipeline.py` treats
  `costal_cartilage_{side}_{n}` as extra classes (`label_path` falls back to the atlas
  label; neighbours: its rib, the sternum, adjacent cartilages; rib n and the sternum
  list it as a neighbour), extracts each as its own object, and imports them into a
  Blender collection `cartilage`. They are stiff links, not bone.
- BodyParts3D atlas: `index_bp3d.py ATLAS_DIR` (here `spine/atlas_bp3d/`) builds
  `atlas_bones.json` and per-part PLYs (metres, Z-Anatomy atlas schema); the
  TotalSegmentator -> BodyParts3D map is `ts_to_bp3d.json` beside them. Fit with
  `register_atlas.py --atlas-json --atlas-meshes --atlas-mapping`. 2021 scan:
  per-bone residuals tie with Z-Anatomy (2.42 vs 2.32 mm mean). BodyParts3D 4.0 is
  CC BY 4.0 per its README (OBJ headers still say CC BY-SA 2.1 JP); keep
  `ATTRIBUTION.txt` and credit "BodyParts3D, © The Database Center for Life Science
  licensed under CC Attribution 4.0 International".
- Group alignment: `align_groups.py RUN --primary Delayed --blender` extracts the
  vertebrae both acquisitions cover from both groups, fits ONE rigid transform (no
  scale) from the secondary group onto the primary, and applies it to every
  secondary-group bone (originals in `meshes/unaligned/`, transform in each bone's
  JSON and `align/alignment.json`; Blender objects get `ct_aligned_to`,
  `ct_align_rms_mm`). The pelvis (Delayed) is the reference frame; no bone is ever
  moved individually. 2021 scan: the two acquisitions were already co-registered to
  0.16 mm (rigid fit 0.01 deg / 0.04 mm on T11-L2), so nothing had to move.
- `check_overlaps.py` flags neighbouring pairs OK/CHECK/BAD from overlapping voxels,
  split disagreements between the two independent runs, penetration depth, and minimum
  joint gap. Its default fast inside test (KD-tree, sign of the nearest face normal)
  can report a false "penetration" on interlocking surfaces whose voxel overlap is
  zero: trust the voxel columns first; `--exact` uses ray tests (needs `rtree`,
  minutes per pair on these meshes).
- **Previous method.** Consistency pass: independent per-bone extractions disagree at shared joints (two
  runs, two decisions); the 2021 overlap report flagged 33 pairs, almost all rib heads
  and facet joints, some with 2-8 mm mesh interpenetration. `resegment_pairs.py RUN
  --pairs A:B ... | --overlaps RUN/qa/overlaps.json --atlas-labels DIR --blender`
  cuts a pair with ONE shared decision per voxel: a voxel in both final masks goes to
  the bone whose fitted-atlas label claims it, else to the nearer bone body; bone one
  run handed to the partner's basin that the partner's run did not keep ("orphans",
  typically a rib head or a facet process) joins the partner when connected to it;
  overlaps are arbitrated again after that step; both bones are re-meshed from their
  own fused volumes with the partner's adjusted mask cut out, so the surfaces meet
  exactly on the shared boundary. Nothing is moved; originals go to
  `meshes/before_resegment/`, each changed bone's JSON gains `resegmented_with`.
  2021 spine: every lumbar pair and the lower thoracic pairs now have zero overlap and
  gaps of 0.03-0.2 mm (facets still meet with no gap at 3 mm slices, as before).
  `pipeline.py --resegment` runs `check_overlaps.py`, then `resegment_pairs.py` over
  all consecutive vertebra pairs plus every BAD pair with >= 1 mm penetration, then
  `check_overlaps.py` again.
- Discs: `extract_discs.py --run RUN [--meshes-dir] [--out RUN/discs] [--levels ...]
  [--blender --collection discs] [--blender-only]` segments every disc between
  consecutive vertebra meshes (C7-T1 ... L5-sacrum) from the fused CT: the
  low-density region (HU window 40-250) between the two endplates (opposed
  distance gradients), inside a corridor through both vertebral-body outlines, with
  a bone trough above and below; footprint = the two endplate outlines' convex hulls
  intersected. Per level a mesh (metres, Blender/LPS frame) and a JSON: centre,
  normal, in-plane axes, mean/min/max/anterior/posterior height, footprint area and
  radii, volume, wedge angle (signed sagittal), endplate centres and normals, a
  5-ring radial HU profile, coverage, flags. `nucleus_radius_ratio` 0.5 is a stated
  default (CT cannot see the nucleus). `--blender`: disc meshes into collection
  `discs` plus an Empty `<level>_center` (ARROWS, size = equivalent radius, local Z =
  disc normal, local X = major axis) carrying all parameters as custom properties,
  for a force-based disc model, not a mesh element. ~25 s per level. 2021 scan:
  lumbar discs 42-48 mm wide, 30-33 mm deep, 4-6 mm high, lordotic wedges 4-8 deg;
  L5-S1 3.4 mm (2.5 mm posteriorly); thoracic discs measure 1.5-2.5 mm, below the
  scan's resolution (real 4-7 mm), flagged thin; T7-T8, T8-T9, T10-T11, C7-T1
  flagged unreliable on the snapshot used.

Limitations: acquisitions differ by a few mm (breathing, table move) until
`align_groups.py` applies one rigid transform per group; facet and costovertebral
joint spaces (0.5-1 mm) are at or below the voxel size (1 mm chest slices, 3 mm
abdomen and pelvis slices), so most of those boundaries are INFERRED by the model, not
measured (see `RUN/nni_fused/qa/contacts.json` and each object's `ct_inferred_boundaries`);
a joint-thickness prior would be a modelling choice, not data. Previous method only:
the costovertebral cut came from a rib prior plus the atlas prior (rib bone left on
T4: 0.8 cm3 coarse, ~0.25 cm3 with the rib prior, 0.09 cm3 with the atlas prior), the
3 mm lumbar facets met with no gap, and `resegment_pairs.py` fixed pairs
`check_overlaps.py` flagged. Previous-method meshes are full resolution
(~400k-900k faces; `--decimate` on import); `rtree` is needed for exact mesh queries
(installed; without it `compare_runs.py` and `check_overlaps.py` fall back to a
KD-tree); no automated tests, validate with `python -m compileall tools/ct`.

Bone overlap check (2026-11-01/02, user: "check each bone and their neighbors to see if there is
overlapping"): `check_overlaps.py --all-pairs` and a vertex signed-distance check both failed on
the fused meshes (OOM at femur/hip; 16 of 329 pairs in 8 h). `tools/ct/overlap_masks.py` instead
measures the model's masks before the joint resolution, which is where bones can actually compete
(the resolved labels are exclusive, so the meshes cannot interpenetrate by more than the 0.5 mm
smoothing): 55 masks, 524 pairs with meeting extents, 11 pairs share any voxel, all between
adjacent vertebrae at the facet joints, 1-57 mm3 each (0.0-0.1% of the smaller bone), deepest
0.85 mm (L2/L3), the rest exactly one 0.6 mm voxel. Ribs/vertebrae, sacrum/hips, hips/femora,
sternum/clavicles/ribs share nothing. Output `RUN/nni_fused/qa/overlap_masks.json`.

Muscles (2026-11-01, user: "use muscles segmentation with the same technique", "add all muscles
you can like the bones, with the model and 3 planes information", "run each muscle in a queue, not
parallel", "always use the highest quality source"):

- `segment_nni.py --structures muscles` = the bone method on ONE merged 0.8 mm volume over both
  acquisitions (`load_groups(merged=True)` -> group `Body`; they line up to 0.01 mm). Crops are
  fused per acquisition and blended across the 190 mm overlap (`fuse_series.fuse_crop_multi`:
  zero weight within 3 mm of a coverage edge, 20 mm ramp, no step at the seam).
- Highest quality source = lowest noise for soft tissue, finest slices for bone: in muscle the
  1 mm chest series (lung kernel) carries 35-50 HU of noise, the 3 mm soft-kernel series
  8-10 HU, so `quality="noise"` (1 / variance) lets the soft-kernel data dominate muscles,
  while bones keep `quality="resolution"`. User (2026-11-01): "use 3mm if its less noise for
  soft tissues, use all 3 planes" -> the 3 mm soft-kernel chest+abdomen acquisition 401/402/403
  (excluded for bones) is a third acquisition of the muscle volume (`--soft-tissue-series`);
  pectoralis noise 35 -> 9 HU in the merged crop.
- Guides: TotalSegmentator tasks `total` (gluteals, iliopsoas), `abdominal_muscles` (20 used;
  psoas_major skipped, iliopsoas covers it), `headneck_muscles` (lower-neck muscles in field);
  `thigh_shoulder_muscles` needs a free licence. Guides at full resolution (user: no 3 mm when
  finer exists). Neighbours = other muscles + saved bone masks reaching into the crop (largest
  12); bones are carved out of the muscle labels after resolution; `--measured-max-hu 0` (fat
  plane visible = measured). Outputs `RUN/nni_fused_muscles`, collection `muscles`, tag
  `ct2021-nni-fused-muscles`, property `ct_structure`.
- `run_ct.py` = single entry point (guides -> bones -> muscles -> Blender), one progress line
  per structure, model output in `OUT/qa/segment.log`; structures strictly sequential.
- Pilot (gluteus_medius_left): 185 s, 303 cm3 (guide 302), median 59 HU, 0% fat / 0% bone
  inside, negative prompts in hip, femur, sacrum, L5; contour follows the fat plane against
  gluteus maximus; imported into `muscles`.
- Full queue (2026-11-01 evening, 42 muscles sequential, ~10.7 min each): gluteals + iliopsoas
  4-12 min, obliques 12-16 min. Interim import of the first 8 showed the left gluteus maximus
  clipped at the lateral face of its 25 mm crop box and a step-shaped dent on the right one
  (model patch edge, not clipping) -> both re-run with `--margin-mm 40` after the queue, then
  `run_ct.py --skip-segment --blender` re-resolves and re-imports all muscles. Re-run result
  (22:24-22:52): left 808 cm3, right 867 cm3, the dent is gone; the left one still ends at
  x = 176.7 mm, which is the edge of the 601 series' reconstruction field (176.5 mm): the lateral
  hip is cut by the scan, like the bottom at z = -370 mm, and no margin changes that. Final
  import: 39 muscles, 6563 cm3, `ct2021-nni-fused-muscles`.
- Result (finished 22:24, 5 h 44 min): 42/42 segmented, 6557 cm3, imported into `muscles`.
  Largest gluteus maximus 801/868, latissimus 432/?, erector spinae 425/386, pectoralis 324 cm3.
  Three neck-edge fragments (platysma both sides 0.0 cm3, posterior_scalene_left 0.1 cm3: the
  scan stops at the lower neck) moved to `RUN/nni_fused_muscles/excluded/` and removed from
  Blender -> 39 muscles kept; posterior_scalene_right (0.9) and middle_scalene_right (1.3) are
  real stubs at the field edge. Contact report: 36 muscle-muscle interfaces, all "inferred"
  because the interface median HU is 17-67: at 0.8 mm with the 3 mm soft-kernel data
  dominating, a 1-2 mm fat plane blurs to +20-40 HU, so `--measured-max-hu 0` is too strict
  for muscles; ~30 HU would call the serratus/rectus plane (17 HU) measured. Visual QA: right
  pectoralis major has a few pinholes laterally; rectus abdominis shows boxy edges at the
  tendinous intersections.
- Under-segmentation found 2026-11-01 23:50 (user: "the abdominal muscles not done yet?"):
  external_oblique 53/35 cm3 against guides of 201/176 (20-26%), rectus_abdominis 106/107
  against 161 (66%); the masks stop at the lower belly while the guides run up to the ribs.
  Cause: the bone rules. The model output of a sheet muscle breaks at its aponeurosis (the
  rectus at every tendinous intersection) and `largest_component` dropped the upper pieces, and
  the 2 positive prompts (distance-transform maxima, 20 mm apart) both sat in the thickest
  lower belly. Fix in `segment_nni.py`: soft tissue keeps every component overlapping the guide
  (`guided_components`, plus any >= 2% of the output) and gets 6 positive prompts spread along
  the guide's longest extent (`--pos-points` default 6 / 2 for bones; separation
  max(20 mm, extent / n)). The four muscles are queued for a re-run after the organ run (user's
  choice: "queue the muscle fix after them"; chain script + log
  `segmentation/logs/nni_abdominal_fix.log`), then all muscles re-resolve and re-import. The
  organ run itself started under the old rules (module loaded before the edit).
- User's later goals: discs redone with this method (seed = endplate gap of the bone masks),
  then cartilage, ligaments and whatever else CT shows (costal cartilage, spinal cord,
  ligamentum flavum; NOT longitudinal/interspinous ligaments or articular cartilage), each in
  its own collection.

Organs (2026-11-01, user: "can you segment the heart and lungs and internal organs?" -> "yes, add
organs to queue, whatever you can"):

- `segment_nni.py --structures organs` / `run_ct.py --structures organs`: 41 structures of
  TotalSegmentator's free task `total` on the merged 0.8 mm volume, bone AND muscle masks as
  neighbours and carved out; per-structure Blender collections `thorax` (10), `abdomen` (14),
  `vessels` (16), `nervous` (spinal cord); outputs `RUN/nni_fused_organs`, tag
  `ct2021-nni-fused-organs`. Queued behind the gluteus re-run (chain scripts in the session
  scratchpad; log `segmentation/logs/nni_organs_queue.log`), guides included (~20 min per
  acquisition), then ~41 x 10-25 min.
- First 11 organs (2026-11-02 01:15, interim import, tag `ct2021-nni-fused-organs-interim`):
  lung surfaces and the right fissures excellent; trachea 44, oesophagus 33, thyroid 19,
  liver 1706 cm3 (guide 1860) good. Two failures of the bone rules on soft tissue (the organ run
  loaded the module before the soft-tissue fix): heart 267 cm3 against a 576 cm3 guide (46%;
  the 2 prompts sat in the contrast-filled left chambers and the largest component kept only
  them) and the left lung lobes each grew over the whole left lung (upper 2455 vs guide 1322,
  lower 2192 vs 1530; 1909 cm3 shared = 87% of the smaller; the right lobes are 0.96-1.04 of
  their guides, so the left major fissure is simply fainter). Queued after the abdominal fix
  (`organ_fix.sh`, log `segmentation/logs/nni_organ_fix.log`): heart + both left lobes re-run
  with the soft-tissue rules and `--neg-points 3`, then all organs re-resolve and re-import. If
  the lobes still overlap, split the left lung by the nearer guide label instead.
- Full run (2026-11-02 04:06, 5 h 04 min for 41 structures, 10466 cm3 of meshes, 122 contacts
  of which 39 measured, final import tag `ct2021-nni-fused-organs`). Mask / guide volume
  ratios: 30 of 41 within 0.8-1.2; vessels fine after joint resolution (iliac veins 1.3-1.4
  before, 1.08-1.19 after). Under the guide, all by the old largest-component rule on
  structures whose model output broke into pieces: colon 0.31 (307 vs 1000 cm3), heart 0.46,
  small_bowel 0.67, duodenum 0.67, brachiocephalic_trunk 0.67. Over: the two left lung lobes
  (1.86 / 1.43, the fissure problem above). The queued fix (`organ_fix2.sh`, log
  `nni_organ_fix.log`) now covers all seven: heart + left lobes with `--neg-points 3`, then the
  four bowel / vessel structures with the default prompts, then all organs re-resolve and
  re-import. The same ratios on the muscle run flag three more under the guide besides the four
  abdominal ones: gluteus_minimus_right 0.60, iliopsoas_right 0.67, iliopsoas_left 0.74; they
  follow in the same chain (stage B) and all muscles re-import. Over the guide only small neck
  muscles at the field-of-view edge (trapezius 1.8-2.4, levator_scapulae_left 1.9,
  middle_scalene_left 2.1): the headneck guides are cut by the scan there, left as they are.
- Abdominal re-run (2026-11-02 05:05): external_oblique_left 229 cm3 (guide 201), right 294
  (176), rectus_abdominis_left 168 (161), right 200 (161): the masks are complete. Two further
  faults showed in Blender. (a) The right rectus object ended 100 mm below its mask: `mesh_bone`
  kept only the largest piece after the Gaussian smoothing, which pinches the rectus apart at its
  tendinous intersections, so two bellies with all their voxels present were dropped (the mesh
  JSON volume is counted from voxels, so it never showed). Fixed: every piece with at least 1% of
  the largest piece's area is kept (`components` in the mesh JSON); muscles re-meshed, organs
  re-meshed at the 09:18 import, bones re-meshed 2026-11-02 13:50 (55 bones, 5 min, every bone
  still a single piece, so nothing changed visibly). (b) Both recti end in a flat cut 85 mm above
  the pubic crest (pubis top at z = -255 mm, recti at -171 / -156): the TotalSegmentator guide
  stops there and the crop (guide box + 25 mm) cut the model's output at its face (mask voxels on
  the bottom face: 1274 / 1422). Fixed by crop growth in `segment_structure`: while the mask
  reaches a crop face that is not the scan's edge, grow 40 mm there and refine again from the
  mask (`--grow-mm`, `--max-grow` 3 for soft tissue, 0 for bones; `grow_steps`,
  `touches_crop_face` in the mask JSON). The recti re-run with growth right after the heart
  (`abs_regrow.sh`, log `nni_abs_regrow.log`), then the chain continues (`organ_fix3.sh`: left
  lobes, bowel + trunk, organs import, then gluteus_minimus_right, both iliopsoas and both
  external obliques, muscles import). (c) The left external oblique lost 87 cm3 in the joint
  resolution to latissimus_dorsi_left (446 vs guide 324) and serratus: contested voxels go to the
  nearer uncontested core, which favours an over-grown neighbour. Fixed in `resolve_group`
  (`guide_tiebreak`, soft tissue only): a contested voxel covered by exactly one of the guides
  goes to that structure, the rest to the nearer core; takes effect at the next muscle import.
- Rectus regrow result (2026-11-02 06:40): three grow steps each, no crop face touched at the
  end; rectus_abdominis_left 239 cm3 (z -253..101 mm), right 243 cm3 (-241..107), both now end
  at the pubic crest (-255) with three bellies; 28-33 min per muscle with the growth steps. The
  guide tie-break moved the contested voxels back: external_oblique_left 141 -> 164 cm3,
  internal_oblique_left 98 -> 121, latissimus_dorsi_left 404 -> 358, right 243 -> 299. Still
  open: a strip of the right rectus's top belly runs laterally along the costal margin toward the
  serratus (contact flagged measured); the heart re-run gave 493 cm3 (guide 576, was 267).
- Organ fix result (2026-11-02 09:18, `organ_fix3.sh` A1b-A3): the left lung lobe masks still
  overlap almost entirely (upper 2602 cm3 vs guide 1322, lower 1668 vs 1530: the model fills the
  whole lung from either seed, 3 negative prompts did not stop it), but the joint resolution with
  the guide tie-break splits them along TotalSegmentator's oblique fissure: meshes 1290 / 1430 cm3,
  a clean fissure in the renders. Re-runs with the current rules: colon 1402 cm3 (guide ~1000, was
  31%), small_bowel 615 (0.80 of guide), duodenum 69 (1.64, slightly over), brachiocephalic_trunk
  3.2 (0.67, tiny vessel, left alone); 41 organs re-resolved, re-meshed with the multi-piece
  mesher and re-imported (4:26). Renders in the session scratchpad `organs_qa/`.
- Muscle fix result (2026-11-02 12:26, `organ_fix3.sh` B1-B2): gluteus_minimus_right 59 cm3 (0.80
  of guide), iliopsoas_right 345 (0.96), iliopsoas_left 359 (0.91), external_oblique_left 200
  (1.00, no grow needed, upper edge now at rib 8). external_oblique_right leaked: the mask reached
  the crop's top face, the crop grew upward three times and each refinement climbed the chest wall
  (intercostals / serratus) to the armpit: 562 cm3 (3.19 of guide, 311 after resolution, 2 h),
  fragments up to the axilla in Blender while its real top edge is still a flat cut. Fix: a leak
  guard in the grow loop (`--grow-max-ratio` 1.8: a grow step whose mask exceeds that multiple of
  the guide volume is rejected and the previous mask kept; `grow_rejected` in the mask JSON); the
  recti reached 1.48 / 1.51 with their three legitimate steps. Re-run (`eo_right_fix.sh`, log
  `nni_eo_right_fix.log`, 13:43): the guard rejected grow step 1 (481 cm3 vs 316 allowed) and kept
  the ungrown mask, 285 cm3 (1.62 of guide), 212 cm3 after resolution (latissimus_dorsi_right back
  to 300); the fragments at the axilla are gone, the top edge is the guide's. Still open: a lateral
  strip on the right rectus's top belly toward the serratus; the right external oblique's top edge
  sits lower than the left's (rib 8) because its guide stops lower and growth is blocked there.
- Expect: lung surfaces excellent (air boundary), fissures partly guide-driven; heart border
  blurred by cardiac motion (not ECG-gated); bowel least reliable; mediastinum leans on the
  soft-kernel 3 mm acquisition through the noise weighting.
- Heart and discs stages (2026-11-03): the user obtained the TotalSegmentator non-commercial
  licence (all 16 gated tasks; key stored with `totalseg_set_license`, config in
  ~/.totalsegmentator). `run_ct.py --structures heart`: tasks heartchambers_highres (its aorta
  stored as heart_aorta so total's aorta survives), coronary_arteries, aortic_sinuses on both
  acquisitions' finest axial series (2 min per task on MPS), then `segment_nni.py --structures
  heart`: 12 parts on 0.6 mm crops, resolution-weighted fusion, bones + muscles + the organ masks
  around the heart as negatives / carve but not the organ run's heart, atrial_appendage_left and
  aorta (HEART_ENVELOPE), collection `heart`, run tag ct2021-nni-fused-heart. The Delayed
  acquisition ends at z = 134 mm (mid-ventricle), so its heart labels are the lower half only;
  the Lung acquisition covers the whole heart and the guide union across acquisitions is used.
  `--structures discs`: task vertebrae_body (vertebrae_body + one intervertebral_discs label),
  split per disc by `split_discs.py` (each label piece -> the refined vertebrae nearest above and
  below its centroid; disc_C2_C3 ... disc_L5_S1; C1/C2 pieces skipped), refined like the heart
  with the vertebra masks as negatives / carve, collection `discs`. Queue: heart parts, then
  discs (`heart_then_discs.sh`, log `nni_heart_discs.log`). User's order for what follows:
  lung airways + vessels (free task lung_vessels), shoulder muscles (thigh_shoulder_muscles),
  fat compartments / skin (tissue_4_types).
- Heart result (2026-11-03, 11:51-16:13, 4:22 h; 12 parts, 680 cm3): myocardium 116, left
  ventricle 130, right ventricle 107 (123 raw), left atrium 69 (79 raw), right atrium 40 (52
  raw), heart_aorta 136 (arch + descending to the diaphragm), pulmonary artery 74 (with both
  main branches), coronaries 1.9 (5 pieces, proximal segments only; the guide had 0.6),
  outflow tract 4.0, cusps 0.6 / 1.8 / 1.6 cm3. The first resolve meshed 11 parts: the right
  coronary cusp lies wholly inside the aorta mask and the nearer-core rule gave it, and most of
  the other cusps and the outflow tract, to the aorta. Fix: `NESTED["heart"]` parts keep their
  contested voxels (resolve_group `priority`), and a part without an uncontested core uses its
  own mask as the core; re-resolved with `--skip-segment` (2 min), 12 parts in collection
  `heart`. Stray pieces: a small aorta fragment above the arch (2 components) and a few
  coronary specks; none were fixed. Median HU: chambers 158-326 (contrast), myocardium 69.
- Discs result (16:13-17:15, 55 min; 18 discs C7/T1 to L5/S1, 56 cm3): the label pieces fell
  between consecutive vertebrae every time (15 on Lung, 9 on Delayed, union 18; the scan has
  no cervical discs above C7). Lumbar discs are right (L4/L5 10.2 cm3, 14.7 cm2 footprint,
  7 mm thick, 90 % inside the inter-body gap) but the thoracic discs filled only a quarter of
  the gap (T4/T5 0.6 cm3 in a 2.4 cm3 gap, 2.5 of 8 cm2 footprint): the vertebrae_body disc
  label marks only the centre of a thin disc and the refinement stayed on it. Fix:
  `split_discs.fill_gap` grows each guide into the inter-body gap (see README); the full
  re-run D2 (`discs_refill.sh`, log `nni_discs_refill.log`, 17:23-19:00) replaced the 18.
- Discs D2 outcome (19:00-21:30, masks post-processed with `--skip-segment` re-resolves, 2 min
  each, no re-segmentation): two discs leaked up the spinal canal on the first pass (T2/T3 and
  T4/T5: 15 cm3, median 30 HU, against a 1 cm3 guide; the grow guard only checked grow steps),
  and the thin thoracic discs stayed under their guide (T6/T7 1.1 of 1.7 cm3, median 159 HU:
  the model follows the endplate rim). Added `finish_mask` in segment_nni: first-pass leak clip
  (mask > 1.8x guide -> guide + 3 mm) for all soft tissue, and for `GUIDE_IS_SPACE` kinds
  (discs) clip to 1 mm around the guide + union with it. The gap fill itself needed four rules
  before the renders were clean: bodies only bound the gap (`vertebrae_body` label), at least
  6 mm of both bodies in a column, no rib head in the gap (costovertebral joint), and a slab of
  half the typical gap + 1.5 mm around the fitted mid-plane (without these a column grazing the
  concave posterior wall or a demifacet filled 8-10 mm of canal / joint space as a wedge hanging
  off the posterolateral corner). Final meshes: C7/T1 0.9, T1/T2 1.0, T2/T3 0.8, T3/T4 0.7,
  T4/T5 0.7, T5/T6 0.9, T6/T7 1.1, T7/T8 1.4, T8/T9 1.7, T9/T10 1.5, T10/T11 2.6, T11/T12 3.2,
  T12/L1 4.6, L1/L2 5.9, L2/L3 8.7, L3/L4 9.1, L4/L5 9.7, L5/S1 5.0 cm3 (mid-thoracic discs
  3 mm thick in this subject; no column over 7 mm). Renders in the session scratchpad
  `heart_discs_qa10/`. `discs_finish.py` (scratchpad) applies finish_mask to saved masks when
  only the post-processing changes.
- Heart in detail (user asked for free models): TotalSegmentator `heartchambers_highres`
  (myocardium, 4 chambers, aorta, pulmonary artery; sub-mm training) + `coronary_arteries` +
  `aortic_sinuses` are licence-gated but free for non-commercial use (request at
  backend.totalsegmentator.com/license-academic, `totalseg_set_license -l aca_...`; their
  outputs may not train new models). Fully free alternatives: platipy (Apache-2.0,
  `pip install platipy[cardiac]`, `platipy segmentation cardiac`, 18 substructures incl.
  chambers, LAD/circumflex/left and right coronaries, 4 valves, SA/AV nodes; nnU-Net + atlas,
  built for radiotherapy planning CT, Linux-tested), NVIDIA VISTA-3D / NV-Segment-CTMR
  (chambers + LV myocardium among 127 classes, research use), Zenodo 11278065 (UMC Utrecht
  nnU-Net v2 models, CC BY 4.0, 4 chambers + great vessels; model B adds valves and 4 coronary
  segments). Any of these can seed the nnInteractive refinement as the guide.

Next priorities (CT):

1. Thoracic disc heights: now measured by the discs stage (`--structures discs`, collection
   `discs`) rather than `extract_discs.py`; this subject's mid-thoracic inter-body gaps are
   3-4.5 mm (T4/T5 3.0, T6/T7 4.5 mm mean), the lumbar ones 7-8 mm. Compare the D2 disc
   volumes with the gap volumes (`disc inside gap` check in the session notes) after the re-run.
2. Muscles from the MyoSuite BodyParts3D-derived set.
3. A second scan compared with `compare_runs.py` against `RUN/nni_fused/meshes`.

## Core Functions

### `alignment.base_object_name(name)`

Removes one trailing Blender suffix matching `.NNN`. This is the current identity
rule for object pairs. Do not broaden it without testing names that naturally end in
digits.

### `alignment.world_vertices(obj)`

Uses Blender's bulk `foreach_get` API and NumPy to return all mesh vertices in world
coordinates. It intentionally accounts for mesh data transforms and differing object
origins.

### `alignment.mesh_alignment(source_object, target_object)`

Calculates the best-fit rotation and translation from corresponding vertex indices.
It uses SVD, corrects reflections, checks geometric resemblance, and returns a
Blender 4x4 matrix containing rotation and translation only.

Important assumption: paired meshes have identical topology and vertex ordering.

### `properties.refresh_transparency_overlay(scene, excluded_pair_name=None)`

Despite its legacy name, this is the central comparison-display refresh function.
It always restores the previous display first, chooses the active comparison
collection, skips the selected pair, snapshots object state, and applies either
wireframe or transparency plus optional On Top.

Future cleanup may rename this to `refresh_comparison_display`, but that rename must
update the operator import and all property callbacks together.

### `properties._restore_transparency_overlay()`

Restores exact prior object alpha, `show_transparent`, `show_in_front`, and
`display_type` values. It also restores each solid viewport's prior color mode and
deletes all temporary `_biomechanics_previous_*` custom properties.

This restoration must run when:

- The comparison mode or collection changes.
- Wireframe, opacity, or On Top changes.
- Comparison Display becomes None.
- The add-on unregisters or reloads.

### `properties._update_transparency_overlay(scene, context)`

Blender property update callback. It finds an active Original or Rigid mesh, derives its base
pair name, and calls the central refresh while excluding that pair.

### `register_properties()` / `unregister_properties()`

Own all scene-property registration and cleanup. Registration sets collection
defaults and reapplies stored display state after reload. Unregistration restores the
viewport before deleting properties.

### `register()` / `unregister()`

Package lifecycle in `biomechanics_addon/__init__.py`. Register classes in dependency
order and unregister them in reverse order, preserving Blender add-on conventions.

## Source Ownership

- `biomechanics_addon/__init__.py`: metadata and add-on lifecycle.
- `biomechanics_addon/alignment.py`: pairing helpers and geometry math.
- `biomechanics_addon/correspondence.py`: paired-point hover overlay and maximum
   corresponding-vertex distance search.
- `biomechanics_addon/muscle_atlas.py`: bpy-free muscle attachment and landmark data.
- `biomechanics_addon/muscle_strings.py`: muscle string building, length measurement,
   length-change colors, hover overlay, and CSV export.
- `biomechanics_addon/muscle_sheets.py`: triangle sheets between neighboring strings
   and their per-face area-change colors.
- `biomechanics_addon/ligament_atlas.py`: bpy-free ligament and disc atlas with
   mechanical parameters and contact-surface landmark specs.
- `biomechanics_addon/distance_field.py`: bpy-free signed distance grid, sign
   recovery, and trilinear queries.
- `biomechanics_addon/bone_fields.py`: Blender-side field computation, disk cache, and
   collision surface samples.
- `biomechanics_addon/chain_solver.py`: bpy-free batched XPBD rigid chain solver.
- `biomechanics_addon/chain_interaction.py`: chain session, target Empty, handlers,
   and operators.
- `biomechanics_addon/chain_package.py`: bpy-free chain package save/load/rebuild.
- `biomechanics_addon/posture_policy.py`: bpy-free posture observation, controllers,
   and policy file I/O.
- `tools/train_posture.py`: headless posture optimization (evolution strategy).
- `tools/train_posture_surrogate.py`: learned solver dynamics and gradient-trained
   posture policy; `tools/surrogate_dashboard.py` + `.html`: live training web page.
- `biomechanics_addon/distance_heatmap.py`: temporary per-vertex distance heatmap and
   viewport legend.
- `biomechanics_addon/dataset_io.py`: bpy-free chunked NPZ dataset serialization.
- `biomechanics_addon/interaction.py`: target-driven constrained transforms.
- `biomechanics_addon/neural_geometry.py`: evaluated mesh geometry and neural pose-labeling helpers.
- `biomechanics_addon/neural_constraints.py`: model loading, cached ONNX sessions,
   prediction, pose metrics, interpolation, and conservative path projection.
- `biomechanics_addon/neural_training.py`: bpy-free command, status, and process helpers
   for the external training orchestrator.
- `biomechanics_addon/operators.py`: alignment action and user-facing validation.
- `biomechanics_addon/panels.py`: sidebar layout and control enabled states.
- `biomechanics_addon/properties.py`: scene settings and reversible display state.
- `biomechanics_addon/sampling.py`: bpy-free deterministic pose sampling helpers.
- `biomechanics_addon/schemas.py`: bpy-free package, dataset, and model contracts.
- `tools/generate_neural_dataset.py`: background-Blender geometric labeling.
- `tools/train_constraint_model.py`: external PyTorch training and ONNX export.
- `tools/active_learning_loop.py`: bounded full-mesh active-learning rounds.
- `tools/run_neural_training.ps1`: resumable dataset, training, and active-learning orchestration.
- `tools/dev_load.py`: clean package/submodule reload during Blender development.

## External Neural Constraint Pipeline

The current neural milestone trains a geometric feasibility model for one fixed and
one moving Rigid mesh. The verified development pair is fixed `sacrum`, moving `L5`.
It is not an anatomical-stability, loading, ligament, cartilage, or pain model.

### Artifacts and schemas

- Package schema v2 records object names, neutral matrices, scene scale, per-axis
   ranges, deterministic seed, proxy/full-mesh settings, and geometry fingerprints.
- Dataset schema v1 stores raw local 6-DOF poses in millimeters/degrees, validity,
   clearance/overlap measurements, label provenance, split, and group id.
- Model manifest schema v1 records normalization, model I/O, fingerprints, neutral
   transforms, scene scale, calibrated validity threshold, metrics, and ranges.
- External tools must load bpy-free add-on modules through `tools/_addon_import.py`;
   importing `biomechanics_addon` directly outside Blender executes `__init__.py` and
   requires `bpy`.

### Training workflow

1. In Blender, choose fixed/moving Rigid meshes and export package v2.
2. Run `tools/generate_neural_dataset.py` in background Blender. It creates
    deterministic anchors/LHS/balancing/refinement stages, checkpoints each stage,
    and uses full geometry for holdout and final boundary labels.
3. Run `tools/train_constraint_model.py` with the workspace `.venv`. It performs
    leakage-safe split grouping, train-only normalization, weighted two-head MLP
    training, conservative holdout-threshold calibration, ONNX export, and
    PyTorch/ONNX Runtime parity validation.
4. Optionally run `tools/active_learning_loop.py` for bounded uncertain-pose
    full-mesh labeling and retraining.
5. Import `model_manifest.json` in Blender. Activation rejects wrong object names,
    fingerprints, scene scale, fixed transform, missing ONNX, invalid model I/O, or
    unavailable ONNX Runtime without replacing the last known-good model.
6. Alternatively, use **Geometric Dataset > Start One-Click Neural Training**. It
   refreshes the package with the standard +/-10 mm / full-circle rotation domain,
   launches `tools/run_neural_training.ps1` outside Blender's UI thread, and polls
   its atomic status file. The script resumes dataset checkpoints and never treats
   an incomplete model bundle as publishable.

The translation domain defaults to +/-10 mm. **Limit Rotation Domain** defaults off,
which selects full-circle training; enabling it exposes a per-axis degree limit for
smaller anatomical neighborhoods. Full-circle training uses an Euler input
domain of -180 to +180 degrees on every local axis (360 degrees total), but the network
internally converts orientation to the first two rotation-matrix columns. This 6D
orientation representation makes equivalent Euler descriptions and the -180/+180 seam
continuous. Dataset generation combines full-domain, regional (+/-45 degree), and local
(+/-10 degree) Latin-hypercube tiers, plus authoritative full-mesh single-axis sweeps;
the one-click full-circle base budget defaults to 10,000 poses. Pose path distance is
translation magnitude plus quaternion rotation angle multiplied by the stored 50 mm
joint radius. Conservative projection bisects the first invalid path interval to a
combined distance of at most 0.5 mm. This is an empirical model resolution target, not
an exact collision guarantee.

### Blender inference behavior

- Whole-pose interaction converts the requested target transform to trained
   neutral-local `[tx, ty, tz, rx, ry, rz]` coordinates.
- In-range predicted-valid requests are accepted. Invalid requests are projected
   from the last accepted pose toward the target using linear translation,
   shortest-path quaternion slerp, a first-invalid scan, and conservative bisection.
- The target Empty remains at the requested pose; the moving mesh uses the accepted
   or projected pose. Explicit attached followers move afterward and do not affect
   model validity.
- Out-of-range requests and active-model projection failures hold the current pose.
- With no compatible active neural model, whole-pose interaction follows the target
   without translation or rotation limits.
- `Validate Current Pose` performs an authoritative full-mesh check on demand and
   reports validity, clearance, and overlap-pair count. Interactive predictions are
   not presented as exact collision proofs.
- The neural panel currently displays model path/source, compatibility errors,
   prediction confidence, predicted clearance, and whether projection occurred.

### Current Phase 4 validation state

- Host tests cover conservative projection, out-of-range rejection, and quaternion
   rotational contribution to the pose metric.
- A live sacrum/L5 exact full-mesh validation returned valid with approximately
   0.74 mm clearance and zero overlap pairs.
- Source reload and operator registration have been verified. The full host suite
   now has 41 tests, and `compileall` is clean after the one-click orchestration
   and panel split implementation.
- Still required before declaring Phase 4 complete: profile live inference, exercise
   mixed translation/rotation paths and disconnected-valid-region behavior, confirm
   target-versus-constrained movement, and exact-check neutral/interior/projected/
   intentionally-invalid/random trajectory poses.

## Invariants

- Only Original moves during alignment; Rigid never moves.
- Alignment includes rotation and translation, never scale.
- Original objects move as one rigid group.
- Object origins are not trusted for alignment; mesh vertices define the fit.
- Exactly one Rigid object must match the selected Original object.
- The selected aligned pair is not transparent, wireframe, or forced on top.
- Comparison display changes never modify materials or rendered appearance.
- Every temporary display change is reversible and leaves no
  `_biomechanics_previous_*` properties behind.
- Transparent mode restores every viewport's prior solid color mode.
- Wireframe and transparency are mutually exclusive for the comparison collection.
- Constrained interaction accepts mesh objects from Rigid only.
- Movement, collision, contact, and spring calculations use Rigid meshes; Original
   remains alignment and comparison data.
- User-facing linear measurements use millimeters and are converted through the
   Blender scene unit scale for transform calculations.

## Current Scene Facts Used During Development

- Original collection: `Collection`.
- Rigid collection: `rigid`.
- Each contains 47 anatomical mesh objects.
- Rigid also contains armatures, which comparison and alignment loops ignore.
- Names pair after suffix removal, for example `sacrum` and `sacrum.002`.
- The sacrum pair has 74,703 vertices and 143,150 polygons.
- Sacrum surface alignment was validated around `5.8e-08` Blender-unit RMS error.
- All 47 mesh pairs passed the topology and geometry-fit check.

## Validation Checklist

Run after changing alignment, properties, UI, or registration:

1. Run `python -m compileall biomechanics_addon`.
2. Reload with `tools/dev_load.py` or the source-launch workflow.
3. Confirm the panel appears under **3D Viewport > Sidebar > Biomechanics**.
4. Align `sacrum`; verify the paired surfaces coincide and all 47 Original meshes
   move as one group.
5. Verify object scale and inter-object distances remain unchanged.
6. Test Comparison Display transitions: None -> Original -> Rigid -> None.
7. Test transparent mode at multiple opacity values.
8. Test Mesh Wireframe on both collections.
9. Test Overlay On Top both enabled and disabled.
10. Confirm the selected pair is excluded in every comparison mode.
11. Confirm switching modes restores the previously displayed collection.
12. Confirm None and add-on unload restore all object and viewport state.
13. Confirm no object retains a `_biomechanics_previous_*` custom property.

## Next Session

State at handoff (2026-10-25):

- Scene: the live Blender file `rig40-osx.blend` (Google Drive `spine/`) is UNSAVED;
  a copy with everything below is saved as `spine/rig40-osx-complete-skeleton.blend`.
  Atlas sources are kept in `spine/zanatomy/SkeletalSystem100.fbx` and `Joints100.fbx`
  (Z-Anatomy, CC BY-SA 4.0; attribution required for any distribution).
- Rigid pose = Original pose (Reset Rigid to Original). Base Bone `femurs`, gravity 0.3,
  muscle strength 0.1, Pin Top off. No chain session, no training running.
- Skeleton completed from the atlas (see Atlas Completion): skull, mandible, hyoid,
  C1-C7, clavicles, scapulae, arms, hands, costal cartilages, extended femurs,
  patellae, tibiae, fibulae, feet, in both collections. Scan sternum re-seated by ICP
  onto the atlas sternum (moved 6 mm, 11 deg). `atlas` and `atlas joints` collections
  (imported FBX, registered) are hidden in the scene and can be deleted.
- Strings: built-in muscles (441) + 127 atlas muscles; built-in ligaments/discs (610,
  incl. costal cartilage) + 104 atlas ligaments, per collection. **Import Atlas
  Muscles / Ligaments** (`atlas_import.py`) snap Z-Anatomy attachment areas to the
  nearest scene-bone vertices; they skip muscles already modeled by name and any
  string whose ends land on one object. Hands and feet are single joined meshes, so
  their intrinsic muscles are skipped (66) and so are ligaments within them (most of
  the 234 skipped). Re-run the imports after Build Muscle Strings / Ligaments, which
  recreate the collections. The per-region transforms the import needs
  (`femur_fit`, `neck_offset`) live in `bpy.app.driver_namespace` and are lost on
  Blender restart; the atlas collections are already transformed by the global fit,
  so only leg (femur ICP) and neck (11.5 mm z) extras matter.
- Chain: the exported package (`biomechanics_cache/chain_package`) predates the
  femur replacement and the atlas strings; re-export before training. New femur
  meshes need their distance fields recomputed (Precompute Collision Fields).
- Training: `tools/train_posture.py` policy mode with `--perturb-mm 8 --perturb-deg 4
  --fail-cm 5` (pelvic joints + hold, sternum body) reached -4.83 at generation 6 vs
  passive -8.5 before it was stopped; the surrogate route
  (`train_posture_surrogate.py iterate`, 4 rounds) did not beat it (best -6.6), see
  Bone Chain. Live view: Move Bones + Posture Controller + Follow Training + Force
  Colors; `surrogate_dashboard.py` at http://localhost:8765 for the network view.
- CT bones (2026-11-01): bone surfaces now come from `tools/ct/segment_nni.py`
  (nnInteractive v1.0; weights CC BY-NC-SA 4.0, non-commercial) run on crops fused
  by `fuse_series.py` from the axial + coronal + sagittal series (0.6 mm isotropic);
  the threshold method is deprecated. 55 bones are in `spine/ct_bones.blend`,
  collection `bones`, run tag `ct2021-nni-fused`; QA lives in `RUN/nni_fused/qa`
  (`reference_check.json` scores the axial-only and fused runs on one yardstick). CT
  next priorities (thoracic disc heights, muscles, second-scan comparison against
  `RUN/nni_fused/meshes`) are listed under CT Bone Extraction.

Next steps, in order:

1. Split hands and feet into their bones (keep atlas names) so intrinsic muscles and
   ligaments import; or accept them as rigid blocks.
2. Rebuild strings, precompute fields for the new femurs, re-export the package,
   retrain (`--mode policy --gravity 0.3 --perturb-mm 8 --perturb-deg 4 --episodes 1
   --population 16 --steps 40`), and watch with Follow Training.
3. Make the registration an operator (import FBX, Umeyama fit, femur ICP, neck
   offset, bone placement) instead of session scripts, and persist the per-region
   transforms on the objects.
4. Cervical discs/ligaments in `ligament_atlas.py` (C1-C7 now exist) and a
   `biomechanics_chain_top` above T01 so the neck and head join the chain.
5. Earlier deferred items remain: neural-pipeline Phase 4 checks, pelvic coupling,
   comparison-display rename.

## Known Constraints

- NumPy and `mathutils` are supplied by Blender; a normal host Python import is not a
  valid add-on test.
- Pairing requires corresponding vertex indices, not merely similar-looking meshes.
- The current comparison state is global across Blender screens and objects, with
  exact restoration managed through temporary object custom properties and an
  in-memory viewport map.
- Blender UI selection changes alone do not invoke a custom callback. The pair
  exclusion refreshes when a display property changes or after Align to Source runs.
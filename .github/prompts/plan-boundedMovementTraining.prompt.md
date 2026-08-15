## Plan: Bounded Two-Bone Movement Training

Replace the current blocking six-dimensional grid run with a cancellable, bounded geometric trainer for two meshes in the Rigid collection, initially sacrum (fixed) and right ilium (moving). Persist geometry metrics separately from optional constraint scores so ligament and pubic spring models can be added or retuned later without repeating collision sampling.

**Steps**

### Phase 1: Stabilize geometry and data contracts
1. Refactor `biomechanics_addon/training.py` so proxy construction, pose generation, pose evaluation, result serialization, and Blender UI orchestration are separate functions/classes. Preserve source meshes and neutral transforms; never move the Rigid source objects during training.
2. Replace the unverified Decimate-only proxy path with a bounded proxy builder: evaluate a clean temporary copy, simplify it, verify the resulting face count, and retry with a coarser voxel-remesh/decimation setting when necessary. Abort before sampling with a clear diagnostic if either proxy still exceeds the configured hard face budget. Record actual proxy vertex/face counts and simplification settings in the result.
3. Define training format version 2 with stable sections for object identity and geometry fingerprints, neutral transforms, sampling settings, proxy metadata, per-pose coordinates, geometric metrics, constraint scores, and summary. Store translations in millimeters and rotations in degrees; retain matrices for reproducibility.
4. Treat neutral contact as an explicit baseline measurement rather than assuming separated meshes. Record overlap/contact and clearance relative to neutral, but do not automatically accept arbitrary penetration merely because neutral has overlap; use a configurable overlap-growth tolerance and document that this is geometric feasibility, not validated physiology.

### Phase 2: Make training bounded and cancellable
5. Replace `BIOMECHANICS_OT_train_two_objects.execute()` with a modal operator using a Blender timer. Initialize proxies and the sample queue once, evaluate a small time-bounded batch per timer tick, update progress and elapsed time, and return control to Blender between batches.
6. Add Cancel support through `Esc` and a panel button. Cancellation must stop the timer, end progress reporting, delete temporary proxy data, restore all source transforms, and retain either no result or an explicitly marked partial result.
7. Add workload guards before training: display the planned sample count, estimated proxy complexity, and a hard maximum sample count. Require an explicit start action; reject configurations above the cap rather than allowing an accidental combinatorial run. Default the first sacrum/right-ilium run to a small coarse grid.
8. Generate the neutral pose first, then a coarse coupled 6-DOF grid. After the coarse pass, optionally refine only cells near validity/contact boundaries instead of increasing every axis globally. Keep deterministic ordering and a fixed refinement budget so runs are reproducible and bounded.

### Phase 3: Geometric evaluation and saved output
9. Cache the fixed proxy BVH once. For each moving pose, transform the proxy and build/evaluate only what is required for overlap and nearest-surface measurements. Record pose, overlap/contact metric, clearance, validity, and failure reason. Avoid full-resolution meshes inside the sampling loop.
10. Add a separate final-validation action that checks selected boundary poses against the full Rigid meshes. Do not run full-mesh validation for every coarse sample. Mark samples/results with whether they are proxy-only or full-mesh validated.
11. Save completed data to a Blender Text datablock as JSON using the version-2 schema. Include `complete`, elapsed time, sample counts, proxy counts, neutral baseline, object geometry fingerprints, and trainer settings so stale data can be detected after mesh or transform changes.
12. Update `properties.py` and `panels.py` with conservative defaults, planned sample count, progress/status, Start/Cancel controls, completion summary, and clear separation between proxy training and full-mesh validation. Keep fixed/moving object validation restricted to the selected Rigid collection.

### Phase 4: Spring-ready constraint layer
13. Add a constraint-evaluator interface that receives a saved relative pose and returns named energy, strain, and hard-limit values. Geometry validity remains a hard collision/contact result; constraint energy ranks geometrically valid poses.
14. Support tension-only ligament springs using attachment points stored in each bone's local coordinates, rest/slack length, stiffness, and optional maximum strain. Evaluate `extension = max(0, length - slack_length)` and store force/energy per ligament without changing the geometric sample.
15. Support a pubic-symphysis six-degree-of-freedom spring as a separate evaluator over the relative transform of the left and right ilia, with translation and rotation stiffness per axis plus hard compression/separation/shear limits. This belongs to later three-body pelvic-ring training, not the initial two-body SI run.
16. Add a `Re-score Constraints` action that loads compatible saved geometry samples and recomputes spring energies/validity only. Changing spring stiffness, slack length, or attachment points must not rebuild proxies or repeat collision training; changed meshes or neutral transforms must invalidate the geometry cache.
17. When spring data is ready for interaction, update `interaction.py` to query the nearest/interpolated valid low-energy pose rather than the current scalar translation/rotation clamp. Keep collision/maximum strain as hard constraints and spring energy as the preference among valid candidates.

### Phase 5: Documentation and cleanup
18. Update `README.md` with the safe training workflow, expected runtimes, cancellation, result location, and the distinction between geometric training and anatomical spring parameters. Update `DEVELOPMENT_PLAN.md` with format version 2, invariants, and the spring re-scoring workflow.
19. Remove or migrate format-version-1 assumptions after confirming version-2 output. Preserve clear errors for stale or partial datasets and clean all temporary objects, meshes, handlers, and timers during add-on unregister/reload.

**Relevant files**
- `c:/src/blender-biomechanics/biomechanics_addon/training.py` — split the synchronous trainer into bounded proxy generation, modal training state, geometry evaluators, serialization, and later constraint re-scoring.
- `c:/src/blender-biomechanics/biomechanics_addon/operators.py` — convert `BIOMECHANICS_OT_train_two_objects` to modal lifecycle; add cancel, full-validation, and future re-score operators.
- `c:/src/blender-biomechanics/biomechanics_addon/properties.py` — add workload limits, status/progress, overlap tolerance, refinement settings, and future spring settings/data pointers.
- `c:/src/blender-biomechanics/biomechanics_addon/panels.py` — show estimated workload, Start/Cancel state, proxy statistics, and saved-result status.
- `c:/src/blender-biomechanics/biomechanics_addon/interaction.py` — later consume trained valid/energy-scored poses; do not change interaction behavior until training output is validated.
- `c:/src/blender-biomechanics/biomechanics_addon/__init__.py` — register new operators in dependency order and guarantee timer/proxy cleanup during unregister.
- `c:/src/blender-biomechanics/README.md` — document user workflow and geometric-versus-anatomical meaning.
- `c:/src/blender-biomechanics/DEVELOPMENT_PLAN.md` — record schema, invariants, validation, and phased pelvic-ring extension.

**Verification**
1. Run `python -m compileall biomechanics_addon` after each implementation phase.
2. Reload with `tools/dev_load.py`; verify no duplicate timers, handlers, temporary proxy objects, or orphan training meshes remain after repeated reloads.
3. Train only the neutral sacrum/right-ilium pose first. Verify both source matrices and mesh datablocks are unchanged, proxy counts obey the cap, and the neutral baseline is serialized.
4. Run a deliberately small coupled grid and confirm Blender remains responsive, progress advances, `Esc`/Cancel cleans up immediately, and the saved sample count matches the estimate.
5. Attempt an oversized grid and verify it is rejected before proxy/sample work begins.
6. Complete a bounded coarse run; inspect JSON format version 2, units, fingerprints, completion flag, settings, proxy metadata, and per-pose metrics.
7. Select representative valid/invalid boundary poses and run full-mesh validation; compare classifications and record mismatches rather than silently overwriting proxy results.
8. Add a test ligament between local attachment points, re-score the existing dataset at two stiffness values, and verify collision metrics remain byte-for-byte unchanged while spring energy changes. Verify slack poses have zero force and over-limit strain creates a hard invalid reason.
9. Reload the add-on during and after training; confirm cancellation/cleanup leaves Rigid sacrum and ilium transforms unchanged.
10. Manually smoke-test the Biomechanics panel in Blender 3.6+ and retain the existing alignment/comparison checks from `DEVELOPMENT_PLAN.md`.

**Decisions**
- Initial scope is two Rigid meshes: fixed sacrum and moving right ilium. Original meshes are never training inputs.
- The first deliverable learns geometric feasibility only; it does not claim anatomical SI freedom from mesh shape alone.
- Training runs on Blender's main thread in small modal timer batches because Blender mesh/BVH APIs are not generally thread-safe; responsiveness comes from yielding between batches, not unsafe background access.
- Dense unbounded `steps^6` execution is excluded. Coarse deterministic sampling plus bounded adaptive refinement is the selected approach.
- Spring constraints are explicitly supported later through re-scoring. Ligaments are tension-only; the future pubic connection is a 6-DOF spring in the three-body pelvic solver.
- Geometry must be retrained when meshes or neutral transforms change. Spring parameter changes alone require only re-scoring.

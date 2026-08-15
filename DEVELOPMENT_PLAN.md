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
3. Select one mesh object from Original.
4. Choose a comparison collection: **None**, **Original**, or **Rigid**.
5. Optionally use **Mesh Wireframe**, **Overlay Opacity**, and **Overlay On Top**.
6. Select **Align to Source**.
7. Inspect how nearby original and rigid meshes differ relative to the aligned pair.

## UI Controls

### Original

Collection picker stored in `Scene.biomechanics_source_collection`.

- Contains the model that will be moved by **Align to Source**.
- Defaults to the collection named `Collection` when available.
- A valid selected alignment object must be a mesh directly in this collection.
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
2. Validate that the active object is a mesh in Original.
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

Blender property update callback. It finds an active Original mesh, derives its base
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
- `biomechanics_addon/interaction.py`: target-driven constrained transforms.
- `biomechanics_addon/operators.py`: alignment action and user-facing validation.
- `biomechanics_addon/panels.py`: sidebar layout and control enabled states.
- `biomechanics_addon/properties.py`: scene settings and reversible display state.
- `tools/dev_load.py`: clean package/submodule reload during Blender development.

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

## Next Development Priorities

1. Replace the prototype scalar pose limits with coupled SI joint limits and
   collision/contact evaluation over the sacrum and both ilia in Rigid.
2. Add the pubic-symphysis spring and solve movement of the opposite rigid ilium.
3. Rename legacy `transparency` symbols to `comparison_display` in one semantics-aware
   refactor, preserving saved Blender property migration if needed.
4. Move pairing into a dedicated function and expose clearer diagnostics for missing,
   duplicate, or incompatible pairs.
5. Add an optional panel status line showing the selected Original mesh and resolved
   Rigid pair before alignment.
6. Consider a lower-density wireframe visualization because anatomical meshes can be
   very dense and visually dark in full `WIRE` mode.
7. Add automated tests for pure name normalization and Kabsch math using Blender's
   background test runner when a stable test harness is introduced.

## Known Constraints

- NumPy and `mathutils` are supplied by Blender; a normal host Python import is not a
  valid add-on test.
- Pairing requires corresponding vertex indices, not merely similar-looking meshes.
- The current comparison state is global across Blender screens and objects, with
  exact restoration managed through temporary object custom properties and an
  in-memory viewport map.
- Blender UI selection changes alone do not invoke a custom callback. The pair
  exclusion refreshes when a display property changes or after Align to Source runs.
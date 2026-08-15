# Original Collection Nearest-Neighbor Map

This map was calculated from the 47 mesh objects in the Blender **Original**
collection (`Collection`). Distances are approximate nearest-surface distances in
millimeters, calculated from world-space mesh geometry rather than object origins.

| Bone | Nearest neighbor | Distance |
|---|---|---:|
| L1 | T12 | 0.187 mm |
| L2 | L1 | 0.593 mm |
| L3 | L2 | 1.003 mm |
| L4 | L3 | 1.061 mm |
| L5 | sacrum | 0.420 mm |
| T01 | T02 | 0.154 mm |
| T01 rib L | T01 | 1.041 mm |
| T01 rib R | T01 | 0.936 mm |
| T02 | T01 | 0.154 mm |
| T02 rib L | T02 | 1.998 mm |
| T02 rib R | T02 | 1.319 mm |
| T03 | T04 | 0.001 mm |
| T03 rib L | T03 | 0.032 mm |
| T03 rib R | T03 | 0.352 mm |
| T04 | T03 | 0.001 mm |
| T04 rib L | T04 | 0.429 mm |
| T04 rib R | T04 | 1.055 mm |
| T05 | T06 | 0.005 mm |
| T05 rib L | T05 | 0.353 mm |
| T05 rib R | T05 | 0.474 mm |
| T06 | T05 | 0.005 mm |
| T06 rib L | T06 | 0.512 mm |
| T06 rib R | T06 | 0.326 mm |
| T07 | T06 | 0.012 mm |
| T07 rib L | T07 | 0.017 mm |
| T07 rib R | T07 | 0.275 mm |
| T08 | T08 rib L | 0.036 mm |
| T08 rib L | T08 | 0.036 mm |
| T08 rib R | T08 | 0.406 mm |
| T09 | T08 | 0.375 mm |
| T09 rib L | T08 | 1.066 mm |
| T09 rib R | T09 | 0.443 mm |
| T10 | T09 | 0.008 mm |
| T10 rib L | T10 | 1.716 mm |
| T10 rib R | T10 | 1.559 mm |
| T11 | T10 | 0.304 mm |
| T11 rib L | T11 | 1.093 mm |
| T11 rib R | T11 | 1.378 mm |
| T12 | L1 | 0.187 mm |
| T12 rib L | T12 | 4.772 mm |
| T12 rib R | T12 | 2.611 mm |
| femur L | ilium L | 0.307 mm |
| femur R | ilium R | 0.434 mm |
| ilium L | femur L | 0.307 mm |
| ilium R | femur R | 0.434 mm |
| sacrum | L5 | 0.420 mm |
| strenum | T01 rib L | 12.680 mm |

## Method

For each mesh, the calculation:

1. Transformed mesh geometry into world space.
2. Built a BVH tree from the complete polygon surface.
3. Shortlisted 12 candidate neighbors using world-space bounding-box distance.
4. Queried up to 2,500 distributed vertices in both directions against the BVH
   surfaces.
5. Selected the candidate with the smallest measured surface distance.
6. Converted Blender distances to millimeters using the scene unit scale.

The result is approximate because the query samples vertices, although nearest
points are evaluated against complete polygon surfaces.

## Interpretation

This is a geometric proximity map, not an anatomical joint-connectivity map. The
closest surface to an ilium is its femur in this model, so the sacroiliac connection
cannot be inferred from only the single nearest whole-mesh distance. SI joints and
other known anatomical connections must be recorded explicitly for biomechanical
training and constraint calculations.

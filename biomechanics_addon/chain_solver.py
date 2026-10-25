"""Quasi-static XPBD solver for a chain of rigid bones (bpy-free).

Bodies are rigid: a position (the bone's mass center), a unit quaternion, an
inverse mass, and an isotropic inverse inertia. Fixed bodies have zero inverse
mass. Three constraint kinds are projected with extended position-based dynamics
(Müller et al. 2020, "Detailed Rigid Body Simulation with XPBD"):

- Distance constraints between attachment points on two bodies (ligaments and disc
  springs). Mode "tension" resists only stretching beyond `rest * (1 + slack)`
  (ligaments), "compression" only shortening below rest (disc struts), "both" acts
  as a spring.
- Contact constraints: surface sample points of one body that lie inside another
  body's signed distance field are pushed out along the field gradient.
- Pose constraints: a handle body is pulled toward a target position and
  orientation with finite compliance, so a target pushed into bone stops at bone.
- Muscle elements: a string between two attachment points pulls them together
  with a displacement per substep of `activation * strength` (a force, applied the
  same way as gravity), where activation in [0, 1] is shared by every string of
  one muscle group. Muscles never push.

Constraints are grouped by the pair of bodies they join. Groups are projected one
after another along the chain (Gauss-Seidel between groups, alternating direction
each sweep) and batched within a group: each constraint computes its own XPBD
step, the step is divided by the number of constraints in the batch that are
moving the body, and that divided value is what is applied and accumulated as the
multiplier. The multiplier therefore equals the force actually applied (plain
averaging leaves a compliant spring thinking it carries n times its real force,
so its effective compliance grows n-fold), and slack constraints do not dilute
the active ones. Collision samples are screened
once per substep against the other bone's field; only samples within a band of
its surface are re-tested in every sweep.

There is no time integration; every solve starts from the current pose and runs
`substeps` rounds of `iterations` sweeps. One round of XPBD also pulls toward the
starting pose (the inertial term), so forces between constraints only balance
across rounds. Distance-constraint multipliers are warm-started: each round first
re-applies `warm_start` times the forces accumulated in the previous round, so a
steady load (gravity, muscle tone) is met by the forces that balanced it last
time instead of being partly absorbed each round, which would otherwise make
the chain creep until bone contact stops it. The handle's pose constraint moves at most `max_position_step` per
sweep, so a target dragged through bone walks up to the surface and stops there
instead of tunneling. The solve is deterministic for a given input, which makes
it a clean ground truth for a later learned surrogate. Positions are in Blender
units (meters).
"""

import numpy as np


def cross(a, b):
    """Row-wise cross product of (N, 3) arrays; np.cross is slow for small arrays."""
    out = np.empty_like(a)
    out[:, 0] = a[:, 1] * b[:, 2] - a[:, 2] * b[:, 1]
    out[:, 1] = a[:, 2] * b[:, 0] - a[:, 0] * b[:, 2]
    out[:, 2] = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    return out


def quat_multiply(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ]
    )


def quat_conjugate(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_to_matrix(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def quats_to_matrices(quaternions):
    """(B, 4) unit quaternions -> (B, 3, 3) rotation matrices."""
    q = quaternions / np.linalg.norm(quaternions, axis=1, keepdims=True)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    matrices = np.empty((len(q), 3, 3))
    matrices[:, 0, 0] = 1 - 2 * (y * y + z * z)
    matrices[:, 0, 1] = 2 * (x * y - z * w)
    matrices[:, 0, 2] = 2 * (x * z + y * w)
    matrices[:, 1, 0] = 2 * (x * y + z * w)
    matrices[:, 1, 1] = 1 - 2 * (x * x + z * z)
    matrices[:, 1, 2] = 2 * (y * z - x * w)
    matrices[:, 2, 0] = 2 * (x * z - y * w)
    matrices[:, 2, 1] = 2 * (y * z + x * w)
    matrices[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return matrices


def matrix_to_quat(m):
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        return np.array([0.25 / s, (m[2, 1] - m[1, 2]) * s, (m[0, 2] - m[2, 0]) * s, (m[1, 0] - m[0, 1]) * s])
    if m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        return np.array([(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s])
    if m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        return np.array([(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s])
    s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
    return np.array([(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s])


class Body:
    def __init__(self, name, position, quaternion, mass=1.0, radius=0.03, fixed=False):
        self.name = name
        self.position = np.asarray(position, dtype=np.float64).copy()
        self.quaternion = np.asarray(quaternion, dtype=np.float64).copy()
        self.quaternion /= np.linalg.norm(self.quaternion)
        self.mass = mass
        self.radius = radius
        self.inverse_mass = 0.0 if fixed else 1.0 / mass
        # Solid sphere of the bone's size; isotropic, so body and world inertia agree.
        self.inverse_inertia = 0.0 if fixed else 1.0 / (0.4 * mass * radius * radius)
        self.fixed = fixed
        self.index = -1  # assigned by ChainSolver

    @property
    def rotation(self):
        return quat_to_matrix(self.quaternion)

    def world_point(self, local):
        return self.position + self.rotation @ np.asarray(local, dtype=np.float64)

    def local_point(self, world):
        return self.rotation.T @ (np.asarray(world, dtype=np.float64) - self.position)

    def matrix(self):
        m = np.eye(4)
        m[:3, :3] = self.rotation
        m[:3, 3] = self.position
        return m

    def apply_rotation(self, angular):
        if self.fixed:
            return
        self.quaternion += 0.5 * quat_multiply(np.array([0.0, *angular]), self.quaternion)
        self.quaternion /= np.linalg.norm(self.quaternion)


class DistanceConstraint:
    MODES = {"both": 0, "tension": 1, "compression": 2}

    def __init__(
        self, body_a, local_a, body_b, local_b, rest_length, compliance=0.0, unilateral=False, slack=0.0, name="", mode=None
    ):
        self.body_a = body_a
        self.body_b = body_b
        self.local_a = np.asarray(local_a, dtype=np.float64)
        self.local_b = np.asarray(local_b, dtype=np.float64)
        self.rest_length = float(rest_length)
        self.compliance = float(compliance)
        self.mode = mode if mode is not None else ("tension" if unilateral else "both")
        self.unilateral = self.mode == "tension"
        self.slack = float(slack)
        self.name = name
        self.lagrange = 0.0

    def current_length(self):
        return float(np.linalg.norm(self.body_b.world_point(self.local_b) - self.body_a.world_point(self.local_a)))

    def strain(self):
        return self.current_length() / self.rest_length - 1.0 if self.rest_length > 0.0 else 0.0


class MuscleElement:
    """One muscle string: pulls local_a on body_a toward local_b on body_b when active."""

    def __init__(self, body_a, local_a, body_b, local_b, group, strength, name=""):
        self.body_a = body_a
        self.body_b = body_b
        self.local_a = np.asarray(local_a, dtype=np.float64)
        self.local_b = np.asarray(local_b, dtype=np.float64)
        self.group = int(group)
        self.strength = float(strength)
        self.name = name

    def current_length(self):
        return float(np.linalg.norm(self.body_b.world_point(self.local_b) - self.body_a.world_point(self.local_a)))


class PoseConstraint:
    """Pull a body toward a target position and orientation with compliance."""

    def __init__(
        self,
        body,
        target_position,
        target_quaternion,
        position_compliance=1e-4,
        rotation_compliance=1e-4,
        max_position_step=0.001,
        max_rotation_step=0.02,
    ):
        self.body = body
        self.target_position = np.asarray(target_position, dtype=np.float64)
        self.target_quaternion = np.asarray(target_quaternion, dtype=np.float64)
        self.position_compliance = position_compliance
        self.rotation_compliance = rotation_compliance
        self.max_position_step = max_position_step
        self.max_rotation_step = max_rotation_step
        self.position_lagrange = np.zeros(3)
        self.rotation_lagrange = np.zeros(3)

    def project(self):
        body = self.body
        if body.fixed:
            return
        error = body.position - self.target_position
        distance = np.linalg.norm(error)
        if distance > 1e-12:
            normal = error / distance
            w = body.inverse_mass
            total = w + self.position_compliance
            lagrange_along = float(self.position_lagrange @ normal)
            delta = (-distance - self.position_compliance * lagrange_along) / total
            step = delta * w
            if abs(step) > self.max_position_step:
                step = np.sign(step) * self.max_position_step
                delta = step / w
            self.position_lagrange += delta * normal
            body.position += step * normal
        relative = quat_multiply(self.target_quaternion, quat_conjugate(body.quaternion))
        if relative[0] < 0.0:
            relative = -relative
        rotation_vector = 2.0 * relative[1:]
        angle = np.linalg.norm(rotation_vector)
        if angle < 1e-9:
            return
        axis = rotation_vector / angle
        w = body.inverse_inertia
        total = w + self.rotation_compliance
        lagrange_along = float(self.rotation_lagrange @ axis)
        delta = (angle - self.rotation_compliance * lagrange_along) / total
        step = delta * w
        if abs(step) > self.max_rotation_step:
            step = np.sign(step) * self.max_rotation_step
            delta = step / w
        self.rotation_lagrange += delta * axis
        body.apply_rotation(step * axis)


class CollisionPair:
    """Samples of body_a tested against body_b's distance field, and vice versa."""

    def __init__(self, body_a, samples_a, field_a, body_b, samples_b, field_b, margin=0.0, band=0.003, max_contacts=4):
        self.body_a = body_a
        self.samples_a = np.asarray(samples_a, dtype=np.float64)
        self.field_a = field_a
        self.body_b = body_b
        self.samples_b = np.asarray(samples_b, dtype=np.float64)
        self.field_b = field_b
        self.margin = margin
        self.band = band
        self.max_contacts = max_contacts
        self.last_penetration = 0.0
        self._candidates = (self.samples_a, self.samples_b)

    def directions(self):
        return (
            (self.body_a, self.samples_a, self.body_b, self.field_b),
            (self.body_b, self.samples_b, self.body_a, self.field_a),
        )

    def screen(self, rotations, positions, travel=0.0):
        """Keep only samples within band + travel of the other bone's surface."""
        kept = []
        for sample_body, samples, field_body, field in self.directions():
            world = samples @ rotations[sample_body.index].T + positions[sample_body.index]
            local = (world - positions[field_body.index]) @ rotations[field_body.index]
            distances = field.sample(local)
            kept.append(samples[distances < self.margin + self.band + travel])
        self._candidates = tuple(kept)

    def contacts(self, rotations, positions):
        """Return [(sample body index, field body index, world points, normals, violation)]."""
        results = []
        depth = 0.0
        for (sample_body, _samples, field_body, field), samples in zip(self.directions(), self._candidates):
            if len(samples) == 0:
                continue
            world = samples @ rotations[sample_body.index].T + positions[sample_body.index]
            rotation_field = rotations[field_body.index]
            local = (world - positions[field_body.index]) @ rotation_field
            distances = field.sample(local)
            penetrating = np.flatnonzero(distances < self.margin)
            if penetrating.size == 0:
                continue
            if penetrating.size > self.max_contacts:
                # Deepest few only: corrections are averaged per body, so a crowd of
                # shallow contacts would dilute the push that matters.
                penetrating = penetrating[np.argsort(distances[penetrating])[: self.max_contacts]]
            _distances, gradients = field.sample_with_gradient(local[penetrating])
            normals = gradients @ rotation_field.T
            violation = self.margin - distances[penetrating]
            depth = max(depth, float(violation.max()))
            results.append(
                (
                    np.full(penetrating.size, sample_body.index),
                    np.full(penetrating.size, field_body.index),
                    world[penetrating],
                    normals,
                    violation,
                )
            )
        self.last_penetration = depth
        return results


class ChainSolver:
    def __init__(
        self,
        bodies,
        distance_constraints,
        collision_pairs,
        pose_constraints=(),
        iterations=8,
        substeps=1,
        gravity=(0.0, 0.0, 0.0),
        muscles=(),
        group_count=0,
        warm_start=1.0,
    ):
        self.bodies = list(bodies)
        for index, body in enumerate(self.bodies):
            body.index = index
        self.distance_constraints = list(distance_constraints)
        self.collision_pairs = list(collision_pairs)
        self.pose_constraints = list(pose_constraints)
        self.muscles = list(muscles)
        self.activations = np.zeros(max(group_count, max((m.group for m in self.muscles), default=-1) + 1))
        self.iterations = iterations
        self.substeps = substeps
        # Displacement every free body receives at the start of each substep (the XPBD
        # position prediction under a constant acceleration); constraints then hold it.
        self.gravity = np.asarray(gravity, dtype=np.float64)
        self.warm_start = warm_start
        self._build_arrays()

    def _build_arrays(self):
        constraints = self.distance_constraints
        self.c_a = np.array([c.body_a.index for c in constraints], dtype=np.int64)
        self.c_b = np.array([c.body_b.index for c in constraints], dtype=np.int64)
        self.c_local_a = np.array([c.local_a for c in constraints]).reshape(-1, 3)
        self.c_local_b = np.array([c.local_b for c in constraints]).reshape(-1, 3)
        self.c_rest = np.array([c.rest_length for c in constraints])
        self.c_compliance = np.array([c.compliance for c in constraints])
        self.c_mode = np.array([DistanceConstraint.MODES[c.mode] for c in constraints], dtype=np.int64)
        self.c_unilateral = self.c_mode == 1
        slack = np.array([c.slack for c in constraints])
        self.c_target = self.c_rest * np.where(self.c_unilateral, 1.0 + slack, 1.0)
        self.c_lagrange = np.zeros(len(constraints))
        self.inverse_mass = np.array([b.inverse_mass for b in self.bodies])
        self.inverse_inertia = np.array([b.inverse_inertia for b in self.bodies])
        self.m_a = np.array([m.body_a.index for m in self.muscles], dtype=np.int64)
        self.m_b = np.array([m.body_b.index for m in self.muscles], dtype=np.int64)
        self.m_local_a = np.array([m.local_a for m in self.muscles]).reshape(-1, 3)
        self.m_local_b = np.array([m.local_b for m in self.muscles]).reshape(-1, 3)
        self.m_group = np.array([m.group for m in self.muscles], dtype=np.int64)
        self.m_strength = np.array([m.strength for m in self.muscles])
        groups = {}
        joints = []
        for index, (a, b) in enumerate(zip(self.c_a, self.c_b)):
            if self.c_mode[index] == 0 and self.c_rest[index] == 0.0:
                joints.append(index)  # zero-length link: a ball joint, projected alone
            else:
                groups.setdefault((min(a, b), max(a, b)), []).append(index)
        # Ordered along the chain so a sweep propagates from the base outward; joints last
        # and on their own so nothing in the batch dilutes them.
        self.c_groups = [np.array(groups[key], dtype=np.int64) for key in sorted(groups)]
        self.c_groups += [np.array([index], dtype=np.int64) for index in joints]
        self._sweep = 0

    def _state(self):
        positions = np.array([b.position for b in self.bodies])
        quaternions = np.array([b.quaternion for b in self.bodies])
        return positions, quaternions

    # Safety valve: no body moves more than this per correction pass. A diverging
    # configuration then stalls instead of flinging bones across the scene.
    MAX_LINEAR_STEP = 0.004  # meters
    MAX_ANGULAR_STEP = 0.12  # radians

    def _apply(self, positions, quaternions, linear, angular, counts):
        scale = 1.0 / np.maximum(counts, 1.0)
        step = linear * (self.inverse_mass * scale)[:, None]
        magnitude = np.linalg.norm(step, axis=1)
        too_far = magnitude > self.MAX_LINEAR_STEP
        if too_far.any():
            step[too_far] *= (self.MAX_LINEAR_STEP / magnitude[too_far])[:, None]
        positions += step
        angular = angular * (self.inverse_inertia * scale)[:, None]
        spin = np.linalg.norm(angular, axis=1)
        too_fast = spin > self.MAX_ANGULAR_STEP
        if too_fast.any():
            angular[too_fast] *= (self.MAX_ANGULAR_STEP / spin[too_fast])[:, None]
        # q += 0.5 * (0, angular) * q
        w, x, y, z = quaternions[:, 0].copy(), quaternions[:, 1].copy(), quaternions[:, 2].copy(), quaternions[:, 3].copy()
        ax, ay, az = angular[:, 0], angular[:, 1], angular[:, 2]
        quaternions[:, 0] += 0.5 * (-ax * x - ay * y - az * z)
        quaternions[:, 1] += 0.5 * (ax * w + ay * z - az * y)
        quaternions[:, 2] += 0.5 * (ay * w + az * x - ax * z)
        quaternions[:, 3] += 0.5 * (az * w + ax * y - ay * x)
        quaternions /= np.linalg.norm(quaternions, axis=1, keepdims=True)

    def _project_batch(
        self, positions, quaternions, index_a, index_b, point_a, point_b, normals, violation, compliance, lagrange, lower=None, upper=None
    ):
        """XPBD batch: move points a toward b along normals by violation. Returns delta lagrange.

        lower/upper clamp the accumulated multiplier (tension-only: upper 0;
        compression-only: lower 0), so a unilateral constraint that goes slack keeps
        its memory instead of being reset, which the warm start relies on.
        """
        count = len(self.bodies)
        offset_a = point_a - positions[index_a]
        offset_b = point_b - positions[index_b]
        lever_a = cross(offset_a, normals)
        lever_b = cross(offset_b, normals)
        w_a = self.inverse_mass[index_a] + self.inverse_inertia[index_a] * np.einsum("ij,ij->i", lever_a, lever_a)
        w_b = self.inverse_mass[index_b] + self.inverse_inertia[index_b] * np.einsum("ij,ij->i", lever_b, lever_b)
        total = w_a + w_b + compliance
        delta = np.where(total > 0.0, (-violation - compliance * lagrange) / np.maximum(total, 1e-30), 0.0)
        if lower is not None or upper is not None:
            delta = np.clip(lagrange + delta, lower, upper) - lagrange
        moving = delta != 0.0
        if not moving.any():
            return delta
        indices = np.concatenate((index_a[moving], index_b[moving]))
        counts = np.bincount(indices, minlength=count).astype(np.float64)
        share = np.maximum(np.maximum(counts[index_a], counts[index_b]), 1.0)
        delta = delta / share  # the multiplier actually applied
        impulse = delta[:, None] * normals
        torque_a = -cross(offset_a, impulse)
        torque_b = cross(offset_b, impulse)
        all_indices = np.concatenate((index_a, index_b))
        linear_terms = np.concatenate((-impulse, impulse))
        angular_terms = np.concatenate((torque_a, torque_b))
        # bincount is far cheaper than add.at for these small scatters.
        linear = np.stack([np.bincount(all_indices, weights=linear_terms[:, axis], minlength=count) for axis in range(3)], axis=1)
        angular = np.stack([np.bincount(all_indices, weights=angular_terms[:, axis], minlength=count) for axis in range(3)], axis=1)
        self._apply(positions, quaternions, linear, angular, np.ones(count))
        return delta

    def _project_group(self, positions, quaternions, members):
        index_a, index_b = self.c_a[members], self.c_b[members]
        rotation_a = quat_to_matrix(quaternions[index_a[0]])
        rotation_b = quat_to_matrix(quaternions[index_b[0]])
        point_a = self.c_local_a[members] @ rotation_a.T + positions[index_a]
        point_b = self.c_local_b[members] @ rotation_b.T + positions[index_b]
        delta = point_b - point_a
        length = np.linalg.norm(delta, axis=1)
        violation = length - self.c_target[members]
        mode = self.c_mode[members]
        active = length > 1e-12
        if not active.any():
            return
        chosen = members[active]
        mode = mode[active]
        # Tension-only multipliers stay <= 0, compression-only >= 0; springs are unbounded.
        lower = np.where(mode == 2, 0.0, -np.inf)
        upper = np.where(mode == 1, 0.0, np.inf)
        normals = delta[active] / length[active][:, None]
        changed = self._project_batch(
            positions,
            quaternions,
            index_a[active],
            index_b[active],
            point_a[active],
            point_b[active],
            normals,
            violation[active],
            self.c_compliance[chosen],
            self.c_lagrange[chosen],
            lower,
            upper,
        )
        self.c_lagrange[chosen] += changed

    def _apply_muscles(self, positions, quaternions):
        """Pull each active muscle string's ends together as impulses at the attachments."""
        if len(self.m_a) == 0:
            return
        magnitude = self.activations[self.m_group] * self.m_strength
        active = magnitude > 0.0
        if not active.any():
            return
        index_a, index_b = self.m_a[active], self.m_b[active]
        rotations = quats_to_matrices(quaternions)
        point_a = np.einsum("nij,nj->ni", rotations[index_a], self.m_local_a[active]) + positions[index_a]
        point_b = np.einsum("nij,nj->ni", rotations[index_b], self.m_local_b[active]) + positions[index_b]
        delta = point_b - point_a
        length = np.linalg.norm(delta, axis=1)
        ok = length > 1e-9
        if not ok.any():
            return
        normals = delta[ok] / length[ok][:, None]
        impulse = (magnitude[active][ok])[:, None] * normals  # on a, toward b
        offset_a = point_a[ok] - positions[index_a[ok]]
        offset_b = point_b[ok] - positions[index_b[ok]]
        indices = np.concatenate((index_a[ok], index_b[ok]))
        linear_terms = np.concatenate((impulse, -impulse))
        angular_terms = np.concatenate((cross(offset_a, impulse), -cross(offset_b, impulse)))
        count = len(self.bodies)
        linear = np.stack([np.bincount(indices, weights=linear_terms[:, axis], minlength=count) for axis in range(3)], axis=1)
        angular = np.stack([np.bincount(indices, weights=angular_terms[:, axis], minlength=count) for axis in range(3)], axis=1)
        # Forces add up; no averaging.
        self._apply(positions, quaternions, linear, angular, np.ones(count))

    def _apply_warm_start(self, positions, quaternions):
        """Re-apply last round's constraint forces (scaled) before the sweeps."""
        if self.warm_start <= 0.0 or len(self.c_a) == 0:
            return
        self.c_lagrange *= self.warm_start
        carried = np.flatnonzero(np.abs(self.c_lagrange) > 1e-15)
        if carried.size == 0:
            return
        index_a, index_b = self.c_a[carried], self.c_b[carried]
        rotations = quats_to_matrices(quaternions)
        point_a = np.einsum("nij,nj->ni", rotations[index_a], self.c_local_a[carried]) + positions[index_a]
        point_b = np.einsum("nij,nj->ni", rotations[index_b], self.c_local_b[carried]) + positions[index_b]
        delta = point_b - point_a
        length = np.linalg.norm(delta, axis=1)
        ok = length > 1e-12
        if not ok.any():
            return
        normals = delta[ok] / length[ok][:, None]
        # Same sign convention as _project_batch: impulse = lambda * n, -impulse on a, +impulse on b.
        impulse = self.c_lagrange[carried][ok][:, None] * normals
        offset_a = point_a[ok] - positions[index_a[ok]]
        offset_b = point_b[ok] - positions[index_b[ok]]
        indices = np.concatenate((index_a[ok], index_b[ok]))
        linear_terms = np.concatenate((-impulse, impulse))
        angular_terms = np.concatenate((-cross(offset_a, impulse), cross(offset_b, impulse)))
        count = len(self.bodies)
        linear = np.stack([np.bincount(indices, weights=linear_terms[:, axis], minlength=count) for axis in range(3)], axis=1)
        angular = np.stack([np.bincount(indices, weights=angular_terms[:, axis], minlength=count) for axis in range(3)], axis=1)
        # Multipliers are the forces actually applied, so they sum without averaging.
        self._apply(positions, quaternions, linear, angular, np.ones(count))

    def reset_warm_start(self):
        self.c_lagrange[:] = 0.0

    def muscle_lengths(self):
        return np.array([m.current_length() for m in self.muscles])

    def _project_distances(self, positions, quaternions):
        groups = self.c_groups if self._sweep % 2 == 0 else self.c_groups[::-1]
        for members in groups:
            self._project_group(positions, quaternions, members)

    def _project_contacts(self, positions, quaternions):
        pairs = self.collision_pairs if self._sweep % 2 == 0 else self.collision_pairs[::-1]
        rotations = quats_to_matrices(quaternions)
        for pair in pairs:
            # Refresh only the two rotations this pair reads; earlier pairs may have moved them.
            for body in (pair.body_a, pair.body_b):
                rotations[body.index] = quat_to_matrix(quaternions[body.index])
            batches = pair.contacts(rotations, positions)
            if not batches:
                continue
            sample_index = np.concatenate([batch[0] for batch in batches])
            field_index = np.concatenate([batch[1] for batch in batches])
            points = np.concatenate([batch[2] for batch in batches])
            normals = np.concatenate([batch[3] for batch in batches])
            violation = np.concatenate([batch[4] for batch in batches])
            # A positive violation moves body a along +normal in _project_batch, which for
            # a contact is the sample body moving out along the field's outward normal.
            self._project_batch(
                positions,
                quaternions,
                sample_index,
                field_index,
                points,
                points,
                normals,
                violation,
                np.zeros(len(violation)),
                np.zeros(len(violation)),
            )

    def _write_back(self, positions, quaternions):
        for body, position, quaternion in zip(self.bodies, positions, quaternions):
            if body.fixed:
                continue
            body.position[:] = position
            body.quaternion[:] = quaternion

    def solve(self):
        positions, quaternions = self._state()
        # A body can travel at most one pose step per sweep, so screening once per
        # substep with that allowance cannot miss a contact.
        travel = max((pose.max_position_step for pose in self.pose_constraints), default=0.0) * self.iterations
        travel += float(np.linalg.norm(self.gravity))
        free = self.inverse_mass > 0.0
        for _substep in range(self.substeps):
            if free.any() and np.any(self.gravity):
                positions[free] += self.gravity
            self._apply_muscles(positions, quaternions)
            self._apply_warm_start(positions, quaternions)
            self._write_back(positions, quaternions)
            rotations = quats_to_matrices(quaternions)
            for pair in self.collision_pairs:
                pair.screen(rotations, positions, travel)
            if self.warm_start <= 0.0:
                self.c_lagrange[:] = 0.0
            for pose in self.pose_constraints:
                pose.position_lagrange[:] = 0.0
                pose.rotation_lagrange[:] = 0.0
            for _iteration in range(self.iterations):
                for pose in self.pose_constraints:
                    pose.project()
                    positions[pose.body.index] = pose.body.position
                    quaternions[pose.body.index] = pose.body.quaternion
                self._project_distances(positions, quaternions)
                self._project_contacts(positions, quaternions)
                self._write_back(positions, quaternions)
                self._sweep += 1
        for constraint, lagrange in zip(self.distance_constraints, self.c_lagrange):
            constraint.lagrange = float(lagrange)
        return {body.name: body.matrix() for body in self.bodies}

    def is_finite(self):
        return bool(np.all(np.isfinite([body.position for body in self.bodies])))

    def max_penetration(self):
        return max((pair.last_penetration for pair in self.collision_pairs), default=0.0)

    def max_strain(self, min_rest=0.003):
        """Largest strain among constraints with a meaningful rest length."""
        return max(
            (constraint.strain() for constraint in self.distance_constraints if constraint.rest_length >= min_rest),
            default=0.0,
        )

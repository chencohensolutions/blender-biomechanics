import re

import numpy as np
from mathutils import Matrix


def base_object_name(name):
    return re.sub(r"\.\d{3}$", "", name)


def world_vertices(obj):
    coordinates = np.empty(len(obj.data.vertices) * 3, dtype=np.float64)
    obj.data.vertices.foreach_get("co", coordinates)
    coordinates = coordinates.reshape((-1, 3))

    world_matrix = np.array(obj.matrix_world, dtype=np.float64)
    return coordinates @ world_matrix[:3, :3].T + world_matrix[:3, 3]


def mesh_alignment(source_object, target_object):
    if len(source_object.data.vertices) != len(target_object.data.vertices):
        raise ValueError("Paired meshes have different vertex counts")
    if len(source_object.data.polygons) != len(target_object.data.polygons):
        raise ValueError("Paired meshes have different polygon counts")
    if len(source_object.data.vertices) < 3:
        raise ValueError("Paired meshes need at least three vertices")

    source_vertices = world_vertices(source_object)
    target_vertices = world_vertices(target_object)
    source_center = source_vertices.mean(axis=0)
    target_center = target_vertices.mean(axis=0)
    centered_source = source_vertices - source_center
    centered_target = target_vertices - target_center

    covariance = centered_source.T @ centered_target
    left_vectors, _, right_vectors = np.linalg.svd(covariance)
    rotation = right_vectors.T @ left_vectors.T
    if np.linalg.det(rotation) < 0:
        right_vectors[-1, :] *= -1
        rotation = right_vectors.T @ left_vectors.T

    translation = target_center - rotation @ source_center
    aligned_vertices = source_vertices @ rotation.T + translation
    errors = np.linalg.norm(aligned_vertices - target_vertices, axis=1)
    target_size = np.linalg.norm(np.ptp(target_vertices, axis=0))
    tolerance = max(target_size * 1e-4, 1e-6)
    if np.sqrt(np.mean(errors**2)) > tolerance:
        raise ValueError("Paired meshes do not have matching vertex geometry")

    alignment = Matrix.Identity(4)
    for row in range(3):
        for column in range(3):
            alignment[row][column] = rotation[row, column]
        alignment[row][3] = translation[row]
    return alignment
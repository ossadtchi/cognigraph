import os.path as op
import numpy as np
import nibabel as nib

from ..gui.brain_visual import BrainMesh


def get_mesh_data_from_surfaces_dir(surfaces_dir, translucent=False,
                                    cortex_type='inflated'):
    surfaces_dir = op.join(surfaces_dir, 'surf')
    if surfaces_dir:
        surf_paths = [op.join(surfaces_dir, '{}.{}'.format(h, cortex_type))
                      for h in ('lh', 'rh')]
    else:
        raise NameError('surfaces_dir is not set')
    lh_mesh, rh_mesh = [nib.freesurfer.read_geometry(surf_path)
                        for surf_path in surf_paths]
    lh_vertexes, lh_faces = lh_mesh
    rh_vertexes, rh_faces = rh_mesh

    # Move all the vertices so that the lh has x (L-R) <= 0 and rh - >= 0
    lh_vertexes[:, 0] -= np.max(lh_vertexes[:, 0])
    rh_vertexes[:, 0] -= np.min(rh_vertexes[:, 0])

    # Combine two meshes
    vertices = np.r_[lh_vertexes, rh_vertexes]
    lh_vertex_cnt = lh_vertexes.shape[0]
    faces = np.r_[lh_faces, lh_vertex_cnt + rh_faces]

    # Move the mesh so that the center of the brain is at (0, 0, 0) (kinda)
    vertices[:, 1:2] -= np.mean(vertices[:, 1:2])

    mesh_data = BrainMesh(vertices=vertices, faces=faces)
    mesh_data.translucent = translucent
    if translucent:
        mesh_data.alpha = 0.2

    return mesh_data

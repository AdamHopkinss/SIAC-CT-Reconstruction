import odl
import numpy as np
from odl.phantom.noise import poisson_noise, white_noise

def make_space_2d(Nx=256, Ny=256, domain=[-1, 1, -1, 1], dtype="float32"):
    """
    Create a 2D uniform reconstruction space.
    """
    xmin, xmax, ymin, ymax = domain
    space = odl.uniform_discr(
        min_pt=[xmin, ymin],
        max_pt=[xmax, ymax],
        shape=[Nx, Ny],
        dtype=dtype
    )
    return space

def shepp_logan_2d(space, modified=True):
    """
    Create a 2D Shepp-Logan phantom on a given ODL space.
    """
    return odl.phantom.shepp_logan(space, modified=modified)

def parallel_geom_2d(angular_coverage=(-60, 60), step=1,
                     det_range = None, det_count = None):   # det_range=(-1.5, 1.5), det_count=400):
    """
    Create a 2D parallel-beam geometry.
    """
    a0, a1 = np.deg2rad(angular_coverage[0]), np.deg2rad(angular_coverage[1])
    n_angles = int((angular_coverage[1] - angular_coverage[0]) / step) + 1

    angles = odl.uniform_partition(a0, a1, n_angles)
    det = odl.uniform_partition(det_range[0], det_range[1], det_count)

    return odl.tomo.Parallel2dGeometry(angles, det)

def ray_transform_2d(space, geom, impl="astra_cuda"):
    """
    Create 2D ray transform.
    """
    return odl.tomo.RayTransform(space, geom, impl=impl)

def reconstruct_bp(sino, A):
    """
    Backprojection (adjoint of the ray transform).
    """
    return A.adjoint(sino)

def reconstruct_fbp(sino, A, filter_name="Ram-Lak"):
    """
    Filtered backprojection with specified filter.
    """
    fbp_op = odl.tomo.fbp_op(A, filter_type=filter_name)
    return fbp_op(sino)

def add_relative_gaussian_noise(data, rel_level=0.01, seed=0):
    noise = white_noise(data.space, mean=0.0, stddev=1.0, seed=seed)
    noise = noise / noise.norm()
    noise = noise * rel_level * data.norm()
    return data + noise


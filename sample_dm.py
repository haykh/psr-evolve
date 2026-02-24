from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

CONSTANTS = {
    "sun_position": np.array([-8.5, 0, 0]),
}


def xyz_to_Galactic(position):
    x, y, z = position
    dx = x + CONSTANTS["sun_position"][0]
    dy = y + CONSTANTS["sun_position"][1]
    dz = z + CONSTANTS["sun_position"][2]
    angl_l = np.arctan2(dy, dx)
    d_xy = (dx**2 + dy**2) ** 0.5
    dist = (dx**2 + dy**2 + dz**2) ** 0.5
    angl_b = np.arctan(dz / d_xy)
    return (
        dist * 1000,
        angl_l * 180 / np.pi,
        angl_b * 180 / np.pi,
    )


xarr = np.linspace(-25, 25, 512)
yarr = np.linspace(-25, 25, 512)
zarr = np.linspace(-2, 2, 64)

xx, yy, zz = np.meshgrid(xarr, yarr, zarr, indexing="ij")

positions = np.stack([xx.flatten(), yy.flatten(), zz.flatten()], axis=-1)
dist_arr, gal_l_arr, gal_b_arr = xyz_to_Galactic(positions.T)

import pygedm


@np.vectorize
def dist_to_dm_vec(gl, gb, dist):
    return pygedm.dist_to_dm(gl, gb, dist)[0].value


dm_arr_local = dist_to_dm_vec(
    gal_l_arr[rank::size], gal_b_arr[rank::size], dist_arr[rank::size]
)

if rank == 0:
    dm_arr = np.empty_like(dist_arr)
else:
    dm_arr = None

comm.Gather(dm_arr_local, dm_arr, root=0)

if rank == 0 and dm_arr is not None:
    np.savez("dm_grid.npz", gal_l=gal_l_arr, gal_b=gal_b_arr, dist=dist_arr, dm=dm_arr)

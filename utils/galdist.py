import numpy as np
from . import ykdist

ArrayOrFloat = np.ndarray | float

defaults = {
    "k_list": [4.25, 4.25, 4.89, 4.89],
    "r0_list": [3.48, 3.48, 4.9, 4.9],  # kpc
    "theta0_list": [1.57, 4.71, 4.09, 0.95],
    "zscale": 0.05,
    "vspread": 180,  # km/sec
}


class GalacticDistribution:
    """
    Generates random positions and velocities for pulsars in the Milky Way.

    Parameters
    ----------
    k_list : list[float], optional
        List of k values for the spiral arms, by default [4.25, 4.25, 4.89, 4.89]
    r0_list : list[float], optional
        List of r0 values for the spiral arms, by default [3.48, 3.48, 4.9, 4.9] [kpc]
    theta0_list : list[float], optional
        List of theta0 values for the spiral arms, by default [1.57, 4.71, 4.09, 0.95]
    zscale : float, optional
        Scale height of the Galactic disk, by default 0.05 [kpc]
    vspread : float, optional
        Spread of the velocity distribution, by default 180 [km/sec]
    seed : int, optional
        Seed for the random number generator, by default None
    **kwargs
        Additional keyword arguments for the YKdist class.
    """

    def __init__(self, **kwargs):
        # spiral arms
        self.k_list = kwargs.pop("k_list", defaults["k_list"])
        self.r0_list = kwargs.pop("r0_list", defaults["r0_list"])
        self.theta0_list = kwargs.pop("theta0_list", defaults["theta0_list"])
        self.krth = np.array([self.k_list, self.r0_list, self.theta0_list]).T

        # disk
        self.zscale = kwargs.pop("zscale", defaults["zscale"])
        self.yk_dist = ykdist.YKdist(**kwargs)

        # velocity spread
        self.vspread = kwargs.pop("vspread", defaults["vspread"])

        self.rng = np.random.default_rng(seed=kwargs.pop("seed", None))

    def gen_XYZ(self, size: int = 1) -> list[ArrayOrFloat]:
        """
        Generates random positions in the Galactic plane.

        Parameters
        ----------
        size : int, optional
            Number of random positions to generate, by default 1

        Returns
        -------
        list[ArrayOrFloat]
            Random x, y, and z positions in the Galactic plane.
        """
        rnd_k, rnd_r0, rnd_th0 = self.rng.choice(self.krth, size).T

        rnd_r = self.yk_dist.rvs(size)

        rnd_th = rnd_k * np.log(rnd_r / rnd_r0) + rnd_th0
        rnd_th += (
            self.rng.choice([-1, 1], size)
            * self.rng.uniform(0, 2 * np.pi, size)
            * np.exp(-0.35 * rnd_r)
        )

        rnd_x, rnd_y = rnd_r * np.cos(rnd_th), rnd_r * np.sin(rnd_th)

        rnd_phi = self.rng.uniform(0, 2 * np.pi, size)
        rnd_dr = 0.07 * rnd_r * self.rng.normal(0, 1, size)
        rnd_x += rnd_dr * np.cos(rnd_phi)
        rnd_y += rnd_dr * np.sin(rnd_phi)

        rnd_z = (
            self.rng.choice([-1, 1], size) * self.zscale * np.log(self.rng.random(size))
        )
        return rnd_x, rnd_y, rnd_z

    def gen_VXVYVZ(self, size: int = 1) -> list[ArrayOrFloat]:
        """
        Generates random velocities for pulsars in the Milky Way.

        Parameters
        ----------
        size : int, optional
            Number of random velocities to generate, by default 1

        Returns
        -------
        list[ArrayOrFloat]
            Random vx, vy, and vz velocities.
        """
        rnd_u = 2 * self.rng.random(size) - 1
        rnd_th = 2 * np.pi * self.rng.random(size)
        rnd_kx = np.sqrt(1 - rnd_u**2) * np.cos(rnd_th)
        rnd_ky = np.sqrt(1 - rnd_u**2) * np.sin(rnd_th)
        rnd_kz = rnd_u
        rnd_v = self.vspread * np.log(self.rng.random(size))
        return rnd_kx * rnd_v, rnd_ky * rnd_v, rnd_kz * rnd_v

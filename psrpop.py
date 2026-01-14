from typing import Callable
import numpy as np
import numpy.typing as npt
from scipy.interpolate import PchipInterpolator
from scipy.spatial.distance import cdist
from oompy import Constants as c, Units as u, Quantity
from tqdm import tqdm

import matplotlib.pyplot as plt


CONSTANTS = {
    "sun_position": np.array([8.5, 0, 0]),
}


def invert_monotone(xs, ys):
    order = np.argsort(ys)
    y_sorted = ys[order]
    x_sorted = xs[order]

    y_unique, idx = np.unique(y_sorted, return_index=True)
    x_unique = x_sorted[idx]

    inv = PchipInterpolator(y_unique, x_unique, extrapolate=False)
    y_min, y_max = y_unique[0], y_unique[-1]
    return inv, (y_min, y_max)


class Distribution:
    def __init__(self, xmin, xmax, pdf, **kwargs):
        self.xmin = xmin
        self.xmax = xmax
        self.pdf = pdf

        resolution = 1000001
        xs = np.linspace(xmin, xmax, resolution)
        cdf_ys = np.cumsum(self.pdf(xs, **kwargs)) * (xmax - xmin) / resolution
        self._invcdf_interpolated, (self._invcdf_min, self._invcdf_max) = (
            invert_monotone(xs, cdf_ys)
        )

    def sample(self, size=1):
        return self._invcdf_interpolated(
            np.random.random(size=size) * (self._invcdf_max - self._invcdf_min)
            + self._invcdf_min
        )


class Pulsars:
    positions: npt.NDArray[np.float64] = np.array([])  # shape: (n, 3), units: kpc
    incl_angles: npt.NDArray[np.float64] = np.array([])  # shape: (n,)
    spin_axes: npt.NDArray[np.float64] = np.array([])  # shape: (n, 3)
    P0s: npt.NDArray[np.float64] = np.array([])  # shape: (n,), units: s
    Bsurf0s: npt.NDArray[np.float64] = np.array([])  # shape: (n,), units: G
    ages: npt.NDArray[np.float64] = np.array([])  # shape: (n,), units: yr
    nbraking: float = 3.0

    def __init__(self, nbraking: float = 3.0):
        self.nbraking: float = nbraking

        self.M: Quantity = Quantity(2, "Msun")
        self.R: Quantity = Quantity(11, "km")
        self.I: Quantity = (2 / 5) * self.M * self.R**2

    @staticmethod
    def from_arrays(
        positions: npt.NDArray[np.float64],
        incl_angles: npt.NDArray[np.float64],
        spin_axes: npt.NDArray[np.float64],
        P0s: npt.NDArray[np.float64],
        Bsurf0s: npt.NDArray[np.float64],
        ages: npt.NDArray[np.float64],
        nbraking: float,
    ) -> "Pulsars":
        new_pulsars = Pulsars(nbraking=nbraking)
        new_pulsars.positions = positions
        new_pulsars.incl_angles = incl_angles
        new_pulsars.spin_axes = spin_axes
        new_pulsars.P0s = P0s
        new_pulsars.Bsurf0s = Bsurf0s
        new_pulsars.ages = ages
        return new_pulsars

    def add(
        self,
        new_positions: npt.NDArray[np.float64],
        new_incl_angles: npt.NDArray[np.float64],
        new_spin_axes: npt.NDArray[np.float64],
        new_P0s: npt.NDArray[np.float64],
        new_Bsurf0s: npt.NDArray[np.float64],
    ):
        self.positions = (
            np.concatenate((self.positions, new_positions), axis=0)
            if len(self.positions) > 0
            else new_positions
        )
        self.incl_angles = np.concatenate((self.incl_angles, new_incl_angles))
        self.spin_axes = (
            np.concatenate((self.spin_axes, new_spin_axes), axis=0)
            if len(self.spin_axes) > 0
            else new_spin_axes
        )
        self.P0s = np.concatenate((self.P0s, new_P0s))
        self.Bsurf0s = np.concatenate((self.Bsurf0s, new_Bsurf0s))

        self.ages = np.concatenate((self.ages, np.zeros(len(new_positions))))

        assert (
            len(self.positions)
            == len(self.incl_angles)
            == len(self.spin_axes)
            == len(self.P0s)
            == len(self.Bsurf0s)
            == len(self.ages)
        ), "Inconsistent array lengths"

    @property
    def number(self) -> int:
        return int(len(self.positions))

    @property
    def distances(self) -> npt.NDArray[np.float64]:
        """Returns the distances of all pulsars from the Sun in kpc."""
        return np.linalg.norm(self.positions - CONSTANTS["sun_position"], axis=1)

    @property
    def dirs_towards(self) -> npt.NDArray[np.float64]:
        """Returns the unit vectors pointing from the Sun to each pulsar."""
        ovec = self.positions - CONSTANTS["sun_position"]
        norms = np.linalg.norm(ovec, axis=1)[:, np.newaxis]
        return ovec / norms

    @property
    def tau0s(self) -> npt.NDArray[np.float64]:
        """Returns the characteristic agespans for all pulsars in years."""
        k = 2 * (2 * c.pi) ** 2 / (3 * c.c**3) * self.R**6 / self.I
        k_val = (k >> "G^-2 sec").value
        sec_to_yr = (u.sec >> "yr").value
        return (
            self.P0s**2
            / ((self.nbraking - 1) * k_val * (self.Bsurf0s) ** 2)
            * sec_to_yr
        )

    @property
    def Pdot0s(self) -> npt.NDArray[np.float64]:
        """Returns the initial period derivatives at birth for all pulsars."""
        yr_to_sec = (u.yr >> "sec").value
        return self.P0s / ((self.nbraking - 1) * self.tau0s * yr_to_sec)

    @property
    def Ps(self) -> npt.NDArray[np.float64]:
        """Returns the current periods for all pulsars."""
        power = 1 / (self.nbraking - 1)
        return self.P0s * (1 + self.ages / self.tau0s) ** power

    @property
    def Pdots(self) -> npt.NDArray[np.float64]:
        """Returns the current period derivatives for all pulsars."""
        power = 1 / (self.nbraking - 1) - 1
        return self.Pdot0s * (1 + self.ages / self.tau0s) ** power

    @property
    def Bsurfs(self) -> npt.NDArray[np.float64]:
        """Returns the current surface magnetic fields for all pulsars."""
        power = -(self.nbraking - 3) / (2 * (self.nbraking - 1))
        return self.Bsurf0s * (1 + self.ages / self.tau0s) ** power

    @property
    def Edots(self) -> npt.NDArray[np.float64]:
        """Returns the current spin-down luminosities for all pulsars."""
        return ((4 * c.pi**2 * self.I) >> "erg sec^2").value * self.Pdots / self.Ps**3

    # PASS DISTRIBUTION FUNCTION FOR R and Z, THEN USE THAT TO SAMPLE
    def new(self, number: int, params: dict):
        number = int(number)
        # R_sun = CONSTANTS["sun_position"][0]

        radial_dist = params["init_model"]["radial_distribution"]
        z_dist = params["init_model"]["z_distribution"]

        # B = params["init_model"]["position"]["B"]
        # C = params["init_model"]["position"]["C"]
        # E = params["init_model"]["position"]["E"]

        rng = np.random.default_rng()

        z_kpc_rnd = z_dist.sample(size=number)
        r_kpc_rnd = radial_dist.sample(size=number)
        # rng.gamma(shape=B + 1.0, scale=R_sun / C, size=number)
        phi_rnd = 2 * np.pi * rng.random(number)

        x_kpc_rnd = r_kpc_rnd * np.cos(phi_rnd)
        y_kpc_rnd = r_kpc_rnd * np.sin(phi_rnd)

        # random orientation for the spin-axis
        costh_rnd = -1 + 2 * rng.random(number)
        phi_rnd = 2 * np.pi * rng.random(number)
        sinth_rnd = np.sqrt(1 - costh_rnd**2)

        # random incl_angles
        incl_angle_rnd = rng.random(number) * np.pi

        spin_periods_sec = []

        while len(spin_periods_sec) < number:
            one = np.random.normal(
                loc=params["init_model"]["P"]["mean"],
                scale=params["init_model"]["P"]["std"],
                size=number,
            )
            two = one[one > params["init_model"]["P"]["threshold"]]
            spin_periods_sec.extend(two.tolist())

        spin_periods_sec = np.array(spin_periods_sec[:number])

        log_bfield_G = rng.normal(
            loc=params["init_model"]["Bsurf"]["mean_log10"],
            scale=params["init_model"]["Bsurf"]["std_log10"],
            size=number,
        )
        bfield_G = 10**log_bfield_G

        self.add(
            new_positions=np.vstack((x_kpc_rnd, y_kpc_rnd, z_kpc_rnd)).T,
            new_incl_angles=incl_angle_rnd,
            new_spin_axes=np.vstack(
                (sinth_rnd * np.cos(phi_rnd), sinth_rnd * np.sin(phi_rnd), costh_rnd)
            ).T,
            new_P0s=spin_periods_sec,
            new_Bsurf0s=bfield_G,
        )

    def evolve_by(self, delta_age: float):
        """Evolves all pulsars by delta_age in years."""
        self.ages += delta_age

    def mask(self, mask_array: npt.NDArray[np.bool_]):
        """Applies a boolean mask to all pulsar attributes."""
        self.positions = self.positions[mask_array]
        self.incl_angles = self.incl_angles[mask_array]
        self.spin_axes = self.spin_axes[mask_array]
        self.P0s = self.P0s[mask_array]
        self.Bsurf0s = self.Bsurf0s[mask_array]
        self.ages = self.ages[mask_array]

    def select(
        self,
        selection_criteria: Callable[["Pulsars", dict], npt.NDArray[np.bool_]],
        params: dict,
    ) -> "Pulsars":
        """Selects pulsars based on the selection_criteria."""
        select_array = selection_criteria(self, params)
        return Pulsars.from_arrays(
            positions=self.positions[select_array],
            incl_angles=self.incl_angles[select_array],
            spin_axes=self.spin_axes[select_array],
            P0s=self.P0s[select_array],
            Bsurf0s=self.Bsurf0s[select_array],
            ages=self.ages[select_array],
            nbraking=self.nbraking,
        )


def Detectable_Geometric(pulsars: Pulsars, params: dict) -> npt.NDArray[np.bool_]:
    """Returns a boolean array indicating which pulsars are detectable
    based on geometric considerations."""

    xis = np.arccos(np.sum(pulsars.spin_axes * pulsars.dirs_towards, axis=1))

    h_dimless = ((params["detectability"]["h_em"] / c.c) >> "sec").value

    rhos = 3 * np.sqrt(np.pi * h_dimless / (2 * pulsars.Ps))

    return (
        (
            (np.abs(xis - pulsars.incl_angles) <= rhos)
            | (np.abs(xis - (np.pi - pulsars.incl_angles)) <= rhos)
        )
        & (pulsars.incl_angles >= rhos)
        & (pulsars.incl_angles <= np.pi - rhos)
    )


def Detectable_Radioflux(pulsars: Pulsars, params: dict) -> npt.NDArray[np.bool_]:
    """Returns a boolean array indicating which pulsars are detectable
    based on radio flux considerations."""

    rng = np.random.default_rng()

    xis = np.arccos(np.sum(pulsars.spin_axes * pulsars.dirs_towards, axis=1))

    h_dimless = ((params["detectability"]["h_em"] / c.c) >> "sec").value

    rhos = 3 * np.sqrt(np.pi * h_dimless / (2 * pulsars.Ps))

    erg_s_to_W = (u.erg / u.sec >> "W").value

    Fr_mJy = (
        9
        * pulsars.distances**-2
        * (pulsars.Edots * erg_s_to_W / 1e29) ** 0.25
        * 10 ** rng.normal(loc=0.0, scale=0.2, size=pulsars.number)
    )

    delta_f_ch_Hz = 3e3 * 1e3
    f_Hz = 1.374 * 1e9
    DM_pc_cm3 = pulsars.distances * 0.017 * 1e3
    tau_DM_sec = 8.3e15 * delta_f_ch_Hz * DM_pc_cm3 / f_Hz**3
    tau_scat_sec = 3.6e-9 * DM_pc_cm3**2.2 * (1 + 1.94e-3 * DM_pc_cm3**2)
    tau_samp_sec = 250e-6

    alphas = np.where(
        np.abs(xis - pulsars.incl_angles) <= rhos,
        pulsars.incl_angles,
        np.pi - pulsars.incl_angles,
    )
    w_r = 2 * np.arccos(
        (np.cos(rhos) - np.cos(alphas) * np.cos(xis)) / (np.sin(alphas) * np.sin(xis))
    )
    tilda_w_r = (
        (
            (w_r * pulsars.Ps / (2 * np.pi)) ** 2
            + tau_samp_sec**2
            + tau_DM_sec**2
            + tau_scat_sec**2
        )
    ) ** 0.5
    tilda_w_r = np.where(tilda_w_r > pulsars.Ps, pulsars.Ps * 0.99, tilda_w_r)

    S_min_survey_mJy = 0.05 * np.sqrt(1 / ((pulsars.Ps / tilda_w_r) - 1))

    return Fr_mJy / S_min_survey_mJy >= 10


def Above_Deathline(pulsars: Pulsars, beta: float = 0.05) -> npt.NDArray[np.bool_]:
    """Returns a boolean array indicating which pulsars are above the death line."""
    return pulsars.Pdots >= (1e-15 * beta * pulsars.Ps ** (11 / 4))


def Simulate_Evolution(
    params: dict,
    birth_rate: float,
    total_time: float,
    time_step: float | None = None,
    nbraking: float = 3.0,
    beta: float | None = 0.05,
) -> Pulsars:
    """
    Simulates the evolution of the pulsar population.

    Parameters:
    - birth_rate: number of pulsars born per year
    - total_time: total simulation time in years
    - time_step: time step for each evolution step in years; if None, generates ages from a uniform distribution
    - nbraking: braking index for pulsar spin-down
    - beta: position of the deathline; if None, deathline is not applied

    Returns:
    Resulting Pulsars object
    """
    pulsars = Pulsars(nbraking)
    if time_step is None:
        pulsars.new(int(birth_rate * total_time), params)
        rng = np.random.default_rng()
        pulsars.ages = rng.uniform(0, total_time, size=pulsars.number)
        if beta is not None:
            pulsars.mask(Above_Deathline(pulsars, beta))
    else:
        for _ in tqdm(np.arange(int(total_time / time_step))):
            # Evolve existing pulsars
            pulsars.evolve_by(time_step)
            # Determine number of new pulsars to add
            num_new_pulsars = np.random.poisson(birth_rate * time_step)
            if num_new_pulsars > 0:
                pulsars.new(num_new_pulsars, params)
            # Apply deathline mask if beta is provided
            if beta is not None:
                pulsars.mask(Above_Deathline(pulsars, beta))
    return pulsars


def Plot_PPdot(
    pulsars: Pulsars,
    ax=None,
    cmap: str = "viridis",
    color: str | None = "ages",
    beta: float | None = 0.05,
    **kwargs,
):
    if ax is None:
        ax = plt.gca()
    if color is not None:
        colors = getattr(pulsars, color)

        if np.min(colors) != np.max(colors):
            if np.min(colors) > 0 and np.max(colors) / np.min(colors) > 100:
                norm_colors = np.log10(colors / np.min(colors)) / np.log10(
                    np.max(colors) / np.min(colors)
                )
            else:
                norm_colors = (colors - np.min(colors)) / (
                    np.max(colors) - np.min(colors)
                )
        elif colors.max() > 0:
            norm_colors = colors / colors.max()
        else:
            norm_colors = colors

        ax.scatter(
            pulsars.Ps,
            pulsars.Pdots,
            color=plt.get_cmap(cmap)(norm_colors),
            **kwargs,
        )
    else:
        ax.scatter(pulsars.Ps, pulsars.Pdots, **kwargs)

    # death line
    x_death_line = np.logspace(-3, 1, 500)  # 1ms to 10s
    y_death_line = 1e-15 * beta * x_death_line ** (11 / 4)
    ax.plot(
        x_death_line,
        y_death_line,
        "r--",
        linewidth=1,
        label=rf"Death Line ($\beta={{{beta}}}$)",
    )

    ax.set(
        xlim=(1e-3, 10),
        ylim=(1e-21, 1e-9),
        xscale="log",
        yscale="log",
        xlabel=r"$P$ [s]",
        ylabel=r"$\dot{P}$",
    )

    ax.set_xlim(1e-3, 10)
    ax.set_ylim(1e-21, 1e-9)
    ax.set_xscale("log")
    ax.set_yscale("log")

    return ax


def Read_Catalogue(dbfile="data.txt") -> dict[str, npt.NDArray[np.str_]]:
    header = [
        "#",
        "NAME",
        "",
        "P0",
        "",
        "",
        "P1",
        "",
        "",
        "W50",
        "",
        "",
        "W10",
        "",
        "",
        "S400",
        "",
        "",
        "S1400",
        "",
        "",
        "S2000",
        "",
        "",
        "DIST",
        "DIST_DM",
        "",
        "ZZ",
        "XX",
        "YY",
    ]

    data = np.loadtxt(dbfile, dtype="str")

    return {k: data[:, i] for i, k in enumerate(header) if k != ""}


def SampleDistance(A, B):
    means_A = A.mean(axis=0)
    vars_A = A.var(axis=0) / 2

    A = (A - means_A) / vars_A
    B = (B - means_A) / vars_A

    nA = len(A)
    nB = len(B)

    diffAB = cdist(A, B).sum()
    diffAA = cdist(A, A).sum()
    diffBB = cdist(B, B).sum()

    return (
        (nA + nB)
        / (nA * nB)
        * (2 * diffAB / (nA * nB) - diffAA / nA**2 - diffBB / nB**2)
    )


def IsNormalPulsar(Ps, Pdots):
    # criterion ms-pulsars
    P1_lower, Pdot1_lower = 0.01, 1e-10  # (P, Pdot)
    P2_lower, Pdot2_lower = 1.0, 1e-19  # (P, Pdot)

    logP1_lower, logPdot1_lower = np.log10(P1_lower), np.log10(Pdot1_lower)
    logP2_lower, logPdot2_lower = np.log10(P2_lower), np.log10(Pdot2_lower)

    slope_lower = (logPdot2_lower - logPdot1_lower) / (logP2_lower - logP1_lower)
    intercept_lower = logPdot1_lower - slope_lower * logP1_lower

    # criterion magnetars
    P1_upper, Pdot1_upper = 0.1, 1e-10
    P2_upper, Pdot2_upper = 10.0, 1e-14

    logP1_upper, logPdot1_upper = np.log10(P1_upper), np.log10(Pdot1_upper)
    logP2_upper, logPdot2_upper = np.log10(P2_upper), np.log10(Pdot2_upper)

    slope_upper = (logPdot2_upper - logPdot1_upper) / (logP2_upper - logP1_upper)
    intercept_upper = logPdot1_upper - slope_upper * logP1_upper

    return (Pdots < 10 ** (slope_upper * np.log10(Ps) + intercept_upper)) & (
        Pdots > 10 ** (slope_lower * np.log10(Ps) + intercept_lower)
    )

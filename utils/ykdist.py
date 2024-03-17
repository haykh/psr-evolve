import numpy as np
from scipy import stats, special

ArrayOrFloat = np.ndarray | float

defaults = {
    "a_0": 1.64,
    "b_0": 4.01,
    "R1_0": 0.55,
    "Rsun_0": 8.5,
}


def YKNorm(**kwargs) -> float:
    a = kwargs.get("a_0")
    b = kwargs.get("b_0")
    R1 = kwargs.get("R1_0")
    Rsun = kwargs.get("Rsun_0")
    X1 = R1 + Rsun
    norm = (
        b ** (-a - 2)
        * np.exp(b)
        * X1**2
        * special.gammaincc(a + 2, b * R1 / X1)
        * special.gamma(a + 2)
    )
    return norm


def YKPDF(r: float, **kwargs) -> ArrayOrFloat:
    a = kwargs.get("a_0")
    b = kwargs.get("b_0")
    R1 = kwargs.get("R1_0")
    Rsun = kwargs.get("Rsun_0")
    X1 = R1 + Rsun
    return (
        (r + R1)
        * ((r + R1) / X1) ** a
        * np.exp(-b * (r - Rsun) / X1)
        / kwargs.get("norm")
    )


def YKCDF(r: float, **kwargs) -> ArrayOrFloat:
    a = kwargs.get("a_0")
    b = kwargs.get("b_0")
    R1 = kwargs.get("R1_0")
    Rsun = kwargs.get("Rsun_0")
    X1 = R1 + Rsun
    return (
        b ** (-a - 2)
        * np.exp(b)
        * X1**2
        * (
            kwargs.get("gam1")
            - special.gammaincc(a + 2, b * (R1 + r) / X1) * kwargs.get("gam2")
        )
        / kwargs.get("norm")
    )


class YKdist_gen(stats.rv_continuous):
    """
    Generates a random radial distance from the Yusifov & Küçük distribution.

    Parameters
    ----------
    a_0 : float, optional
        Shape parameter, by default 1.64
    b_0 : float, optional
        Scale parameter, by default 4.01
    R1_0 : float, optional
        Inner radius, by default 0.55 [kpc]
    Rsun_0 : float, optional
        Distance of the Sun from GC, by default 8.5 [kpc]
        
    Methods
    -------
    pdf(r)
        Probability density function.
    
    cdf(r)
        Cumulative distribution function.
        
    rvs(size)
        Random variates.
    """

    def __init__(
        self,
        a_0: float = defaults["a_0"],
        b_0: float = defaults["b_0"],
        R1_0: float = defaults["R1_0"],
        Rsun_0: float = defaults["Rsun_0"],
        **kwargs
    ):
        self.kwargs = {
            "a_0": a_0,
            "b_0": b_0,
            "R1_0": R1_0,
            "Rsun_0": Rsun_0,
        }
        self.kwargs["norm"] = YKNorm(**self.kwargs)
        self.kwargs["gam1"] = special.gammaincc(
            a_0 + 2, b_0 * R1_0 / (R1_0 + Rsun_0)
        ) * special.gamma(a_0 + 2)
        self.kwargs["gam2"] = special.gamma(a_0 + 2)
        stats.rv_continuous.__init__(self, **kwargs)

    def _pdf(self, r: ArrayOrFloat):
        return np.where(r > 0, YKPDF(r, **self.kwargs), 0.0)

    def _cdf(self, r: ArrayOrFloat):
        return np.where(r > 0, YKCDF(r, **self.kwargs), 0.0)


YKdist = YKdist_gen(
    a=0,
    b=np.inf,
    name="YKdist",
)

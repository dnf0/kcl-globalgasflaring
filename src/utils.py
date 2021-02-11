import numpy as np

def planck_radiance(wvl, temp):
    """

    :param wvl:
    :param temp:
    :return:
    """
    c1 = 1.19e-16  # W m-2 sr-1
    c2 = 1.44e-2  # mK
    wt = (wvl * 1.e-6) * temp  # m K
    d = (wvl * 1.e-6) ** 5 * (np.exp(c2 / wt) - 1)
    return c1 / d * 1.e-6  # W m-2 sr-1 um-1

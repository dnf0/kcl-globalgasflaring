
"""
Computes ratio over two different spectral channels
to see how they vary as a function of emitter temperature.
They are monotonically increasing functions, so can use the
ratio to identify which ratios are gas falres.
"""

import numpy as np
import matplotlib.pyplot as plt

import src.utils as utils

if __name__ == "__main__":

    flare_size = 10000  # in sq. m.
    pixel_size = 1e6  # in sq. m.
    flare_area_pc = flare_size / pixel_size  # unitless

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 5))

    temps = np.arange(0,4000, 1)  # in K

    spect_rad_swir = flare_area_pc * utils.planck_radiance(1.6, temps)
    spect_rad_swir_2 = flare_area_pc * utils.planck_radiance(2.2, temps)
    spect_rad_mwir = flare_area_pc * utils.planck_radiance(3.7, temps)
    spect_rad_lwir = flare_area_pc * utils.planck_radiance(11, temps)

    ratio_mwir = spect_rad_swir / spect_rad_mwir
    ratio_swir = spect_rad_swir / spect_rad_swir_2

    print 'MWIR ratio 1400K', ratio_mwir[1400]
    print 'MWIR ratio 3500K', ratio_mwir[3500]
    print 'SWIR ratio 1400K', ratio_swir[1400]
    print 'SWIR ratio 2500K', ratio_swir[2500]

    ax.plot(temps, ratio_mwir,  "k-", markersize=4)
    ax.set_xlabel("Temperature (K)", fontsize=16)
    ax.set_ylabel("$L_{1.6} / L_{3.7}$", fontsize=16)
    plt.show()

__author__ = 'daniel'


"""
Computes ratio over two different spectral channels
to see how they vary as a function of emitter temperature.
They are monotonically increasing functions, so can use the
ratio to identify which ratios are gas falres.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize



def planck_radiance(wvl, temp):
    '''
    wvl: wavelngth (microns)
    temp: temperature (kelvin)
    '''
    c1 = 1.19e-16  # W m-2 sr-1
    c2 = 1.44e-2  # mK
    wt = (wvl*1.e-6) * temp # m K
    d = (wvl*1.e-6)**5 * (np.exp(c2/wt)-1)
    return c1 / d * 1.e-6  # W m-2 sr-1 um-1


if __name__ == "__main__":


    flare_size = 10000  # in sq. m.
    pixel_size = 1e6  # in sq. m.
    flare_area_pc = flare_size / pixel_size  # unitless

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 5))

    temps = np.arange(0,3000, 1)  # in K

    spect_rad_swir = flare_area_pc * planck_radiance(1.6, temps)
    spect_rad_mwir = flare_area_pc * planck_radiance(3.7, temps)
    spect_rad_lwir = flare_area_pc * planck_radiance(11, temps)


    ratio_mwir = spect_rad_swir / spect_rad_mwir
    ratio_lwir = spect_rad_swir / spect_rad_lwir

    print 'MWIR ratio 1500K', ratio_mwir[1500]
    print 'MWIR ratio 2300K', ratio_mwir[2300]
    print 'MWIR ratio 1500K', ratio_lwir[1500]
    print 'MWIR ratio 2300K', ratio_lwir[2300]

    ax.plot(temps, ratio_mwir,  "k-", markersize=4)

    ax.set_xlabel("Temperature (K)", fontsize=16)
    ax.set_ylabel("$L_{1.6} / L_{4}$", fontsize=16)
    plt.show()

    #fp ='/Users/dnf/Dropbox/working_documents/papers/TGRS-swir-frp-notshared/figures/Figure_2.png'
    #plt.savefig(fp, bbox_inches='tight')
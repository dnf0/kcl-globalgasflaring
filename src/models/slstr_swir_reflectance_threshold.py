
import logging
import numpy as np
import matplotlib.pyplot as plt


def planck_func(l, t):
    '''
    l: wavelngth
    t: temperature
    '''

    c1 = 1.19e-16  # W m-2 sr-1
    c2 = 1.44e-2  # mK
    lt = l*t  # meter K

    denominator = l**5 * (np.exp(c2/lt)-1)
    return c1 / denominator  # W m-2 sr-1 m-1


def radiance_to_reflectance():
    pass


def main():

    l = 1.6e-6  # wavelegnth in m

    # pixel_area = 1.0**2 * 1000000  # m^2
    # areas = np.logspace(0, 6, 1000)  # m^2

    pixel_area = 0.5**2 * 1000000  # m^2
    areas = np.logspace(0, np.log10(pixel_area), 1000)  # m^2

    temps = np.arange(400, 1600, 10)  # Kelvin

    temps, areas = np.meshgrid(temps, areas)

    # dvidie the flare areas by the pixel areas
    fractional_areas = areas / pixel_area

    # now get the radiance by plugging the temperatures in to the
    # planck function and multiplying by the areass
    radiances = fractional_areas * planck_func(l, temps) / 1000000

    # now convert to reflectances
    reflectances = radiances / 254.23103333 * np.pi * 100

    valid = (reflectances <= 0.105) & (reflectances > 0.0995)

    plt.pcolormesh(fractional_areas, temps, reflectances, vmax = 0.1)
    plt.plot(fractional_areas[valid], temps[valid], 'k-', label="0.1")
    plt.plot([0.0004, 0.0004], [400, 1590], 'k--')
    cbar = plt.colorbar()
    cbar.set_label('1.6 $\mu m$ Reflectance')
    plt.ylabel("Flare Temperature ($K$)")
    plt.xscale('log')
    #plt.ylim((0, 3))
    #plt.xlim(1600, 2000)
    plt.xlabel("Fractional Flare Area ($m^2$)")
    plt.legend(loc=1)

    plt.show()

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)
    main()
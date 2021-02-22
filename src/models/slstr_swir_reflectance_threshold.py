
import logging
import numpy as np
import matplotlib.pyplot as plt


def planck_func(target_wavelength, target_temperature):
    '''
    l: wavelngth
    t: temperature
    '''

    c1 = 1.19e-16  # W m-2 sr-1
    c2 = 1.44e-2  # mK
    lt = target_wavelength * target_temperature  # meter K

    denominator = target_wavelength**5 * (np.exp(c2/lt)-1)
    return c1 / denominator  # W m-2 sr-1 m-1


def main():

    target_wvl = 1.6e-6  # wavelegnth in m

    # pixel_area = 1.0**2 * 1000000  # m^2
    # areas = np.logspace(0, np.log10(pixel_area), 1000)  # m^2

    pixel_area = 0.5**2 * 1000000  # m^2
    areas = np.logspace(0, np.log10(pixel_area), 1000)  # m^2

    temps = np.arange(400, 1600, 10)  # Kelvin

    temps, areas = np.meshgrid(temps, areas)

    # divide the flare areas by the pixel areas
    fractional_areas = areas / pixel_area

    # now get the radiance by plugging the temperatures in to the
    # planck function and multiplying by the areass
    radiances = fractional_areas * planck_func(target_wvl, temps) / 1000000

    # now convert to reflectances
    reflectances = radiances / 254.23103333 * np.pi * 100

    valid = (reflectances <= 0.405) & (reflectances > 0.395)

    plt.pcolormesh(fractional_areas, temps, reflectances, vmax=0.4)
    plt.plot(fractional_areas[valid], temps[valid], 'k-', label="0.4")
    plt.plot([0.0004, 0.0004], [400, 1590], 'k--')
    plt.plot([fractional_areas.min(), fractional_areas.max()], [944, 944], 'r--')
    cbar = plt.colorbar()
    cbar.set_label('1.6 Micron Reflectance')
    plt.ylabel("Flare Temperature ($K$)")
    plt.xscale('log')
    # plt.ylim((0, 3))
    # plt.xlim(1600, 2000)
    plt.xlabel("Fractional Flare Area ($m^2$)")
    plt.legend(loc=1)

    plt.show()


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)
    main()

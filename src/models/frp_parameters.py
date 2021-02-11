import os
import logging

import numpy as np
import scipy.constants as const

import src.config.filepaths as fp
import src.utils as utils


def spectral_responses(srf, temperatures):
    """

    :param srf: sensor spectral response function
    :param temperatures: range of temperatures over which to evaluate
    :return: spectral raidiances weighted by the response function over a temperature range
    """

    # setup srf
    sensor_spectra = srf[:, 0]
    sensor_responses = srf[:, 1]

    srf_weighted_spectral_radiances = np.zeros(temperatures.size)

    for i, T in enumerate(temperatures):  # K

        # compute the spectral radiance at each wavelength
        # defined in the instrument spectral reponse
        # function.
        spectral_radiances_at_sensor_spectra = [utils.planck_radiance(ss, T) for ss in sensor_spectra]

        # now integrate the srf weighted spectral
        # radiances over all wavelengths to get the
        # srf weighted radiance for the wavelength
        # range
        srf_weighted_radiance = np.trapz(spectral_radiances_at_sensor_spectra * sensor_responses, sensor_spectra)

        # the above weighted radiances are in W m-2 sr-1
        # and we need to get it back to spectral radiances W m-2 sr-1 m-1.
        # To do this we divide by the integral of the
        # srf over all wavelengths, which has units m-1
        # in this instance.
        srf_weighted_spectral_radiance = srf_weighted_radiance / np.trapz(sensor_responses, sensor_spectra)

        srf_weighted_spectral_radiances[i] = srf_weighted_spectral_radiance

    return srf_weighted_spectral_radiances


def opt_temp(temps, frp_assumed, frp_true):
    """
    Function to calculate optimal temperature to use in deriving
    single channel FRP coefficient

    :param temps: range of temperatures to evaluate
    :param frp_assumed: range of assumed FRP values
    :param frp_true: true FRP values associated with the above assumed FRPs
    :return: The temperature that provides the optimal fit
    """

    min_temp = 1600
    max_temp = 2150

    temp_mask = (temps > min_temp) & (temps < max_temp)
    frp_abs_bias = np.abs((frp_assumed - frp_true) / frp_true)
    frp_abs_bias[~temp_mask, ...] = np.nan

    nan_max = np.nanmax(frp_abs_bias, axis=0)
    arg_min = np.argmin(nan_max)

    return arg_min, temps[arg_min]


def main():
    # experiment set up
    temp_min = 500
    temp_max = 3000
    temps = np.arange(temp_min, temp_max, 1)  # in K
    estimate_temps, actual_temps = np.meshgrid(temps, temps)
    flare_size = 100  # in sq. m.
    pixel_size = 1e6  # in sq. m.
    flare_area_pc = flare_size / pixel_size  # unitless
    frp_true = pixel_size * flare_area_pc * (const.sigma * actual_temps ** 4)

    sensors = ['ATSR2', 'ATSR1', 'AATSR', 'SLSTR']
    for sensor in sensors:
        sensor_fname = sensor + "_16.txt"
        sensor_srf = np.loadtxt(os.path.join(fp.path_to_srfs, sensor_fname))

        # generate L_true
        l_true = spectral_responses(sensor_srf, temps)

        # now generate the fpr coefficients (effectively spectral radiance / temperature as  aT^4 ~ B(T))
        a = l_true / temps ** 4

        # now apply a to the estimated temperatures using take, in effect we are
        # using the temperatures as indexes (hence we take off the min temp).  For example
        # in the case of 500K to 2000K there are 1500 temperature samples, and 'a' has 1500
        # samples.  We can therefore populate estimated temps with 'a' using take, and subtracting
        # the min temp off of estimate temps (effectively making them an index from 0-1499).
        temp_ind = estimate_temps - temp_min
        a = np.take(a, temp_ind)

        # same thing for spectral radiances
        temp_ind = actual_temps - temp_min
        l_true = np.take(l_true, temp_ind)

        # now calculate the assumed frp using a and l_true
        frp_assumed = ((pixel_size * const.sigma / a) * l_true)  # in MW

        # compute the optimum temperature
        optimised_index, temp = opt_temp(temps, frp_assumed, frp_true)

        opt_param = a[0, optimised_index]

        print('Optimised frp parameter for', sensor, ':', str(opt_param), 'at temp of:', str(temp))


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()

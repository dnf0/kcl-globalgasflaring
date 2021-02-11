import logging
import os

import numpy as np
from scipy import interpolate

import src.config.filepaths as fp


def crop(sensor_srf, sol_rad):
    sen_wvl = sensor_srf[:, 0] * 1000
    sol_wvl = sol_rad[:,0]
    sol_rad = sol_rad[:,1]
    sol_mask = (sol_wvl > sen_wvl.min()) & (sol_wvl < sen_wvl.max())
    return sol_rad[sol_mask], sol_wvl[sol_mask]


def interp_srf(sensor_srf, sol_wvl):
    sen_wvl = sensor_srf[:, 0] * 1000  # convert from microns to nm
    sen_res = sensor_srf[:, 1]
    f = interpolate.interp1d(sen_wvl, sen_res)
    return f(sol_wvl)


def inband_rad(solar_rad, res):
    numerator = np.sum(solar_rad * res * 5)
    denominator = np.sum(res * 5)
    return numerator / denominator * 1000


def main():

    # read in solar irradiance
    sol_rad = np.loadtxt(fp.path_to_solar_data)

    sensors = ['ATSR2', 'ATSR1', 'AATSR']
    for sensor in sensors:
        sensor_fname = sensor + "_16.txt"
        sensor_srf = np.loadtxt(os.path.join(fp.path_to_srfs, sensor_fname))

        # crop solar radiances so that the solar wvl are within the sensor wvl range
        cropped_sol_rad, cropped_sol_wvl = crop(sensor_srf, sol_rad)

        # interpolate the spectral reponses onto the solar wavelengths
        interp_spectral_responses = interp_srf(sensor_srf, cropped_sol_wvl)

        # compute in band solar irradiance
        inband_sol_rad = inband_rad(cropped_sol_rad, interp_spectral_responses)
        print(sensor, 'in band solar irradiance:', inband_sol_rad)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
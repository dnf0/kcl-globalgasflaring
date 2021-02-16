import numpy as np
import zipfile
from netCDF4 import Dataset
import os
import shutil
import epr


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


def extract_zip(input_zip, path_to_temp):
    data_dict = {}
    to_extract = ["S5_radiance_an.nc", "S6_radiance_an.nc",
                  "geodetic_an.nc", "geometry_tn.nc",
                  "cartesian_an.nc", "cartesian_tx.nc",
                  "indices_an.nc", "flags_an.nc"]
    with zipfile.ZipFile(input_zip) as input:
        for name in input.namelist():
            split_name = name.split('/')[-1]
            if split_name in to_extract:
                var_name = split_name.split('.')[0]
                source = Dataset(input.extract(name, path_to_temp))
                data_dict[var_name] = source

    # remove the unzip files
    dir_to_remove = os.path.join(path_to_temp, input_zip.split('/')[-1].replace('zip', 'SEN3'))
    if os.path.isdir(dir_to_remove):  # test if the path points to a directory
        shutil.rmtree(dir_to_remove, ignore_errors=True)
    else:  # normal file
        os.remove(dir_to_remove)

    return data_dict


def read_atsr(path_to_ats_data):
    return epr.Product(path_to_ats_data)


def define_sensor(path_to_data):
    if 'N1' in path_to_data:
        sensor = 'ats'
    if 'E2' in path_to_data:
        sensor = 'at2'
    if 'E1' in path_to_data:
        sensor = 'at1'
    return sensor
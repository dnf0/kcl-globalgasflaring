'''
This file contains some constants used in the processing
'''
import scipy.constants as const

solar_irradiance = {'ats': 254.752,
                    'at2': 249.604,
                    'at1': 250.728}

day_night_angle = 101  # solar zenith angles must be > 101 deg VIIRS night-time lights Elvidge et al., 2017

# flare detection limit.  We do not use the upper threshold of 3% as this removes a number of
# significant flares pixels.  E.g. over Iraq.  TODO make a visualisation to show this!
swir_thresh = 0.1  # from Casadio et al. 2012 paper

frp_coeff = {'ats': const.sigma / 8.29908189231e-09,
             'at2': const.sigma / 8.2215268253e-09,
             'at1': const.sigma / 8.23565040885e-09}

# thresholds use for determining whether a pixel is a flare or not
mwir_max_ratio = 2.04
mwir_min_ratio = 5.97

lwir_max_ratio = 53.18
lwir_min_ratio = 240.10

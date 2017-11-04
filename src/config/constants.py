'''
This file contains some constants used in the processing
'''
import scipy.constants as const

sensor = 'ats'  # 'at2', 'at1'

if sensor == 'ats':
    solar_irradiance = 254.752  # W m-2 sr-1 um-1 
elif sensor == 'at2':
    solar_irradiance = 249.604  # W m-2 sr-1 um-1  
elif sensor == 'at1':
    solar_irradiance = 250.728  # W m-2 sr-1 um-1  

day_night_angle = 101  # solar zenith angles must be > 101 deg VIIRS night-time lights Elvidge et al., 2017

# flare detection limit.  We do not use the upper threshold of 3% as this removes a number of
# significant flares pixels.  E.g. over Iraq.  TODO make a visualisation to show this!
lower_thresh = 0.1  # from Casadio et al. 2012 paper

if sensor == 'ats':
    planck_approx = 8.29908189231e-09
elif sensor == 'at2':
    planck_approx = 8.2215268253e-09
elif sensor == 'at1':
    planck_approx = 8.23565040885e-09

frp_coeff = const.sigma / planck_approx

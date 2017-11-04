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

day_night_angle = -5  # less than this angle is considered night

# flare detection limit
swir_thresh = 0.1  # from Casadio et al. 2012 paper

if sensor == 'ats':
    planck_approx = 8.29908189231e-09
elif sensor == 'at2':
    planck_approx = 8.2215268253e-09
elif sensor == 'at1':
    planck_approx = 8.23565040885e-09

frp_coeff = const.sigma / planck_approx

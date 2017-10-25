import logging

import scipy.interpolate as interpolate
import numpy as np

import matplotlib.pyplot as plt

radius_earth = 6371.0  # km
radius_orbit = 7148.0
sat_alt = radius_orbit - radius_earth
factor = radius_orbit / radius_earth


def create_grid():
    x_size = 512
    return (np.arange(x_size, dtype=np.float64) - (float(x_size - 1)) / 2) / radius_earth


def parameterised_rotation_angle(n):
    return np.arange(n, dtype=np.float64) * 2.0 * np.pi / float(n)


def delta_gamma_alpha(beta, eps):
    # delta: The angular distance which is subtended by the along-track distance as seen from the satellite
    # gamma:  the across-track distance as seen from the satellite
    # alpha:  the nadir angle at the satellite
    delta = beta * (1. - np.cos(eps))  # eq. (11) Denis et al., 2007
    gamma = beta * np.sin(eps)  # eq. (10) Denis et al., 2007
    alpha = np.arccos(np.cos(delta) * np.cos(gamma))  # eq. (12) Denis et al., 2007
    return delta, gamma, alpha


def c_d_a(alpha, gamma):
    # c: distance between the pixel at B and the sub-satellite point
    # d: e the distance d, between the pixel point B and the satellite
    # a: the angle a, which is subtended by the across-track distance

    # some constants for this routine
    radius_earth_2 = radius_earth * radius_earth
    radius_orbit_2 = radius_orbit * radius_orbit
    earth_orbit = radius_earth * radius_orbit

    c = np.arcsin(factor * np.sin(alpha)) - alpha  # eq. (6) Denis et al., 2007
    d = np.sqrt(radius_orbit_2 + radius_earth_2 - 2. * earth_orbit * np.cos(c))  # eq. (14) Denis et al., 2007
    a = np.arcsin(d / radius_earth * np.sin(gamma))  # eq. (13) Denis et al., 2007
    return c, d, a


def angles_mannstien(forward=False):

    beta = 23.45*np.pi/180.0 #0.40976452
    n = 1000

    # This array describes the angular distance
    # between the nadir-track and the pixel point
    # as seen from the earth's centre.
    # It goes from -255.5 to 255.5 * 1km/re.
    # It is equivalent to the angle b.
    grid = create_grid()

    # Define the parameterised rotation angle
    # of the radiometer's off-set mirror
    # It goes from 0 to 2Pi and corresponds
    # then to the full scan (nadir + forward)
    eps = parameterised_rotation_angle(n)

    # Compute the different angles and
    # the distance d for the full scan
    delta, gamma, alpha = delta_gamma_alpha(beta, eps)
    c, d, a = c_d_a(alpha, gamma)

    # nadir view
    min_a = np.where(a == a.min())[0]
    max_a = np.where(a == a.max())[0]

    # select the nadir part in the angle arrays a and eps
    # eps is from 3pi/2 to pi/2
    a_shift = np.roll(a, max_a[0], 0)[0:min_a[0] - max_a[0]]
    eps_shift = np.roll(eps, max_a[0])[0:min_a[0] - max_a[0]]

    # Shift eps values in order
    # to have eps values in [-Pi/2,Pi/2]
    # ind = WHERE(eps_s GT !dpi,nind)
    # IF (nind GT 0) THEN eps_s(ind) = eps_s(ind) - 2.*!dpi
    eps_shift = np.where(np.greater(eps_shift, np.pi), (eps_shift - 2 * np.pi), eps_shift)

    # Interpolate for the nadir scan the
    # rotation angle of the mirror from
    # the a basis (nadir part)
    # to the interpolation grid basis
    # eps1 = INTERPOL(eps_s,a_s,grid)
    eps1_interpolated = interpolate.interp1d(a_shift, eps_shift)(grid)

    # now compute the angles again now with interpolation
    delta, gamma, alpha = delta_gamma_alpha(beta, eps1_interpolated)

    if forward:
        a_shift = a[max_a[0]:min_a[0]+1]
        eps_shift = eps[max_a[0]:min_a[0]+1]

        # need to reverse shifted arrays as scipy interpolates only over x where
        # x is increasing
        eps2_interpolated = interpolate.interp1d(a_shift[::-1], eps_shift[::-1])(grid)
        delta, gamma, alpha = delta_gamma_alpha(beta, eps2_interpolated)

    # for pixel size we need zenith and alpha
    theta = (np.arcsin(np.sin(alpha) * factor))
    return alpha, theta


def get_semi_major_axis(alpha):
    # alpha: scan angle at nadir
    ifov = 0.00154  # rad
    return (radius_earth / 2) * (np.arcsin(factor * np.sin(alpha + ifov/2)) -
                                 np.arcsin(factor * np.sin(alpha - ifov/2)) - ifov)

def get_semi_minor_axis(alpha, theta):
    # alpha: scan angle at nadir
    ifov = 0.00129  # rad
    return (radius_earth / 2) * ifov * np.sin(theta - alpha) / np.sin(alpha)


def compute(forward=False):
    alpha, theta = angles_mannstien(forward)

    semi_major = get_semi_major_axis(alpha)
    semi_minor = get_semi_minor_axis(alpha, theta)

    return np.pi * semi_major * semi_minor  # ellipse area formula


def main():
    sizes = compute()
    print sizes[[1,20,45,366, 511]]

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)
    main()
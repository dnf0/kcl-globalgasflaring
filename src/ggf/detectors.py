from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from scipy.interpolate import RectBivariateSpline
from skimage.filters import rank
from skimage.morphology import square
from datetime import datetime

import src.config.constants as proc_const
import src.config.filepaths as fp
from src.models import atsr_pixel_size


class BaseDetector(ABC):

    def __init__(self,
                 day_night_angle=None,
                 swir_thresh=None,
                 cloud_window_size=None):

        """
        BaseDetector base class that contains all the shared attributes
        and methods used by the child classes that inherit from it.

        Args:
            day_night_angle: Solar zenith angle that defines the day/night boundary
            swir_thresh: Threshold which hotspots must exceed to be detected
            cloud_window_size: Image window over which local cloud statistics are computed
        """

        self.day_night_angle = day_night_angle
        self.swir_thresh = swir_thresh
        self.cloud_window_size = cloud_window_size
        self.sensor = None

        # setup attributes generated
        self.latitude = None
        self.longitude = None
        self.sza = None
        self.night_mask = None
        self.swir_16 = None
        self.swir_22 = None
        self.mwir = None
        self.background_mwir = None
        self.background_mask = None
        self.cloud_free = None
        self.cloudy = None
        self.local_cloudiness = None
        self.potential_hotspots = None
        self.sensor = None
        self.pixel_size = None
        self.frp = None
        self.hotspots = None

    def _make_night_mask(self) -> None:
        """
        Computes the day/night binary mask from
        the solar zenith angle.

        Returns:
            None
        """
        self.night_mask = self.sza >= self.day_night_angle

    def _detect_potential_hotspots(self) -> None:
        """
        Identifies pixels with raised signal in the
        Shortwave Infrared (SWIR) imagery.

        Returns:
            None
        """
        self.potential_hotspots = self.swir_16 > self.swir_thresh

    def _compute_frp(self) -> None:
        """
        Computes the pixel fire radiative power based on the
        Fisher and Wooster SWIR Radiance Method.
        https://doi.org/10.3390/rs10020305

        Returns:
            None
        """
        self.frp = self.pixel_size * proc_const.frp_coeff[self.sensor] * self.swir_16 / 1000000  # in MW

    def _compute_local_cloudiness(self) -> None:
        """
        Computes the local mean cloudiness from binary cloud masks.

        Returns:
            None
        """
        # k = np.ones([self.cloud_window_size, self.cloud_window_size])
        # s = convolve(self.cloudy.astype(int), k, mode='constant', cval=0.0)
        # count = convolve(np.ones(self.cloudy.shape), k, mode='constant', cval=0.0)
        # self.local_cloudiness = s/count
        selem = square(self.cloud_window_size)
        self.local_cloudiness = rank.mean(self.cloudy, selem)

    def _build_dataframe(self, keys, mask, joining_df=None) -> pd.DataFrame:
        """
        A flexible dataframe builder that takes in a set of keys that
        correspond to data contained within the object.  For each item
        of data, the samples associated with hotspot activity are selected
        using the provided mask.  The joining_df keyword allows the dataframe
        to be reduced as is sometimes required in the processing chain.

        Args:
            keys: The variables to be included in the dataframe (columns)
            mask: The samples to be included in the dataframe (rows)
            joining_df: Used in reduced the dataframe through an Inner Join

        Returns:
            Dataframe containing the requested data defined by the input
            keys and mask and, if included, the reducing dataframe.

        """
        df = pd.DataFrame()

        # store data associated with product
        for k in keys:
            if k not in self.__dict__:
                raise KeyError(k + ' not found in available product keys ' + self.__dict__)
            if self.__dict__[k] is None:
                continue
            df[k] = self.__dict__[k][mask]

        # store additional derived data
        lines, samples = np.where(mask)
        df['line'] = lines
        df['sample'] = samples
        df['grid_x'] = self._find_arcmin_gridcell(df['latitude'])
        df['grid_y'] = self._find_arcmin_gridcell(df['longitude'])

        if joining_df is not None:
            df = pd.merge(joining_df, df, on=['grid_x', 'grid_y'])
        return df

    def _find_arcmin_gridcell(self, coordinates):
        """
        Ingests cartesian coordinates and rescales them
        to an integer representation of a 1-arminute grid.
        This is an approximate representation and is used
        solely for aggregation purposes (i.e. cannot be
        used for visualisation).

        Args:
            coordinates: A set of cartesian coordinates

        Returns:
            An integer value corresponding to an arcminute gridcell

        """
        neg_values = coordinates < 0

        abs_x = np.abs(coordinates)
        floor_x = np.floor(abs_x)
        decile = abs_x - floor_x
        minute = np.around(decile * 60)  # round to nearest arcmin
        minute_fraction = minute * 0.01  # convert to fractional value (ranges from 0 to 0.6)

        max_minute = minute_fraction > 0.59

        floor_x[neg_values] *= -1
        floor_x[neg_values] -= minute_fraction[neg_values]
        floor_x[~neg_values] += minute_fraction[~neg_values]

        # deal with edge cases - just round them all up
        if np.sum(max_minute) > 0:
            floor_x[max_minute] = np.around(floor_x[max_minute])

        # rescale
        floor_x = (floor_x * 100).astype(int)

        return floor_x

    @abstractmethod
    def _load_arrays(self) -> None:
        raise NotImplementedError("Must override _load_arrays")

    @abstractmethod
    def run_detector(self) -> None:
        raise NotImplementedError("Must override run_detector")

    @abstractmethod
    def to_dataframe(self) -> pd.DataFrame:
        raise NotImplementedError("Must override to_dataframe")


class ATXDetector(BaseDetector):

    def __init__(self,
                 product,
                 day_night_angle=proc_const.day_night_angle,
                 swir_thresh=proc_const.swir_thresh_ats,
                 cloud_window_size=17,
                 background_window_size=17):
        """
        Detector implementation for the Along Track Scanning Radiomter
        instruments (ATSR-1, ATSR-2, AATSR).
        Args:
            product: An ATSR product
            day_night_angle: Solar zenith angle that defines the day/night boundary
            swir_thresh: Threshold which hotspots must exceed to be detected
            cloud_window_size: Image window over which local cloud statistics are computed
            background_window_size: Image window over which local MWIR background statistics are computed
        """
        super().__init__(day_night_angle, swir_thresh, cloud_window_size)
        self.product = product
        self.sensor = self._define_sensor()
        self.background_window_size = background_window_size

    def _define_sensor(self):
        """
        Identifies satellite type based on product id.
        Returns:
            Sensor code string

        """
        if 'N1' in self.product.id_string:
            return 'ats'
        if 'E2' in self.product.id_string:
            return 'at2'
        if 'E1' in self.product.id_string:
            return 'at1'

    def _load_arrays(self) -> None:
        """
        Loads the product data needed for all processing.
        Returns:
            None
        """
        self.latitude = self.product.get_band('latitude').read_as_array()
        self.longitude = self.product.get_band('latitude').read_as_array()
        self.cloud_free = self.product.get_band('cloud_flags_nadir').read_as_array() <= 1
        self.pixel_size = atsr_pixel_size() * 1000000  # convert from km^2 to m^2
        # TODO repeat pixel size to full array

        swir_reflectance = self.product.get_band('reflec_nadir_1600').read_as_array()
        self.swir_16 = np.nan_to_num(self._rad_from_ref(swir_reflectance))  # set nan's to zero

        mwir_brightness_temp = self.product.get_band('btemp_nadir_0370').read_as_array()
        self.mwir = self._rad_from_BT(3.7, mwir_brightness_temp)

        solar_elev_angle_rad = np.deg2rad(self.product.get_band('sun_elev_nadir').read_as_array())
        self.sza = np.rad2deg(np.arccos(np.sin(solar_elev_angle_rad)))

    def _rad_from_ref(self, reflectances):
        """
        Converts from 1.6 micron SWIR reflectances to spectral radiances
        Args:
            reflectances: 1.6 micron reflectances

        Returns:
            1.6 micron spectral radiances (W m-2 sr-1 um-1)

        """
        # convert from reflectance to radiance see Smith and Cox 2013
        se_dist = self._compute_sun_earth_distance() ** 2 / np.pi
        return reflectances / 100.0 * proc_const.solar_irradiance[self.sensor] * se_dist

    def _rad_from_BT(self, wvl, b_temp):
        """
        Converts from brightness temperatures to spectral radiances
        Args:
            wvl: target wavelength
            b_temp: observed brightness temperature

        Returns:
            spectral radiances (W m-2 sr-1 um-1)
        """
        c1 = 1.19e-16  # W m-2 sr-1
        c2 = 1.44e-2  # mK
        wt = (wvl * 1.e-6) * b_temp  # m K
        d = (wvl * 1.e-6) ** 5 * (np.exp(c2 / wt) - 1)
        return c1 / d * 1.e-6  # W m-2 sr-1 um-1

    def _compute_sun_earth_distance(self):
        """
        Computes the Sun Earth distance based on the sensor date

        Returns:
            The Sun Earth distance at the time of observation

        """
        doy = datetime.strptime(self.product.id_string[14:22], "%Y%m%d").timetuple().tm_yday
        return 1 + 0.01672 * np.sin(2 * np.pi * (doy - 93.5) / 365.0)

    def _compute_background(self):
        """
        Calculates local mean background MWIR radiances
        for cloud-free, non-hotspot, valid radiance containing  pixels.
        If less than a certain fraction (typically 60%) of the pixels within
        the background window are valid, then the background is set to null

        Returns:
            None

        """
        selem = square(self.background_window_size)
        valid_background = self.background_mask & (self.mwir > 0)
        valid_fraction = rank.mean(valid_background.astype(float), selem)
        self.background_mwir = rank.mean(self.mwir, selem, mask=valid_background)
        self.background_mwir[valid_fraction > proc_const.min_background_proportion] = proc_const.null_value

    def run_detector(self, flares_or_sampling=False) -> None:
        """
        Runs the detector methods on the input data.  If flares_or_sampling
        flag is set then additional processing is performed on fire radiative power,
        local cloud cover and background radiance statistics.

        Args:
            flares_or_sampling: flag to set processing level

        Returns:
            None
        """
        self._load_arrays()
        self._make_night_mask(self.sza)
        self._detect_potential_hotspots(self.swir_16)
        self.hotspots = self.potential_hotspots & self.night_mask

        if flares_or_sampling:
            self.background_mask = ~self.potential_hotspot_mask & self.cloud_free & self.night_mask
            self._compute_frp()
            self.cloudy = ~self.potential_hotspots & ~self.cloud_free & self.night_mask
            self._compute_local_cloudiness()
            self._compute_background()

    def to_dataframe(self,
                     keys=['latitude', 'longitude'],
                     joining_df=None) -> pd.DataFrame:
        """
        Used to return a dataframe containing all needed information
        on the hotspots detected during the run_detector call.  The information
        returned is dependent on the keys provided, and if needed can be screened
        using the joining_df keyword arg.
        Args:
            keys: The data required in the dataframe
            joining_df: An optional  joining dataframe that can be used to reduce the hotspots.

        Returns:
            A dataframe containing the requested and possibly reduced data.

        """
        if not('latitude' in keys and 'longitude' in keys):
            raise KeyError('At a minimum latitude and longitude are required')
        return self._build_dataframe(keys, self.hotspots, joining_df=joining_df)


class SLSDetector(BaseDetector):

    def __init__(self,
                 product,
                 day_night_angle=proc_const.day_night_angle,
                 swir_thresh=proc_const.swir_thresh_sls,
                 cloud_window_size=33):
        """
        Detector implementation for the Sea and Land Surface Temperature Scanning (SLSTR)
        radiometer instrument series.
        Args:
            product: SLSTR data product
            day_night_angle: Solar zenith angle that defines the day/night boundary
            swir_thresh: Threshold which hotspots must exceed to be detected
            cloud_window_size: Image window over which local cloud statistics are computed
        """
        super().__init__(day_night_angle, swir_thresh, cloud_window_size)
        self.product = product
        self.max_view_angle = 22  # degrees
        self.sensor = 'sls'

    def _load_arrays(self) -> None:
        """
        Loads the product data needed for all processing.
        Returns:
            None
        """
        self.latitude = self.product['geodetic_an']['latitude_an'][:]
        self.longitude = self.product['geodetic_an']['longitude_an'][:]
        self.swir_16 = self.product['S5_radiance_an']['S5_radiance_an'][:].filled(0)
        self.swir_22 = self.product['S6_radiance_an']['S6_radiance_an'][:].filled(0)
        self.sza = self._interpolate_array('solar_zenith_tn').filled(0)
        self.vza = self._interpolate_array('sat_zenith_tn').filled(9999)
        self.cloud_free = self.product['flags_an']['cloud_an'][:] == 0
        self.pixel_size = np.loadtxt(fp.path_to_sls_pix_sizes) * 1000000

    def _interpolate_array(self, target) -> np.array:
        """
        Interpolates SLSTR data arrays based on cartesian information
        contained within the sensor product using the RectBivariateSpline
        approach.

        Args:
            target: the product to be interpolated

        Returns:
            The interpolated data
        """
        sat_zn = self.product['geometry_tn'][target][:]

        tx_x_var = self.product['cartesian_tx']['x_tx'][0, :]
        tx_y_var = self.product['cartesian_tx']['y_tx'][:, 0]

        an_x_var = self.product['cartesian_an']['x_an'][:]
        an_y_var = self.product['cartesian_an']['y_an'][:]

        spl = RectBivariateSpline(tx_y_var, tx_x_var[::-1], sat_zn[:, ::-1].filled(0))
        interpolated = spl.ev(an_y_var.compressed(),
                              an_x_var.compressed())
        interpolated = np.ma.masked_invalid(interpolated, copy=False)
        sat = np.ma.empty(an_y_var.shape,
                             dtype=sat_zn.dtype)
        sat[np.logical_not(np.ma.getmaskarray(an_y_var))] = interpolated
        sat.mask = an_y_var.mask
        return sat

    def _make_view_angle_mask(self, vza):
        """
        Screen SLSTR data based on viewing zenith angle so
        that the data is limited to the max view angle (e.g.
        it can be set to 22 degrees to create an ATSR like product).
        Args:
            vza: SLSTR viewing zenith angles

        Returns:
            None
        """
        self.vza_mask = vza <= self.max_view_angle

    def run_detector(self, flares_or_sampling=False) -> None:
        """
        Runs the detector methods on the input data.  If flares_or_sampling
        flag is set then additional processing is performed on fire radiative power,
        local cloud cover and background radiance statistics.

        Args:
            flares_or_sampling: flag to set processing level

        Returns:
            None
        """
        self._load_arrays()
        self._make_night_mask(self.sza)
        self._make_view_angle_mask(self.vza)
        self._detect_potential_hotspots(self.swir_16)
        self.hotspots = self.potential_hotspots & self.night_mask & self.vza_mask

        if flares_or_sampling:
            self._compute_frp()
            self.cloudy = ~self.potential_hotspots & ~self.cloud_free & self.night_mask & self.vza_mask
            self._compute_local_cloudiness()

    def to_dataframe(self,
                     keys=['latitude', 'longitude'],
                     joining_df=None) -> pd.DataFrame:
        """
        Used to return a dataframe containing all needed information
        on the hotspots detected during the run_detector call.  The information
        returned is dependent on the keys provided, and if needed can be screened
        using the joining_df keyword arg.
        Args:
            keys: The data required in the dataframe
            joining_df: An optional  joining dataframe that can be used to reduce the hotspots.

        Returns:
            A dataframe containing the requested and possibly reduced data.

        """
        if not('latitude' in keys and 'longitude' in keys):
            raise KeyError('At a minimum latitude and longitude are required')
        return self._build_dataframe(keys, self.hotspots, joining_df=joining_df)




from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from scipy.interpolate import RectBivariateSpline


class BaseHotspotDetector(ABC):

    def __init__(self,
                 day_night_angle=None,
                 swir_thresh=None):
        self.day_night_angle = day_night_angle
        self.swir_thresh = swir_thresh
        self.hotspots = None

    def _make_night_mask(self, sza: np.ndarray) -> None:
        self.night_mask = sza >= self.day_night_angle

    def _detect_potential_hotspots(self, swir: np.ndarray) -> None:
        self.potential_hotspots = swir > self.swir_thresh

    def _build_dataframe(self, keys) -> pd.DataFrame:
        df = pd.DataFrame()
        for k in keys:
            df[k] = self.__dict__[k][self.hotspots]
        lines, samples = np.where(self.hotspots)
        df['lines'] = lines
        df['samples'] = samples
        return df

    @abstractmethod
    def _load_arrays(self) -> None:
        raise NotImplementedError("Must override _load_arrays")

    @abstractmethod
    def run_detector(self) -> None:
        raise NotImplementedError("Must override run_detector")

    @abstractmethod
    def to_dataframe(self) -> pd.DataFrame:
        raise NotImplementedError("Must override to_dataframe")




class ATXHotspotDetector(BaseHotspotDetector):

    def __init__(self, day_night_angle, swir_thresh, product):
        super().__init__(day_night_angle, swir_thresh)
        self.product = product

    def _load_arrays(self) -> None:
        self.latitude = self.product.get_band('latitude').read_as_array()
        self.longitude = self.product.get_band('latitude').read_as_array()
        self.swir_16 = self.product.get_band('reflec_nadir_1600').read_as_array()
        solar_elev_angle_rad = np.deg2rad(self.product.get_band('sun_elev_nadir').read_as_array())
        self.sza = np.rad2deg(np.arccos(np.sin(solar_elev_angle_rad)))

    def run_detector(self) -> None:
        self._load_arrays()
        self._make_night_mask(self.sza)
        self._detect_potential_hotspots(self.swir_16)
        self.hotspots = self.potential_hotspots & self.night_mask

    def to_dataframe(self) -> pd.DataFrame:
        keys = ['latitude', 'longitude', 'swir_16', 'sza']
        return self._build_dataframe(keys)


class SLSHotspotDetector(BaseHotspotDetector):

    def __init__(self, day_night_angle, swir_thresh, product):
        super().__init__(day_night_angle, swir_thresh)
        self.product = product
        self.max_view_angle = 22  # degrees

    def _load_arrays(self) -> None:
        self.latitude = self.product['geodetic_an']['latitude_an'][:]
        self.longitude = self.product['geodetic_an']['longitude_an'][:]
        self.swir_16 = self.product['S5_radiance_an']['S5_radiance_an'][:].filled(0)
        self.swir_22 = self.product['S6_radiance_an']['S6_radiance_an'][:].filled(0)
        self.sza = self._interpolate_array('solar_zenith_tn').filled(0)
        self.vza = self._interpolate_array('sat_zenith_tn').filled(100)

    def _interpolate_array(self, target) -> np.array:
        sat_zn = self.product['geometry_tn'][target][:]

        tx_x_var = self.product['cartesian_tx']['x_tx'][0, :]
        tx_y_var = self.product['cartesian_tx']['y_tx'][:, 0]

        an_x_var = self.product['cartesian_an']['x_an'][:]
        an_y_var = self.product['cartesian_an']['y_an'][:]

        spl = RectBivariateSpline(tx_y_var, tx_x_var[::-1], sat_zn[:, ::-1].filled(0))
        interpolated = spl.ev(an_y_var.compressed(),
                              an_x_var.compressed())
        interpolated = np.ma.masked_invalid(interpolated, copy=False)
        sat_zn = np.ma.empty(an_y_var.shape,
                             dtype=sat_zn.dtype)
        sat_zn[np.logical_not(np.ma.getmaskarray(an_y_var))] = interpolated
        sat_zn.mask = an_y_var.mask
        return sat_zn

    def _make_view_angle_mask(self, vza):
        self.vza_mask = vza <= self.max_view_angle

    def run_detector(self) -> None:
        self._load_arrays()
        self._make_night_mask(self.sza)
        self._make_view_angle_mask(self.vza)
        self._detect_potential_hotspots(self.swir_16)
        self.hotspots = self.potential_hotspots & self.night_mask & self.vza_mask

    def to_dataframe(self) -> pd.DataFrame:
        keys = ['latitude', 'longitude', 'swir_16', 'swir_22', 'sza', 'vza']
        return self._build_dataframe(keys)


import numpy as np
from astropy.time import Time


def _r1(angle_rad: float) -> np.ndarray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, c, s],
            [0.0, -s, c],
        ]
    )


def _r3(angle_rad: float) -> np.ndarray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array(
        [
            [c, s, 0.0],
            [-s, c, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def _moon_orientation_iau_2009(mjd: float, mjd_scale: str = "utc") -> tuple[float, float, float]:
    """
    Return Moon orientation angles (RA pole, Dec pole, W) in radians.

    Uses IAU 2009/2015-style trigonometric series for the Moon orientation,
    with time measured in TT from J2000.
    """
    t = Time(mjd, format="mjd", scale=mjd_scale).tt
    d = t.jd - 2451545.0
    T = d / 36525.0

    deg2rad = np.deg2rad

    E1 = deg2rad(125.045 - 0.0529921 * d)
    E2 = deg2rad(250.089 - 0.1059842 * d)
    E3 = deg2rad(260.008 + 13.0120009 * d)
    E4 = deg2rad(176.625 + 13.3407154 * d)
    E5 = deg2rad(357.529 + 0.9856003 * d)
    E6 = deg2rad(311.589 + 26.4057084 * d)
    E7 = deg2rad(134.963 + 13.0649930 * d)
    E8 = deg2rad(276.617 + 0.3287146 * d)
    E9 = deg2rad(34.226 + 1.7484877 * d)
    E10 = deg2rad(15.134 - 0.1589763 * d)
    E11 = deg2rad(119.743 + 0.0036096 * d)
    E12 = deg2rad(239.961 + 0.1643573 * d)
    E13 = deg2rad(25.053 + 12.9590088 * d)

    alpha_deg = (
        269.9949
        + 0.0031 * T
        - 3.8787 * np.sin(E1)
        - 0.1204 * np.sin(E2)
        + 0.0700 * np.sin(E3)
        - 0.0172 * np.sin(E4)
        + 0.0072 * np.sin(E6)
        - 0.0052 * np.sin(E10)
        + 0.0043 * np.sin(E13)
    )

    delta_deg = (
        66.5392
        + 0.0130 * T
        + 1.5419 * np.cos(E1)
        + 0.0239 * np.cos(E2)
        - 0.0278 * np.cos(E3)
        + 0.0068 * np.cos(E4)
        - 0.0029 * np.cos(E6)
        + 0.0009 * np.cos(E7)
        + 0.0008 * np.cos(E10)
        - 0.0009 * np.cos(E13)
    )

    w_deg = (
        38.3213
        + 13.17635815 * d
        - 1.4e-12 * d * d
        + 3.5610 * np.sin(E1)
        + 0.1208 * np.sin(E2)
        - 0.0642 * np.sin(E3)
        + 0.0158 * np.sin(E4)
        + 0.0252 * np.sin(E5)
        - 0.0066 * np.sin(E6)
        - 0.0047 * np.sin(E7)
        - 0.0046 * np.sin(E8)
        + 0.0028 * np.sin(E9)
        + 0.0052 * np.sin(E10)
        + 0.0040 * np.sin(E11)
        + 0.0019 * np.sin(E12)
        - 0.0044 * np.sin(E13)
    )

    return deg2rad(alpha_deg), deg2rad(delta_deg), deg2rad(w_deg)


def lunar_altaz_from_mcmf(
    lon: float,
    lat: float,
    mjd: float,
    ra: float,
    dec: float,
    *,
    input_in_degrees: bool = False,
    mjd_scale: str = "utc",
) -> tuple[float, float]:
    """
    Compute local altitude and azimuth of a celestial object as seen from the Moon.

    Parameters
    ----------
    lon, lat : float
        Lunar MCMF longitude and latitude of observer.
    mjd : float
        Observation timestamp in Modified Julian Date.
    ra, dec : float
        Object right ascension and declination in Earth equatorial inertial frame
        (treated as ICRS/J2000-like inertial direction).
    input_in_degrees : bool, optional
        If True, lon/lat/ra/dec are interpreted as degrees. Output is always radians.
    mjd_scale : str, optional
        Time scale for MJD input, e.g. 'utc', 'tt', 'tdb'.

    Returns
    -------
    altitude : float
        Local altitude in radians (pi/2 at zenith, 0 at horizon).
    azimuth : float
        Local azimuth in radians (0 = north, pi/2 = east), wrapped to [0, 2*pi).
    """
    if input_in_degrees:
        lon, lat, ra, dec = np.deg2rad([lon, lat, ra, dec])

    alpha_p, delta_p, w = _moon_orientation_iau_2009(mjd=mjd, mjd_scale=mjd_scale)

    # Inertial -> MCMF rotation (IAU convention)
    r_inertial_to_mcmf = _r3(w) @ _r1(np.pi / 2 - delta_p) @ _r3(alpha_p + np.pi / 2)
    r_mcmf_to_inertial = r_inertial_to_mcmf.T

    # Local ENU basis in MCMF
    cos_lat, sin_lat = np.cos(lat), np.sin(lat)
    cos_lon, sin_lon = np.cos(lon), np.sin(lon)

    up_mcmf = np.array([cos_lat * cos_lon, cos_lat * sin_lon, sin_lat])
    east_mcmf = np.array([-sin_lon, cos_lon, 0.0])
    north_mcmf = np.array([-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat])

    # Rotate local basis to inertial frame
    up = r_mcmf_to_inertial @ up_mcmf
    east = r_mcmf_to_inertial @ east_mcmf
    north = r_mcmf_to_inertial @ north_mcmf

    # Object line-of-sight unit vector in inertial frame from RA/Dec
    obj = np.array([
        np.cos(dec) * np.cos(ra),
        np.cos(dec) * np.sin(ra),
        np.sin(dec),
    ])

    altitude = np.arcsin(np.clip(np.dot(obj, up), -1.0, 1.0))
    azimuth = np.mod(np.arctan2(np.dot(obj, east), np.dot(obj, north)), 2.0 * np.pi)

    return float(altitude), float(azimuth)

import os
import numpy as np
from datetime import datetime

import sunpy.map as smp
from sunpy.coordinates import frames
import astropy.units as u
from astropy.coordinates import SkyCoord

class ts_processing:
    """
    This class is responsible for retrieving the magnetogram information.
    """
    def __init__(self, path, ar_no):
        """
        A constructor that initializes the class variables.

        :param path: Initialize path to samples
        :param ar_no: Initialize the HARP number of the sample input
        """
        self.path = path
        self.ar_no = ar_no

    def process(self):
        """
        This method process the file from the given magnetogram folder and check for severe projection effect (greater than 70 degrees East-West)

        :return:
            List of magnetogram maps after checking
        """
        # Read whole series
        time_list, mag_map_list = self.get_magnetogram_files()

        # Check SPEI
        time_list_centroid, mag_map_list_centroid = self.check_SPEI(time_list, mag_map_list)
        print("Number of Filtered Magnetograms after checking SPEI: " + str(len(mag_map_list)-len(mag_map_list_centroid)))

        # Check corner
        time_list_final, mag_map_list_final = self.patch_corner_check(time_list_centroid, mag_map_list_centroid)
        print("Number of Filtered Magnetograms after checking Patch Corner: " + str(len(mag_map_list_centroid)-len(mag_map_list_final)))

        return mag_map_list_final

    def get_TS(self, filename):
        """
        This method retrieved time information based on the filename.

        :param filename: Filename of the input FIT maps

        :return: An ISO format of the magnetogram time
        """
        part = filename.split('.')[3]
        year = part[0:4]
        month = part[4:6]
        day = part[6:8]
        hr = part[9:11]
        minute = part[11:13]
        sec = part[13:15]
        iso = year + '-' + month + '-' + day + 'T' + hr + ':' + minute + ':' + sec
        TS = datetime.fromisoformat(iso)
        return TS

    def get_magnetogram_files(self):
        """
        This method generate magnetogram maps from the folder provided.
        
        :return: 
            List of ISO format of the magnetogram time
            List of magnetogram maps
        """
        TS_list = []
        map_list = []
        os.chdir(self.path + str(self.ar_no) + '/')
        files = os.listdir()

        magnetogram_files = sorted([file for file in files if 'magnetogram' in file])
        if magnetogram_files == None or magnetogram_files == []:
            # If no magnetogram containing in the file, return empty
            return [], []

        for f in magnetogram_files:
            TS_list.append(self.get_TS(f))
            map_list.append(smp.Map(f))

        os.chdir('../..')

        return TS_list, map_list

    def great_circle_distance(self, x, y):
        """
        This method calculates the great circle distance (central angle) between two points if the angle is required
        from the disk center on the surface of a sphere with radius r.

        :param x: Carrington Longitude between the patch center and disk center
        :param y: Carrington Latitude between the patch center and disk center

        :return: Calculated great circle distance
        """
        x = x * np.pi / 180
        y = y * np.pi / 180
        dlong = x[1] - y[1]
        den = np.sin(x[0]) * np.sin(y[0]) + np.cos(x[0]) * np.cos(y[0]) * np.cos(dlong)
        num = (np.cos(y[0]) * np.sin(dlong)) ** 2 + (
                    np.cos(x[0]) * np.sin(y[0]) - np.sin(x[0]) * np.cos(y[0]) * np.cos(dlong)) ** 2

        sig = np.arctan2(np.sqrt(num), den) * 180 / np.pi
        return sig

    def heliocentric_angle(self, CRVAL1, CRVAL2, CRLN_OBS, CRLT_OBS):
        """
        This method calculate the heliocentric angle from the given Carrington Longitude and Latitude

        :param CRVAL1: Carrington Longitude at the center of patch
        :param CRVAL2: Carrington Latitude at the center of patch
        :param CRLN_OBS: Carrington Longitude of the Observer (at center of solar disk)
        :param CRLT_OBS: Carrington Latitude of the Observer (at center of solar disk)

        :return: Calculated great circle distance
        """
        # Calculate the Stonyhurst Latitude and Longitude:
        longitude = CRVAL1 - CRLN_OBS
        latitude = CRVAL2 - CRLT_OBS

        x = np.array([latitude, longitude])
        y = np.array([0., 0.])

        return self.great_circle_distance(x, y)

    def get_hc_angle(self, header):
        """
        This function calculate radial heliocentric angle of any coordinate on image

        :param header: The header information containing in magnetogram maps

        :return: Calculated heliocentric angle
        """
        crval1 = header['CRVAL1']
        crval2 = header['CRVAL2']
        crln_obs = header['CRLN_OBS']
        crlt_obs = header['CRLT_OBS']

        hc_angle = self.heliocentric_angle(crval1, crval2, crln_obs, crlt_obs)
        header['HC_ANGLE'] = hc_angle

        return hc_angle

    def check_SPEI(self, TS_list, map_list):
        """
        This method check for SPEI within 70 degree of heliocentric angle

        :param TS_list: List of ISO formated magnetogram time
        :param map_list: List of magnetogram maps

        :return:
            List of ISO formated magnetogram time after checking SPEI
            List of magnetogram maps after checking SPEI
        """
        if TS_list == [] or TS_list == None:
            # If no magnetogram containing in the file, return empty
            return [], []

        SPEI_TS_list = []
        SPEI_map_list = []
        for time, value in zip(TS_list, map_list):
            angle = self.get_hc_angle(value.fits_header)
            if angle < 70:
                SPEI_TS_list += [time]
                SPEI_map_list += [value]

        return SPEI_TS_list, SPEI_map_list

    def patch_corner_check(self, TS_list, map_list):
        """
        This method checks for whether the patch corner is within the great circle distance.

        :param TS_list: List of ISO formated magnetogram time
        :param map_list: List of magnetogram maps

        :return:
            List of ISO formated magnetogram time after checking patch corner
            List of magnetogram maps after checking patch corner
        """
        ls_map = []
        ls_n_TS = []

        if TS_list == [] or TS_list == None:
            return [], []

        else:
            for t, file in zip(TS_list, map_list):
                ob_time = file.fits_header['DATE-OBS']

                bl_coor = self.coord_transformer_hsc(file.bottom_left_coord.lon.deg, file.bottom_left_coord.lat.deg, ob_time)
                tr_coor = self.coord_transformer_hsc(file.top_right_coord.lon.deg, file.top_right_coord.lat.deg, ob_time)

                bl_lat_lon = np.array([bl_coor.lat.deg, bl_coor.lon.deg])  # bottom_left
                #br_lat_lon = np.array([bl_coor.lat.deg, tr_coor.lon.deg])  # bottom_right
                #tl_lat_lon = np.array([tr_coor.lat.deg, bl_coor.lon.deg])  # top_left
                tr_lat_lon = np.array([tr_coor.lat.deg, tr_coor.lon.deg])  # top_right

                bl_circle_dist = self.great_circle_distance(bl_lat_lon, np.array([0., 0.]))  # bottom_left
                #br_circle_dist = self.great_circle_distance(br_lat_lon, np.array([0., 0.]))  # bottom_right
                #tl_circle_dist = self.great_circle_distance(tl_lat_lon, np.array([0., 0.]))  # top_left
                tr_circle_dist = self.great_circle_distance(tr_lat_lon, np.array([0., 0.]))  # top_right

                if bl_circle_dist < 70 and tr_circle_dist < 70:
                    # Check if the corners are within the great circle distance
                    ls_map.append(file)
                    ls_n_TS.append(t)

            return ls_n_TS, ls_map

    def coord_transformer_hgc(self, x, y, obs_time):
        """
        This method checks the coordinate based on Stonyhurst measurements and transform to Carrington measurements

        :param x: Stonyhurst longitude
        :param y: Stonyhurst latitude
        :param obs_time: Observation time of current magnetogram

        :return: Transformed coordinate
        """
        c = SkyCoord(x * u.deg, y * u.deg, frame=frames.HeliographicStonyhurst, obstime=obs_time, observer="earth")
        c_hgs = c.transform_to(frames.HeliographicCarrington)
        return c_hgs

    def coord_transformer_hsc(self, x, y, obs_time):
        """
        This method checks the coordinate based on Carrington measurements and transform to Stonyhurst measurements

        :param x: Carrington longitude
        :param y: Carrington latitude
        :param obs_time: Observation time of current magnetogram

        :return: Transformed coordinate
        """
        c = SkyCoord(x * u.deg, y * u.deg, frame=frames.HeliographicCarrington, obstime=obs_time, observer="earth")
        c_hgs = c.transform_to(frames.HeliographicStonyhurst)
        return c_hgs


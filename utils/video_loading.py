import os
import cv2
import glob
import sunpy.map
import numpy as np
from itertools import islice
import matplotlib.pyplot as plt

from IPython.display import Video

import warnings
warnings.filterwarnings('ignore')


class video_loader:
    """
    This class is responsible for a generation of a video of magnetogram images overlaid with Polarity Inversion Line, Region of Polarity Inversion, and Convex Hull masks.
    """

    def __init__(self):
        """
        A constructor that initializes the class variables.
        """
        pass

    def build_images(self, path, outpath):
        """
        Build and save overlaid magnetogram image outputs.

        :param path: Initialize path to samples

        :param outpath: Initialize path for outputs
        """
        self.outpath = outpath
        backgrounds = sorted(glob.glob(path + "/*magnetogram*.fits"))
        foregrounds_pil = sorted(glob.glob(path + "/*_PIL*.png"))

        for bg in backgrounds:
            b_date = bg.split('_TAI')[0].rsplit('.')[-1].replace('_', '')
            for pil in foregrounds_pil:
                date = pil.split('_BLOS')[0].split('_', 2)[-1]
                pil_date = date.replace('-', '').replace(':', '').replace('_', '')
                if (b_date == pil_date):
                    ropi = glob.glob(path + "/*" + date + "*RoPI*.png")
                    chpil = glob.glob(path + "/*" + date + "*CHPIL*.png")
                    self.apply_params(bg, pil, ropi[0], chpil[0], date)

    def mask_img(self, img):
        """
        Create masked images.

        :param img: Array of image values

        :return: Masked image values
        """
        return np.ma.masked_where(img.astype(float) == 0, img.astype(float))

    def apply_params(self, background, pil, ropi, chpil, date):
        """
        Overlay Polarity Inversion Line, Region of Polarity Inversion, and Convex  Hull masks over magnetogram images.

        :param background: Initialize background image

        :param pil: Initialize PIL image

        :param ropi: Initialize RoPI image

        :param chpil: Initialize Convex Hull image

        :param date: Initialize dat
        """
        hmi_magmap = sunpy.map.Map(background)

        ropi_mask = self.mask_img(plt.imread(ropi))
        pil_mask = self.mask_img(plt.imread(pil))
        chpil_mask = self.mask_img(plt.imread(chpil))

        cmap = plt.cm.spring
        cmap = cmap.set_bad(color='white')

        fig = plt.figure(figsize=(10, 8))

        hmi_magmap.plot_settings['cmap'] = 'hmimag'
        hmi_magmap.plot_settings['norm'] = plt.Normalize(-1500, 1500)
        im_hmi = hmi_magmap.plot()
        cb = plt.colorbar(im_hmi, fraction=0.019, pad=0.1)

        plt.xlabel('Carrington Longitude [deg]', fontsize=16)
        plt.ylabel('Latitude [deg]', fontsize=16)
        plt.imshow(chpil_mask, 'bone', interpolation='none', alpha=0.6)
        plt.imshow(ropi_mask, 'cool', interpolation='none', alpha=0.8)
        plt.imshow(pil_mask, cmap, interpolation='none', alpha=1)

        cb.set_label("LOS Magnetic Field [Gauss]")
        file_path = os.path.join(self.outpath, date + '.png')
        plt.savefig(file_path)
        plt.close(fig)

    def display_video(self, path):
        """
        Create and save the video out of the images in the given path.

        :param path: Initialize path to samples

        :return: Video slide of magnetogram images
        """
        img_array = []
        file_name = path + '_video/mag_map.mp4'
        size = (None, None)
        for filename in sorted(glob.glob(path + '/*.png')):
            img = cv2.imread(filename)
            height, width, layers = img.shape
            size = (width, height)
            img_array.append(img)
        out = cv2.VideoWriter(file_name, cv2.VideoWriter_fourcc(*'vp09'), 2, size)

        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()

        return Video(file_name, embed=True)
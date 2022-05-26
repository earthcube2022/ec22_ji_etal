import sys
sys.path.append('../utils')
from utils.ts_processing import ts_processing
from utils.region_detection import pos_neg_detection

import os
import numpy as np
import pandas as pd
from skimage import feature
from skimage.morphology import erosion, dilation, opening, closing, white_tophat
from skimage.morphology import black_tophat, skeletonize, convex_hull_image, thin
from skimage.morphology import square, rectangle, diamond, disk, cube, octahedron, ball, octagon, star
from skimage.measure import label, regionprops,regionprops_table,inertia_tensor_eigvals,inertia_tensor,moments_central,moments_normalized,moments_hu,perimeter

class detection:
    """
    This class is the main functionality of detecting the Magnetic Polarity Inversion Lines (MPILs).
    """
    def __init__(self, path, ar_no):
        """
        A constructor that initializes the class variables.

        :param path: Initialize path to samples
        :param ar_no: Initialize the HARP number of the sample input
        """
        self.path = path
        self.ar_no = ar_no
        self.dt = pos_neg_detection()
        self.ts = ts_processing(self.path, self.ar_no)

    def PIL_detect(self, pos_gauss = 100, neg_gauss= -100, size_kernel = 4):
        """
        This method includes all the process for detecting MPILs from magnetograms.
        
        :param pos_gauss: Gauss threshold for identifying positive polarity regions
        :param neg_gauss: Gauss threshold for identifying negative polarity regions
        
        :return: 
            Detected MPILs
            Detected MPILs labels
        """
        self.mag_map_list = self.ts.process()
        
        self.PIL_series_map = self.PIL_series(self.mag_map_list, pos_g_val=pos_gauss, neg_g_val=neg_gauss, size=size_kernel)
        
        self.pil_df_orig, self.pil_label_orig = self.PIL_dataframe(self.PIL_series_map)

        return self.pil_df_orig, self.pil_label_orig

    def PIL_series(self, map_list, pos_g_val=100, neg_g_val=-100, size=4):
        """
        This method detects the series of MPILs (candidate RoPIs) from the given list of magnetogram maps.
        
        :param map_list: List of magnetogram maps
        :param pos_g_val: Gauss threshold for identifying positive polarity regions
        :param neg_g_val: Gauss threshold for identifying negative polarity regions
        
        :return: The PIL series and submap of single HARP number
        """
        pil_maps = []
        success_count = 0

        for i, sub_map in enumerate(map_list):
            pos_map, neg_map = self.dt.identify_pos_neg_region(sub_map, pos_gauss=pos_g_val, neg_gauss=neg_g_val)

            pos_edge = self.dt.edge_detection(pos_map)
            neg_edge = self.dt.edge_detection(neg_map)

            pos_dil_edge = self.dt.buff_edge(pos_edge, size=size)
            neg_dil_edge = self.dt.buff_edge(neg_edge, size=size)

            pil_maps.append(self.PIL_extraction(pos_dil_edge, neg_dil_edge, sub_map))

            success_count += 1

        print("Number of Detected Candidate RoPIs: ", success_count)

        return pil_maps  

    def PIL_extraction(self, buff_pos, buff_neg, fits_map):
        """
        This method extract candidate Region of Polarity Inversions (RoPIs) based on the intersection areas of spatially buffered opposite polarity region edges.

        :param buff_pos: Buffered edges of positive polarity regions
        :param buff_neg: Buffered edges of negative polarity regions
        :param fits_map: Magnetogram maps

        :return: Detected candidate RoPIs magetogram maps
        """
        pil_mask = np.invert(np.isnan(fits_map.data))

        # Index (pixel) coordinates of PIL intersection
        pil_result = np.where(buff_pos & buff_neg & pil_mask)

        pil_map = np.zeros(fits_map.data.shape)

        pil_map[pil_result[0], pil_result[1]] = 1

        return pil_map

    def PIL_dataframe(self, ropi_map_list):
        """
        This method builds up dataframes and generates boolean labels for candidate RoPIs.

        :param ropi_map_list: List of Detected candidate RoPIs

        :return:
            List of Detected candidate RoPIs dataframe
            List of Detected candidate RoPIs labels
        """
        ropi_df = []
        ropi_labels = []

        for i, file in enumerate(ropi_map_list):
            if (np.all(file == False)):
                ropi_df.append(None)
                ropi_labels.append(None)
            else:
                ob_labels = label(file, connectivity=2)
                ropi_df.append(self.get_prop_labels(file))
                ropi_labels.append(ob_labels)

        return ropi_df, ropi_labels

    def get_prop_labels(self, input):
        """
        This method creates a dataframe including all the property labels

        :param input: Input label

        :return: A dataframe of property labels
        """
        ob_labels = label(input, connectivity=2)

        prop = ['label', 'centroid', 'orientation', 'major_axis_length', 'minor_axis_length', 'bbox', 'area',
                'coords', 'image', 'bbox_area']
        props_table = regionprops_table(ob_labels, properties=prop)

        return pd.DataFrame(props_table)

    def filter_by_strength(self, threshold=0.95):
        """
        This method generates a label matrix based on the given unsigned magnetic flux threshold.

        :param threshold: Flux filter threshold for keeping the unsigned magnetic flux contained

        :return: Label matrix after filtering by strength
        """
        filter_label = []

        for i, file in enumerate(self.pil_df_orig):
            if file is not None:
                file['strength'] = self.single_strength(file, self.mag_map_list[i])
                file.sort_values(by=['strength'], ascending=False, inplace=True)

                # Generate ['cum_percent'] column
                file['cum_percent'] = file['strength'].cumsum() / sum(abs(file['strength']))  
                file['str_percent'] = file['strength'] / sum(abs(file['strength']))
                # Cut threshold (handle if exist minority PIL)
                file['cut_threshold'] = file['cum_percent'] - file['str_percent']  
                file['strength_keep'] = file.apply(lambda row: row.cut_threshold <= threshold, axis=1)

                # Generate the label matrix satisfy the filtering threshold
                n_label = self.filter_strength_pil(file, threshold, self.pil_label_orig[i])
                filter_label.append(n_label)
            else:
                filter_label.append(None)

        return filter_label

    def single_strength(self, pil_df, submap_series):
        """
        This method calculate the total Gauss of each MPIL using: sum(abs(pil[row,column].data))
        
        :param pil_df: List of Detected candidate MPILs dataframe
        :param submap_series: Series of magnetogram maps
        
        :return: List of Gauss strengths for each MPIL
        """
        pil_strgt_lst = []

        for i, pil_row in pil_df.iterrows():
            r_idx = pil_row.coords[:, 0]
            col_idx = pil_row.coords[:, 1]

            # Check for single strength
            pil_strength = sum(abs(submap_series.data[r_idx, col_idx]))

            pil_strgt_lst.append(pil_strength)

        return pil_strgt_lst

    def filter_strength_pil(self, pil_df, threshold, pil_label):
        """
        This method generate the label matrix that satisfied the filtering threshold.

        :param pil_df: Detected candidate MPIL dataframe
        :param threshold: Flux filter threshold for keeping the unsigned magnetic flux contained
        :param pil_label: Corresponding detected candidate MPIL label

        :return: MPIL label within the filtering threshold
        """
        # Create single dataframe with the cutting threshold which contains ~95% PIL flux
        cut_label = pil_df[pil_df['cut_threshold'] > threshold].label.values

        # Check individual pil index which total strength not equal to initial total flux
        cut_idx = np.isin(pil_label, cut_label)

        # Set small PIL flux label to 0, only keep big PIL flux label
        pil_label[cut_idx] = 0

        return pil_label

    def thin_strength_label(self, label_matrix):
        """
        This method creates label matrix by using morphological thinning operation.

        :param label_matrix: MPIL label matrix

        :return:
            List of thinned MPIL dataframes
            List of thinned MPIL labels
        """
        strength_binary_image = []
        thin_dataframe = []
        thin_binary_labels = []

        for i, file in enumerate(label_matrix):
            if file is not None:
                strength_binary_image.append(np.zeros(file.shape) + file)
                
                thin_binary = self.label_thin(file)

                prop = ['label', 'centroid', 'orientation', 'major_axis_length', 'minor_axis_length', 'bbox', 'area',
                        'coords', 'image', 'bbox_area', 'perimeter', 'convex_area']
                props_table = regionprops_table(thin_binary, properties=prop)

                thin_dataframe.append(pd.DataFrame(props_table))
                thin_binary_labels.append(thin_binary)

            else:
                strength_binary_image.append(None)
                thin_dataframe.append(None)
                thin_binary_labels.append(None)

        self.strength_binary_image = strength_binary_image

        return thin_dataframe, thin_binary_labels

    def label_thin(self, orig_label):
        """
        This method thin the original filtered label matrix and create a binary mask of the original label matrix.

        :param orig_label: Original Label matrix

        :return: Label matrix mask after thinning operation
        """
        pil_thin = thin(orig_label)
        # Keep the original label and set non-thinning label as 0
        orig_label[~pil_thin] = 0

        return orig_label

    def filter_by_size(self, thin_df, thin_image, size_threshold=14):
        """
        This method further filter the thinned MPILs with a size threshold.

        :param thin_df: List of thinned MPIL dataframes
        :param thin_image: List of thinned MPILs

        :return: List of length filtering binary MPIL image with label
        """
        for i, file in enumerate(thin_df):
            if file is not None:
                # Filter MPIL by predefined size threshold
                cut_thin_label = file[file['area'] < size_threshold].label.values
                file['length_keep'] = file.apply(lambda row: row.area >= size_threshold, axis=1)

                if len(cut_thin_label) > 0:
                    cut_idx = np.isin(thin_image[i], cut_thin_label)
                    # Remove filtered instances after filtering
                    thin_image[i][cut_idx] = 0

        return thin_image

    def get_convex_image(self, pil_binary):
        """
        This method creates the convex image from MPIL binary labels.

        :param pil_binary: Binary label of MPILs

        :return: Binary labels of convex image
        """
        convex_binary = convex_hull_image(pil_binary).astype('int')

        return convex_binary

    def get_final_pil(self, filter_label, final_label):
        """
        This method checks and removes labels from matrix that are not within our previous detected MPIL labels.

        :return: Final MPIL Label matrix after checking
        """
        lab_matrix = np.zeros(filter_label.shape) + filter_label

        remove_idx = np.isin(lab_matrix, final_label, invert=True)

        # Final binary image of PIL (after strength and thinning)
        lab_matrix[remove_idx] = 0

        return lab_matrix

    def RoPI(self, filter_size_df):
        """
        This method generates the final RoPI after filtering based on MPILs size threshold.

        :param filter_size_df: List of thinned MPIL dataframes

        :return: RoPI binary labels
        """
        RoPI_label = []

        for i, file in enumerate(filter_size_df):
            if file is not None:
                # Filter MPIL based on size label
                size_label = list(set(list(file[file['length_keep'] == True].label.values)))
                RoPI_label.append(self.get_final_pil(self.strength_binary_image[i], size_label))

            else:
                RoPI_label.append(None)

        return RoPI_label


    def convex_pil(self, pil_thin):
        """
        This method finds the convex hull from the given MPIL binary strength.

        :param pil_thin: List of size filtered binary MPIL images

        :return: Convex hull binary labels
        """
        convex_binary = []

        for i, file in enumerate(pil_thin):
            if file is not None:
                f_conv_image = self.get_convex_image(file)
                convex_binary.append(f_conv_image)

            else:
                convex_binary.append(None)

        return convex_binary

    def check_header(self, magmap):
        """
        This method checks for the index of matching magnetogram observation time

        :param magmap: Magnetogram maps

        :return: Matching index
        """
        for i in range(len(self.mag_map_list)):
            if magmap.fits_header['DATE-OBS'] == self.mag_map_list[i].fits_header['DATE-OBS']:
                return i

    def check_outpath(self, outpath):
        """
        This method checks for whether the output path exists

        :param outpath: Output path
        """
        if not os.path.isdir(outpath+str(self.ar_no)):
            ar_outpath = os.path.join(outpath,str(self.ar_no))
            ar_outpath_video = os.path.join(outpath,str(self.ar_no)+'_video')
            os.makedirs(ar_outpath)
            os.makedirs(ar_outpath_video)
            print("Path does not exist, create: ")
            print(ar_outpath)
            print(ar_outpath_video)
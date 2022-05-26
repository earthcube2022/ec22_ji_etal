import numpy as np
from skimage import feature
from skimage.morphology import erosion, dilation, opening, closing, white_tophat
from skimage.morphology import black_tophat, skeletonize, convex_hull_image, thin
from skimage.morphology import square, rectangle, diamond, disk, cube, octahedron, ball, octagon, star
from skimage.measure import label, regionprops,regionprops_table,inertia_tensor_eigvals,inertia_tensor,moments_central,moments_normalized,moments_hu,perimeter

class pos_neg_detection:
    """
    This class is responsible for detecting region of opposite polarity inversions based on the Gauss thresholds
    """
    def __init__(self):
        """
        A constructor that initializes the class variables.
        """
        pass

    def identify_pos_neg_region(self, fits_map, pos_gauss = 100, neg_gauss= -100):
        """
        Identifying the positive and negative polarity regions based on given thresholds
        
        :param fits_map: HMI Image Map
        :param pos_gauss: Gauss threshold for positive polarity regions
        :param neg_gauss: Gauss threshold for negative polarity regions
        
        :return: The binary masks of opposite polarity regions
        """
        pos_map = np.zeros(fits_map.data.shape)
        neg_map = np.zeros(fits_map.data.shape)

        np.warnings.filterwarnings('ignore')
        result_pos = np.where(fits_map.data >= pos_gauss)
        result_neg = np.where(fits_map.data <= neg_gauss)

        pos_map[result_pos[0],result_pos[1]] = 1
        neg_map[result_neg[0],result_neg[1]] = 1

        return pos_map, neg_map

    def edge_detection(self, binary_map):
        """
        Obtain the boundaries of positive and negative polarity regions separately by using the Canny edge operator.
        
        :param binary_map: The binary masks of polarity regions
        
        :return: Detected edges of polarity regions
        """
        sig = 1
        # sigma: float, optional
        # Standard deviation of the Gaussian filter.

        low_threshold = None
        # low_threshold: float, optional
        # Lower bound for hysteresis thresholding (linking edges). If None, low_threshold is set to 10% of dtype’s max.
        
        high_threshold = None
        # high_threshold: float, optional
        # Upper bound for hysteresis thresholding (linking edges). If None, high_threshold is set to 20% of dtype’s max.

        edges = feature.canny(binary_map, sigma=sig, low_threshold=low_threshold, high_threshold=high_threshold)

        return edges

    def buff_edge(self, edges, size=4):
        """
        Spatially buffer the edges of polarity regions in binary images with a pre-defined kernel

        :param edges: Detected edges of polarity regions
        :param size: Kernel size for buffering

        :return: Buffered edges of polarity regions
        """
        selem = square(size)
        dilated_edges = dilation(edges, selem)

        return dilated_edges

    def mask_img(self, img):
        """
        Create masked images
        
        :param img: Array of image values
        
        :return: Masked image values
        """
        return np.ma.masked_where(img.astype(float) == 0, img.astype(float))

    def mask_pil(self, img):
        """
        Create masked images

        :param img: Label of original PILs

        :return: Masked PILs
        """
        return np.ma.masked_where(img.astype(bool).astype(float) == 0, img.astype(bool).astype(float))

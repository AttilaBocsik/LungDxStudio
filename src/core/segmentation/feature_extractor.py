# src/core/segmentation/feature_extractor.py
import cv2
import numpy as np
from skimage.filters import gaussian
from skimage.segmentation import active_contour


class FeatureExtractor:
    """
    Képjellemzők kinyerésére szolgáló osztály, amely Gabor-szűrőket és
    aktív kontúr (snake) algoritmust használ a textúraelemzéshez és alakzatfinomításhoz.
    """

    def __init__(self):
        """
        Inicializálja a FeatureExtractor osztályt és előre legyártja a Gabor-szűrőmagokat.
        """
        self.kernels = self._create_gabor_kernels()

    def _create_gabor_kernels(self):
        """
        Létrehoz egy Gabor-szűrőbankot különböző orientációkkal, skálákkal és hullámhosszakkal.

        Returns:
            list: OpenCV Gabor-kernelek listája.
        """
        kernels = []
        for theta in range(2):
            theta = theta / 4. * np.pi
            for sigma in (1, 3):
                for lamda in np.arange(0, np.pi, np.pi / 4):
                    for gamma in (0.05, 0.5):
                        kernel = cv2.getGaborKernel((5, 5), sigma, theta, lamda, gamma, 0, ktype=cv2.CV_32F)
                        kernels.append(kernel)
        return kernels

    def apply_gabor(self, image):
        """
        Alkalmazza a Gabor-szűrőbankot a bemeneti képen.

        Args:
            image (numpy.ndarray): A bemeneti szürkeárnyalatos kép.

        Returns:
            numpy.ndarray: A kinyert textúra-jellemzők lapított (reshaped) tömbje.
        """
        feats = []
        for kernel in self.kernels:
            fimg = cv2.filter2D(image, cv2.CV_8UC3, kernel)
            feats.append(fimg.reshape(-1))
        return np.array(feats)

    def refine_with_snake(self, image, bbox):
        """
        Aktív kontúr (Snake) algoritmus segítségével finomítja a daganat körvonalát
        egy megadott befoglaló kereten (bounding box) belül.

        Args:
            image (numpy.ndarray): Szürkeárnyalatos kép.
            bbox (dict): A detektált terület koordinátái (xmin, ymin, xmax, ymax).

        Returns:
            tuple: (final_image, snake_points, init_points) - a rajzolt kép,
                   a finomított pontok és a kiinduló körvonal pontjai.
        """
        # image = ds.pixel_array.astype('float32')
        # Let's normalize the image between 0 and 1
        image -= np.min(image)
        image /= np.max(image)

        # Initialize the contour
        '''
        s = np.linspace(0, 2 * np.pi, 400)
        r = rows / 2 + rows / 4 * np.sin(s)
        c = columns / 2 + columns / 4 * np.cos(s)
        init = np.array([r, c]).T
        
        # Define the coordinates of the rectangle
        rr, cc = rectangle_perimeter((rectangle_position["ymin"] - 3, rectangle_position["xmin"] - 3),
                                     end=(rectangle_position["ymax"] + 3, rectangle_position["xmax"] + 3),
                                     shape=image.shape)  # (row, column)
        # Initialize the snake with the rectangle coordinates
        init = np.array([rr, cc]).T

        # We execute the GVF Snake algorithm
        # snake = active_contour(image, init, alpha=0.015, beta=10, gamma=0.001), preserve_range=False
        snake = active_contour(gaussian(image, 3), init, alpha=0.01, beta=3, gamma=0.001)
        # We draw the final contour
        final_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        init_array = []
        for point in init.astype(int):
            cv2.circle(final_image, (point[1], point[0]), 1, (255, 6, 0), -1)
            init_array.append([point[1], point[0]])
        init_points = np.array(init_array)

        snake_array = []
        for point in snake.astype(int):
            cv2.circle(final_image, (point[1], point[0]), 1, (0, 255, 0), -1)
            snake_array.append([point[1], point[0]])
        snake_points = np.array(snake_array)

        return final_image, snake_points, init_points
        '''
        # A logika jelenleg pass-ra van állítva vagy kommentezve van az eredeti fájlban.
        pass

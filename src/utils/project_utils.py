# src/utils/project_utils.py
import numpy as np
import cv2
import random
from skimage.segmentation import active_contour
from skimage.filters import gaussian
from skimage.draw import rectangle_perimeter

def class_colors(num_classes):
    """
    Segédfüggvény: Véletlenszerű színeket generál a visualizációhoz.
    """
    colors = []
    for _ in range(num_classes):
        colors.append((random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
    # Ha nincs osztály, adjunk vissza legalább egy színt
    if not colors:
        return [(0, 255, 0)]
    return colors

# --- A TE METÓDUSAID VÁLTOZATLANUL ---

def gvf_snake(image, rectangle_position):
    """
    Ez az algoritmus határozza meg a daganat ROI-ját a területen.
    :param image: Szürkeárnyalatos opencv kép numpy tömb alakja.
    :param rectangle_position:  Regions of Interest(ROI) pozicó
    :return: Illustrative image, Tumor points, ROI points
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
    '''
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

def roi2rect(img_name, img_np, img_data, label_list, image):
    """
    Prepare tumor ROI mask and rectangle bounding box.
    """
    # Másolat, hogy ne írjuk felül az eredeti képet
    img_vis = img_np.copy()

    colors = class_colors(len(label_list))

    # Ha nincs ROI → térjen vissza üres maskkal
    if img_data is None or len(img_data) == 0:
        empty_mask = np.zeros_like(image, dtype=np.uint8)
        return empty_mask, {}, None

    final_mask = None
    rectangle_position = None
    label = None

    for rect in img_data:

        # rect = [xmin, ymin, xmax, ymax, label1, label2, label3 ...]
        bounding_box = rect[:4]
        xmin, ymin, xmax, ymax = map(int, bounding_box)

        pmin = (xmin, ymin)
        pmax = (xmax, ymax)

        width = xmax - xmin
        height = ymax - ymin

        rectangle_position = {
            "xmin": xmin,
            "xmax": xmax,
            "ymin": ymin,
            "ymax": ymax,
            "width": width,
            "height": height,
        }

        # ROI mask
        roi_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.rectangle(roi_mask, pmin, pmax, 255, cv2.FILLED)

        # A nagy maszk az összes ROI jelöléssel
        if final_mask is None:
            final_mask = np.zeros_like(image, dtype=np.uint8)

        # label_array = [0,1,0,0] stb.
        label_array = np.array(rect[4:], dtype=float)

        # Hol van a "1" érték?
        indices = np.where(label_array == 1)[0]

        if len(indices) == 0:
            print(f"[WARN] Nincs címke a ROI-ban ({img_name}). label_array={label_array}")
            index = 0
        else:
            index = int(indices[0])

        # Label kiválasztása biztosan valid indexszel
        if index >= len(label_list):
            print(f"[WARN] Label index túl nagy: index={index}, label_list_len={len(label_list)}")
            index = 0

        label = label_list[index]
        '''
        if label == 'A':
            label = 'Adenocarcinoma'
        elif label == 'B':
            label = 'Small Cell Carcinoma'
        elif label == 'D':
            label = 'Large Cell Carcinoma'
        elif label == 'G':
            label = 'Squamous Cell Carcinoma'
        '''
        # Szín kiválasztása
        color = colors[index]

        # Bounding box vizualizáció
        cv2.rectangle(img_vis, pmin, pmax, color, 1)

        # Tumor mask létrehozása
        mask_tmp = np.zeros_like(image, dtype=np.uint8)
        mask_tmp = cv2.bitwise_and(image, image, mask=roi_mask)

        # Final mask-ba rakjuk
        final_mask = final_mask + mask_tmp

    # Bináris maszk
    _, segmented_image = cv2.threshold(final_mask, 127, 255, cv2.THRESH_BINARY)

    return segmented_image, rectangle_position, label
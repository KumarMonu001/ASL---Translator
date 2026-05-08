import random
import cv2
import numpy as np
from PIL import Image

class RobustTransform:

    def __init__(self):
        pass

    def apply_grayscale(self, image):
        gray = cv2.cvtColor(
            image, 
            cv2.COLOR_RGB2GRAY
        )
        gray_rgb = cv2.cvtColor(
            gray, 
            cv2.COLOR_GRAY2RGB
        )
        return gray_rgb

    def apply_threshold(self, image):
        gray = cv2.cvtColor(
            image, 
            cv2.COLOR_RGB2GRAY
        )
        blur = cv2.GaussianBlur(
            gray, 
            (5, 5), 
            0
        )
        
        # Otsu's method dynamically calculates the ideal threshold limit
        _, thresh = cv2.threshold(
            blur, 
            0, 
            255, 
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        
        thresh_rgb = cv2.cvtColor(
            thresh, 
            cv2.COLOR_GRAY2RGB
        )
        return thresh_rgb

    def apply_contrast(self, image):
        # Constrained multipliers to prevent pure white/black pixel clipping
        alpha = random.uniform(0.8, 1.3)
        beta = random.randint(-15, 15)
        
        contrast = cv2.convertScaleAbs(
            image, 
            alpha=alpha, 
            beta=beta
        )
        return contrast

    def apply_blur(self, image):
        blur = cv2.GaussianBlur(
            image, 
            (5, 5), 
            0
        )
        return blur

    def __call__(self, img):
        # ==================================
        # PIL -> NUMPY
        # ==================================
        image = np.array(img)

        # ==================================
        # RANDOM REPRESENTATION
        # ==================================
        mode = random.choice([
            "rgb",
            "grayscale",
            "threshold",
            "contrast",
            "blur"
        ])

        # ==================================
        # APPLY TRANSFORMATION
        # ==================================
        if mode == "rgb":
            transformed = image
            
        elif mode == "grayscale":
            transformed = self.apply_grayscale(image)
            
        elif mode == "threshold":
            transformed = self.apply_threshold(image)
            
        elif mode == "contrast":
            transformed = self.apply_contrast(image)
            
        elif mode == "blur":
            transformed = self.apply_blur(image)

        # ==================================
        # NUMPY -> PIL
        # ==================================
        return Image.fromarray(transformed)
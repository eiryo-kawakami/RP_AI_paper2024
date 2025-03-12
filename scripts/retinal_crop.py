from albumentations import Compose
from albumentations import CenterCrop, Crop
# Internal libraries/scripts
from aucmedi.data_processing.subfunctions.sf_base import Subfunction_Base

#-----------------------------------------------------#
#           Subfunction class: Retinal Crop           #
#-----------------------------------------------------#
""" Retinal cropping function for the specific microscopes utilized in the RIADD challenge.

Methods:
    __init__                Object creation function
    transform:              Crop retinal image.
"""
class Retinal_Crop(Subfunction_Base):
    #---------------------------------------------#
    #                Initialization               #
    #---------------------------------------------#
    def __init__(self):
        # Initialize center cropping transform for each microscope
        self.cropper_alpha = Compose([CenterCrop(width=1106,
                                                 height=1106,
                                                 p=1.0, always_apply=True)])
        self.cropper_beta = Compose([CenterCrop(width=1674,
                                                height=1674,
                                                p=1.0, always_apply=True)])
        self.cropper_gamma = Compose([CenterCrop(width=2632,
                                                height=2632,
                                                p=1.0, always_apply=True)])
        self.cropper_delta = Compose([CenterCrop(width=1120,
                                                height=1120,
                                                p=1.0, always_apply=True)])        

    #---------------------------------------------#
    #                Transformation               #
    #---------------------------------------------#
    def transform(self, image):
        # Microscope: TOPCON 3D OCT-2000
        if image.shape[1] == 1360:
            image_cropped = self.cropper_alpha(image=image)["image"]
        # TOPCON TRC-NW300
        elif image.shape[1] ==2000:
            image_cropped = self.cropper_beta(image=image)["image"]
        # Microscope: Kowa VX – 10α
        elif image.shape[1] ==3696:
            image_cropped = self.cropper_gamma(image=image)["image"]
        elif image.shape[1] ==1200:
            image_cropped = self.cropper_delta(image=image)["image"]
        # Return resized image
        return image_cropped

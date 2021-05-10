import os
import nibabel as nib
from monai.transforms import *
from DataLoader.CustomTransformation import *


class LoadData(object):

    def __call__(self, file_dir):

        images = []
        files_list = [os.path.join(file_dir, dI) for dI in os.listdir(file_dir)]
        for file in files_list:
            if 'seg' in file:
                seg = nib.load(file).get_fdata()
            else:
                image = (nib.load(file).get_fdata())
                images.append(image)
        images = np.stack(images, axis=0)
        seg = np.expand_dims(seg, axis=0).astype('float32')
        seg = np.where(seg == 4, 3, seg)
        return {"image": images, "seg": seg}


train_transforms = Compose([
    LoadData(),
    CenterSpatialCropd(keys=["image", "seg"], roi_size=[128, 128, 128]),
    # RandSpatialCropd(keys=["image", "seg"], roi_size=[128, 128, 128], random_size=False),
    NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
    RandFlipd(keys=["image", "seg"], prob=0.2),
    # Rotate3D(keys=["image", "seg"], prob=1, mag=0),
    # RandGaussianNoise3D(keys=["image"], prob=0, mag=0.1),
    # TranslateXYZ(keys=["image", "seg"], prob=0, mag=1),
    # Scale3D(keys=["image", "seg"], prob=1, mag=1),
    ToTensord(keys=["image", "seg"])
])

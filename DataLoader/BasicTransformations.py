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


class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
    """
    Convert labels to multi channels based on brats classes:
    label 1 is the necrotic and non-enhancing tumor core
    label 2 is the peritumoral edema
    label 4 is the GD-enhancing tumor
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).

    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            # merge label 1 and label 4 to construct TC
            result.append(np.logical_or(d[key] == 1, d[key] == 4))
            # merge labels 1, 2 and 4 to construct WT
            result.append(
                np.logical_or(
                    np.logical_or(d[key] == 2, d[key] == 4), d[key] == 1
                )
            )
            # label 4 is ET
            result.append(d[key] == 4)
            d[key] = np.stack(result, axis=0).astype(np.float32)
        return d


train_transforms = Compose([
    # LoadData(),
    LoadImaged(keys=["image", "seg"]),
    ConvertToMultiChannelBasedOnBratsClassesd(keys="seg"),
    Orientationd(keys=["image", "seg"], axcodes="RPS"),
    RandSpatialCropd(keys=["image", "seg"], roi_size=[128, 128, 128], random_size=False),
    NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
    RandFlipd(keys=["image", "seg"], prob=0.5),
    # Rotate3D(keys=["image", "seg"], prob=1, mag=0),
    # RandGaussianNoise3D(keys=["image"], prob=0, mag=0.1),
    # TranslateXYZ(keys=["image", "seg"], prob=0, mag=1),
    # Scale3D(keys=["image", "seg"], prob=1, mag=1),
    RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
    RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
    ToTensord(keys=["image", "seg"])
])

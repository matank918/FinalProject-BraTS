from monai.transforms import LoadImaged, CenterSpatialCropd, Compose, MapTransform, RandShiftIntensityd, \
    NormalizeIntensityd, \
    RandFlipd, RandScaleIntensityd, Orientationd, ToTensord, Spacingd, RandSpatialCropd, AsChannelFirstd
import numpy as np


class ConvertToMultiChannelBasedOnBratsClassesd2018(MapTransform):
    """
    Convert labels to multi channels based on brats classes:
    label 1 is the peritumoral edema
    label 2 is the GD-enhancing tumor
    label 3 is the necrotic and non-enhancing tumor core
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).

    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            # merge label 2 and label 3 to construct TC
            result.append(np.logical_or(d[key] == 2, d[key] == 3))
            # merge labels 1, 2 and 3 to construct WT
            result.append(
                np.logical_or(
                    np.logical_or(d[key] == 2, d[key] == 3), d[key] == 1
                )
            )
            # label 2 is ET
            result.append(d[key] == 2)
            d[key] = np.stack(result, axis=0).astype(np.float32)
        return d


class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
    """
    Convert labels to multi channels based on brats classes:
    label 1 is the necrotic and non-enhancing tumor core (NCR/NET)
    label 2 is the peritumoral edema (ED)
    label 3 is the non-enhancing part of the tumor combined with label 1
    label 4 is the GD-enhancing tumor (ET)
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).

    """

    def __call__(self, data):
        d = dict(data)
        result = []
        for key in self.keys:
            # merge labels 1, 2 and 4 to construct WT
            result.append(
                np.logical_or(
                    np.logical_or(d[key] == 2, d[key] == 4), d[key] == 1
                )
            )
            # label 4 is ET
            result.append(d[key] == 4)

            # merge label 1 and label 4 to construct TC
            result.append(np.logical_or(d[key] == 1, d[key] == 4))

            d[key] = np.stack(result, axis=0).astype(np.float32)
        return d


train_transforms = Compose([
    LoadImaged(keys=["image", "seg"]),
    ConvertToMultiChannelBasedOnBratsClassesd(keys="seg"),
    Orientationd(keys=["image", "seg"], axcodes="RAS"),
    RandSpatialCropd(keys=["image", "seg"], roi_size=[128, 128, 128], random_size=False),
    RandFlipd(keys=["image", "seg"], prob=0.5, spatial_axis=0),
    NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
    RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
    RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
    ToTensord(keys=["image", "seg"])
])

val_transform = Compose(
    [
        LoadImaged(keys=["image", "seg"]),
        ConvertToMultiChannelBasedOnBratsClassesd(keys="seg"),
        Orientationd(keys=["image", "seg"], axcodes="RAS"),
        CenterSpatialCropd(keys=["image", "seg"], roi_size=[128, 128, 128]),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ToTensord(keys=["image", "seg"]),
    ]
)

train_transform2018 = Compose(
    [
        # load 4 Nifti images and stack them together
        LoadImaged(keys=["image", "label"]),
        AsChannelFirstd(keys="image"),
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.5, 1.5, 2.0),
            mode=("bilinear", "nearest"),
        ),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        RandSpatialCropd(
            keys=["image", "label"], roi_size=[128, 128, 64], random_size=False
        ),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
        ToTensord(keys=["image", "label"]),
    ]
)
val_transform2018 = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        AsChannelFirstd(keys="image"),
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.5, 1.5, 2.0),
            mode=("bilinear", "nearest"),
        ),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        CenterSpatialCropd(keys=["image", "label"], roi_size=[128, 128, 64]),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ToTensord(keys=["image", "label"]),
    ]
)

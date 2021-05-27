from monai.transforms import LoadImaged, CenterSpatialCropd, Compose, MapTransform, RandShiftIntensityd, \
    NormalizeIntensityd, \
    RandFlipd, RandScaleIntensityd, Orientationd, ToTensord, Spacingd, RandSpatialCropd, RandRotated, RandZoomd
import numpy as np
from math import pi


class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
    """
    Convert labels to multi channels based on brats classes:
    label 1 is the necrotic and non-enhancing tumor core (NCR/NET)
    label 2 is the peritumoral edema (ED)
    label 4 is the GD-enhancing tumor (ET)
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).

    """

    def __call__(self, data):
        d = dict(data)
        result = []
        for key in self.keys:
            # label zero for background
            result.append(d[key] == 0)
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
    LoadImaged(keys=["image", "seg"]),
    ConvertToMultiChannelBasedOnBratsClassesd(keys="seg"),
    Orientationd(keys=["image", "seg"], axcodes="RAS"),
    RandSpatialCropd(keys=["image", "seg"], roi_size=[128, 128, 128], random_size=False),
    RandFlipd(keys=["image", "seg"], prob=0.5, spatial_axis=(0, 1, 2)),
    NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
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

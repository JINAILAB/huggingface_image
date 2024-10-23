import torch
from torchvision.transforms.functional import InterpolationMode
import datasets
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
    RandomRotation
)

def get_module(use_v2):
    # We need a protected import to avoid the V2 warning in case just V1 is used
    if use_v2:
        import torchvision.transforms.v2

        return torchvision.transforms.v2
    else:
        import torchvision.transforms

        return torchvision.transforms

class ClassificationPresetTrain:
    # Note: this transform assumes that the input to forward() are always PIL
    # images, regardless of the backend parameter. We may change that in the
    # future though, if we change the output type from the dataset.
    def __init__(
        self,
        *,
        crop_size,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        interpolation=InterpolationMode.BILINEAR,
        hflip_prob=0.5,
        auto_augment_policy=None,
        ra_magnitude=9,
        augmix_severity=3,
        random_erase_prob=0.1,
        use_v2=False,
    ): 
        T = get_module(use_v2)

        transforms = []

        transforms.append(T.PILToTensor())


        transforms.append(T.RandomResizedCrop(crop_size, interpolation=interpolation, scale=(0.7, 1.0), ratio=(0.8, 1.2), antialias=True))
        if hflip_prob > 0:
            transforms.append(T.RandomHorizontalFlip(hflip_prob))
            transforms.append(T.RandomRotation(90))
        if auto_augment_policy is not None:
            if auto_augment_policy == "ra":
                transforms.append(T.RandAugment(interpolation=interpolation, magnitude=ra_magnitude))
            elif auto_augment_policy == "ta_wide":
                transforms.append(T.TrivialAugmentWide(interpolation=interpolation))
            elif auto_augment_policy == "augmix":
                transforms.append(T.AugMix(interpolation=interpolation, severity=augmix_severity))
            else:
                aa_policy = T.AutoAugmentPolicy(auto_augment_policy)
                transforms.append(T.AutoAugment(policy=aa_policy, interpolation=interpolation))


        transforms.extend(
            [
                T.ToDtype(torch.float, scale=True) if use_v2 else T.ConvertImageDtype(torch.float),
                T.Normalize(mean=mean, std=std),
            ]
        )
        if random_erase_prob > 0:
            transforms.append(T.RandomErasing(p=0.1))

        if use_v2:
            transforms.append(T.ToPureTensor())

        self.transforms = T.Compose(transforms)

    def __call__(self, img):
        return self.transforms(img)
    
    
class ClassificationPresetEval:
    def __init__(
        self,
        *,
        crop_size,
        resize_size=256,
        mean=(0.485, 0.456, 0.406), 
        std=(0.229, 0.224, 0.225),
        interpolation=InterpolationMode.BILINEAR,
        use_v2=False,
    ):
        T = get_module(use_v2)
        transforms = []
        transforms.append(T.PILToTensor())


        transforms += [
            T.Resize(resize_size, interpolation=interpolation, antialias=True),
            T.CenterCrop(crop_size),
        ]

        transforms += [
            T.ToDtype(torch.float, scale=True) if use_v2 else T.ConvertImageDtype(torch.float),
            T.Normalize(mean=mean, std=std),
        ]

        if use_v2:
            transforms.append(T.ToPureTensor())

        self.transforms = T.Compose(transforms)

    def __call__(self, img):
        return self.transforms(img)
    



    
def preprocess_train(batch, train_transforms):
    batch["pixel_values"] = [
        train_transforms(image.convert("RGB")) for image in batch["image"]
    ]
    return batch


def preprocess_valid(batch, valid_transforms):
    """Apply train_transforms across a batch."""
    if "image" in batch:
        batch["pixel_values"] = [
            valid_transforms(image.convert("RGB")) for image in batch["image"]
    ]
    return batch



def load_dataset(dataset, args):
    val_resize_size, val_crop_size, train_crop_size = (
        args.val_resize_size,
        args.val_crop_size,
        args.train_crop_size,
    )
    
    interpolation = InterpolationMode(args.interpolation)
    

    auto_augment_policy = getattr(args, "auto_augment", None)
    random_erase_prob = getattr(args, "random_erase", 0.0)
    ra_magnitude = getattr(args, "ra_magnitude", None)
    augmix_severity = getattr(args, "augmix_severity", None)
    
    train_transform = ClassificationPresetTrain(
                crop_size=train_crop_size,
                interpolation=interpolation,
                auto_augment_policy=auto_augment_policy,
                random_erase_prob=random_erase_prob,
                ra_magnitude=ra_magnitude,
                augmix_severity=augmix_severity,
                use_v2=args.use_v2,
            )
    
    valid_transform = ClassificationPresetEval(
                crop_size=val_crop_size,
                resize_size=val_resize_size,
                interpolation=interpolation,
                use_v2=args.use_v2,
            )
    
    
    train_ds = dataset['train']
    val_ds = dataset['valid']
    
    
    train_ds.set_transform(lambda batch: preprocess_train(batch, train_transform))
    val_ds.set_transform(lambda batch: preprocess_valid(batch, valid_transform))

    return train_ds, val_ds, train_transform, valid_transform


def load_test_dataset(dataset, args):
    val_resize_size, val_crop_size, train_crop_size = (
        args.val_resize_size,
        args.val_crop_size,
        args.train_crop_size,
    )
    
    interpolation = InterpolationMode(args.interpolation)
    
    valid_transform = ClassificationPresetEval(
                crop_size=val_crop_size,
                resize_size=val_resize_size,
                interpolation=interpolation,
                use_v2=args.use_v2,
            )
    
    val_ds = dataset['test']
    
    val_ds.set_transform(lambda batch: preprocess_valid(batch, valid_transform))

    return val_ds, valid_transform
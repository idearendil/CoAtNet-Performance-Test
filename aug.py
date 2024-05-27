import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch

def hsv2colorjitter(h, s, v):
    """Map HSV (hue, saturation, value) jitter into ColorJitter values (brightness, contrast, saturation, hue)"""
    return v, v, s, h

# config : Changed setting for the xlarge_0325 experiments (1) 
def classify_albumentations(
        augment=True,
        size=224,
        # scale=(0.08, 1.0),
        # p_scale=1.0,
        crop_rate = (0.3 / 1.4),
        rotate=10,
        p_rotate=0.5,
        shear=10,
        p_shear=0.5,
        perspective=0.1,
        p_perspective=0.5,
        hsv_h=0.05,  # image HSV-Hue augmentation (fraction)
        hsv_s=0.7,  # image HSV-Saturation augmentation (fraction)
        hsv_v=0.6,  # image HSV-Value augmentation (fraction)
        p_hsv=0.5,
        motion_blur_limit=[7, 51],  # motion blur
        p_motion_blur=0.5,
        gaussian_blur_limit=[7, 55],  # gaussian blur
        p_gaussian_blur=0.5,
        gaussian_noise_var_limit=[10.0, 90.0],  # gaussian noise
        p_gaussian_noise=0.5,
        # mean=(0.0, 0.0, 0.0),  # IMAGENET_MEAN -> set as default Normalize() function's mean
        # std=(1.0, 1.0, 1.0),  # IMAGENET_STD -> set as default Normalize() function's std
):
    """YOLOv8 classification Albumentations (optional, only used if package is installed)."""
    try:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        # from gg.data.augmentations import DefaultScaleRandomResizedCrop

        if augment:  # Resize and crop
            T = []
            T += [A.RandomCropFromBorders(crop_left=crop_rate, crop_right=crop_rate, crop_top=crop_rate, crop_bottom=crop_rate)]
            T += [A.LongestMaxSize(max_size=size)]
            # if len(scale) == 2:
            #     T += [A.RandomResizedCrop(height=size, width=size, scale=scale, p=p_scale)]
            # elif len(scale) == 3:
            #     T += [DefaultScaleRandomResizedCrop(height=size, width=size, default_scale=scale[0], scale=scale[1:], p=p_scale)]
            # else:
            #     raise TypeError(f'classify_albumentations() scale {scale} must be [min, max] or [default, min, max]')

            
            if gaussian_noise_var_limit and p_gaussian_noise > 0:
                assert len(gaussian_noise_var_limit) == 2, 'gaussian_noise_var_limit must be [min, max] in either tuple or list'
                T += [A.GaussNoise(var_limit=gaussian_noise_var_limit, p=p_gaussian_noise)]

            if any((hsv_h, hsv_s, hsv_v)) and p_hsv > 0:
                T += [A.ColorJitter(*hsv2colorjitter(hsv_h, hsv_s, hsv_v), p=p_hsv)]  # brightness, contrast, saturation, hue
            if rotate > 0:
                T += [A.Rotate(limit=rotate, p=p_rotate)]
            if shear > 0:
                T += [A.Affine(shear=(-shear, shear), p=p_shear, mode=cv2.BORDER_REFLECT)]
            if perspective > 0:
                T += [A.Perspective(scale=perspective, p=p_perspective)]

            if gaussian_blur_limit and p_gaussian_blur > 0:
                assert len(gaussian_blur_limit) == 2, 'gaussian_blur_limit must be [min, max] in either tuple or list'
                T += [A.GaussianBlur(blur_limit=gaussian_blur_limit, p=p_gaussian_blur)]

            if motion_blur_limit and p_motion_blur > 0:
                assert len(motion_blur_limit) == 2, 'motion_blur_limit must be [min, max] in either tuple or list'
                T += [A.MotionBlur(blur_limit=motion_blur_limit, p=p_motion_blur)]

        else:  # Use fixed crop for eval set (reproducibility)
            T = [A.LongestMaxSize(max_size=size)]
        T += [A.Normalize()]  # Normalize and convert to Tensor
        T += [A.PadIfNeeded(min_height=size, min_width=size, position=A.PadIfNeeded.PositionType.TOP_LEFT, border_mode=cv2.BORDER_CONSTANT, value=0)]
        T += [ToTensorV2()]
        return A.Compose(T)

    except ImportError:  # package not installed, skip
        print('package not installed!!!')
        pass


def denormalize_img(img_tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0):
    if isinstance(img_tensor, torch.Tensor):
        mean = torch.tensor(mean).view(3, 1, 1)
        std = torch.tensor(std).view(3, 1, 1)

        img = img_tensor.cpu().clone()  # 이미지 텐서를 복제하여 원본 텐서 보존
        img = img * std * max_pixel_value + mean * max_pixel_value
        img = torch.clamp(img, 0, max_pixel_value)  # 픽셀 값을 0에서 max_pixel_value 사이로 제한

        return img.byte()  # torch.ByteTensor로 변환하여 반환
    else:
        raise ValueError("Input image must be a torch.Tensor.")



# def denormalize_img(img_tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0):
#         if isinstance(img_tensor, torch.Tensor):
#             img = img_tensor.clone().detach().cpu().numpy()
#             img = img.astype(np.float32)

#             mean = np.array(mean)[:, None, None]
#             std = np.array(std)[:, None, None]

#             img = (img * (np.array(std) * max_pixel_value)) + (np.array(mean) * max_pixel_value)
#             img = np.clip(img, 0, 255)

#             # numpy 
#             img = np.transpose(img, (1, 2, 0))
#             img = img.astype(np.uint8)
#             # img = img[:, :, ::-1]
#             return img
#         else:
#             raise ValueError("Input image must be a torch.Tensor.")

# 이미지 로드
# image_path = "/usr/src/data/20240319_tsinghua/test/others/539_cropped_47768_il50.jpg"
# image = Image.open(image_path)

# # # augmentation 적용
# transform = classify_albumentations(augment=True)

# # # 이미지 크기 출력
# # print("변환 후 이미지 크기:", transformed_image.shape)

# transformed_image = transform(image=np.array(image))["image"]

# img = transformed_image.clone().detach().cpu().numpy()
# img = img.astype(np.float32)

# mean=(0.485, 0.456, 0.406)
# std=(0.229, 0.224, 0.225)
# max_pixel_value=255.0

# mean = np.array(mean)[:, None, None]
# std = np.array(std)[:, None, None]

# img = (img * (np.array(std) * max_pixel_value)) + (np.array(mean) * max_pixel_value)
# img = np.clip(img, 0, 255)

# img = np.transpose(img, (1, 2, 0))
# img = img.astype(np.uint8)


# # # 넘파이 배열을 PIL 이미지로 변환
# image_pil = Image.fromarray(img)

# # # 결과 이미지 출력
# plt.imshow(image_pil)
# plt.axis('off')
# plt.show()
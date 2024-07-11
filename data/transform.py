import numbers
import torch
import random
import warnings

import torch
import torchvision
from torchvision.transforms import (
    RandomCrop,
    RandomResizedCrop,
)

from PIL import Image, ImageFilter
from typing import List

def _is_tensor_video_clip(clip):
    if not torch.is_tensor(clip):
        raise TypeError("clip should be Tensor. Got %s" % type(clip))

    if not clip.ndimension() == 4:
        raise ValueError("clip should be 4D. Got %dD" % clip.dim())

    return True


def crop(clip, i, j, h, w):
    """
    Args:
        clip (torch.tensor): Video clip to be cropped. Size is (C, T, H, W)
    """
    assert len(clip.size()) == 4, "clip should be a 4D tensor"
    return clip[..., i:i + h, j:j + w]


def resize(clip, target_size, interpolation_mode):
    assert len(target_size) == 2, "target size should be tuple (height, width)"
    return torch.nn.functional.interpolate(
        clip, size=target_size, mode=interpolation_mode, align_corners=False
    )


def resized_crop(clip, i, j, h, w, size, interpolation_mode="bilinear"):
    """
    Do spatial cropping and resizing to the video clip
    Args:
        clip (torch.tensor): Video clip to be cropped. Size is (C, T, H, W)
        i (int): i in (i,j) i.e coordinates of the upper left corner.
        j (int): j in (i,j) i.e coordinates of the upper left corner.
        h (int): Height of the cropped region.
        w (int): Width of the cropped region.
        size (tuple(int, int)): height and width of resized clip
    Returns:
        clip (torch.tensor): Resized and cropped clip. Size is (C, T, H, W)
    """
    assert _is_tensor_video_clip(clip), "clip should be a 4D torch.tensor"
    clip = crop(clip, i, j, h, w)
    clip = resize(clip, size, interpolation_mode)
    return clip


def center_crop(clip, crop_size):
    assert _is_tensor_video_clip(clip), "clip should be a 4D torch.tensor"
    h, w = clip.size(-2), clip.size(-1)
    th, tw = crop_size
    assert h >= th and w >= tw, "height and width must be no smaller than crop_size"

    i = int(round((h - th) / 2.0))
    j = int(round((w - tw) / 2.0))
    return crop(clip, i, j, th, tw)


def to_tensor(clip):
    """
    Convert tensor data type from uint8 to float, divide value by 255.0 and
    permute the dimensions of clip tensor
    Args:
        clip (torch.tensor, dtype=torch.uint8): Size is (T, H, W, C)
    Return:
        clip (torch.tensor, dtype=torch.float): Size is (C, T, H, W)
    """
    clip = torch.from_numpy(clip).contiguous()
    _is_tensor_video_clip(clip)
    if not clip.dtype == torch.uint8:
        raise TypeError("clip tensor should have data type torch.uint8. Got %s" % str(clip.dtype))
    return clip.float().permute(3, 0, 1, 2) / 127.5 - 1


def normalize(clip, mean, std, inplace=False):
    """
    Args:
        clip (torch.tensor): Video clip to be normalized. Size is (C, T, H, W)
        mean (tuple): pixel RGB mean. Size is (3)
        std (tuple): pixel standard deviation. Size is (3)
    Returns:
        normalized clip (torch.tensor): Size is (C, T, H, W)
    """
    assert _is_tensor_video_clip(clip), "clip should be a 4D torch.tensor"
    if not inplace:
        clip = clip.clone()
    mean = torch.as_tensor(mean, dtype=clip.dtype, device=clip.device)
    std = torch.as_tensor(std, dtype=clip.dtype, device=clip.device)
    clip.sub_(mean[:, None, None, None]).div_(std[:, None, None, None])
    return clip.permute(1, 0, 2, 3)


def hflip(clip):
    """
    Args:
        clip (torch.tensor): Video clip to be normalized. Size is (C, T, H, W)
    Returns:
        flipped clip (torch.tensor): Size is (C, T, H, W)
    """
    assert _is_tensor_video_clip(clip), "clip should be a 4D torch.tensor"
    return clip.flip((-1))

class CenterCropVideo(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, clip):
        return center_crop(clip, self.size)

class ResizedVideo(object):
    def __init__(self, size, interpolation_mode="bilinear"):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

        self.interpolation_mode = interpolation_mode

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): Video clip to be resized. Size is (C, T, H, W)
        Returns:
            torch.tensor: randomly resized video clip.
                size is (C, T, OH, OW)
        """
        return resize(clip, self.size, self.interpolation_mode)


class RandomCropVideo(RandomCrop):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): Video clip to be cropped. Size is (C, T, H, W)
        Returns:
            torch.tensor: randomly cropped/resized video clip.
                size is (C, T, OH, OW)
        """
        i, j, h, w = self.get_params(clip, self.size)
        return crop(clip, i, j, h, w)


class RandomResizedCropVideo(RandomResizedCrop):
    def __init__(
        self,
        size,
        scale=(0.08, 1.0),
        ratio=(3.0 / 4.0, 4.0 / 3.0),
        interpolation_mode="bilinear",
    ):
        if isinstance(size, tuple):
            assert len(size) == 2, "size should be tuple (height, width)"
            self.size = size
        else:
            self.size = (size, size)

        self.interpolation_mode = interpolation_mode
        self.scale = scale
        self.ratio = ratio

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): Video clip to be cropped. Size is (C, T, H, W)
        Returns:
            torch.tensor: randomly cropped/resized video clip.
                size is (C, T, H, W)
        """
        i, j, h, w = self.get_params(clip, self.scale, self.ratio)
        return resized_crop(clip, i, j, h, w, self.size, self.interpolation_mode)


class NormalizeVideo(object):
    """
    Normalize the video clip by mean subtraction and division by standard deviation
    Args:
        mean (3-tuple): pixel RGB mean
        std (3-tuple): pixel RGB standard deviation
        inplace (boolean): whether do in-place normalization
    """

    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): video clip to be normalized. Size is (C, T, H, W)
        """
        return normalize(clip, self.mean, self.std, self.inplace)


class ToTensorVideo(object):
    """
    Convert tensor data type from uint8 to float, divide value by 255.0 and
    permute the dimensions of clip tensor
    """

    def __init__(self):
        pass

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor, dtype=torch.uint8): Size is (T, H, W, C)
        Return:
            clip (torch.tensor, dtype=torch.float): Size is (C, T, H, W)
        """
        return to_tensor(clip)


class RandomHorizontalFlipVideo(object):
    """
    Flip the video clip along the horizonal direction with a given probability
    Args:
        p (float): probability of the clip being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): Size is (C, T, H, W)
        Return:
            clip (torch.tensor): Size is (C, T, H, W)
        """
        if random.random() < self.p:
            clip = hflip(clip)
        return clip


class ColorJitterVideo:
    """
    A custom sequence of transforms that randomly performs Color jitter,
    Gaussian Blur and Grayscaling on the given clip.
    Particularly useful for the SSL tasks like SimCLR, MoCoV2, BYOL, etc.
    Args:
        bri_con_sat (list[float]): A list of 3 floats reprsenting brightness,
        constrast and staturation coefficients to use for the
        `torchvision.transforms.ColorJitter` transform.
        hue (float): Heu value to use in the `torchvision.transforms.ColorJitter`
        transform.
        p_color_jitter (float): The probability with which the Color jitter transform
        is randomly applied on the given clip.
        p_convert_gray (float): The probability with which the given clip is randomly
        coverted into grayscale.
        p_gaussian_blur (float): The probability with which the Gaussian transform
        is randomly applied on the given clip.
        gaussian_blur_sigma (list[float]): A list of 2 floats with in which
        the blur radius is randomly sampled for Gaussian blur transform.
    """

    def __init__(
        self,
        bri_con_sat: List[float],
        hue: float,
        p_color_jitter: float = 0.0,
        p_convert_gray: float = 0.3,
        p_gaussian_blur: float = 0.5,
        gaussian_blur_sigma: List[float] = (0.1, 2.0),
    ) -> None:

        self.color_jitter = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.RandomApply(
                    [
                        torchvision.transforms.ColorJitter(
                            bri_con_sat[0], bri_con_sat[1], bri_con_sat[2], hue
                        )
                    ],
                    p=p_color_jitter,
                ),
                torchvision.transforms.RandomGrayscale(p=p_convert_gray),
                torchvision.transforms.RandomApply(
                    [GaussianBlur(gaussian_blur_sigma)], p=p_gaussian_blur
                ),
                torchvision.transforms.ToTensor(),
            ]
        )

    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frames (tensor): frames of images sampled from the video. The
            dimension is `channel` x `num frames` x `height` x `width`.
        Returns:
            frames (tensor): frames of images sampled from the video. The
            dimension is `channel` x `num frames` x `height` x `width`.
        """
        c, t, h, w = frames.shape
        frames = frames.view(c, t * h, w)
        frames = self.color_jitter(frames)  # pyre-ignore[6,9]
        frames = frames.view(c, t, h, w)

        return frames


class GaussianBlur(object):
    """
    A PIL image version of Gaussian blur augmentation as
    in SimCLR https://arxiv.org/abs/2002.05709
    Args:
        sigma (list[float]): A list of 2 floats with in which
        the blur radius is randomly sampled during each step.
    """

    def __init__(self, sigma: List[float] = (0.1, 2.0)) -> None:
        self.sigma = sigma

    def __call__(self, img: Image.Image) -> Image.Image:
        """
        img (Image): A PIL image with single or 3 color channels.
        """
        sigma = self.sigma[0]
        if len(self.sigma) == 2:
            sigma = random.uniform(self.sigma[0], self.sigma[1])

        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
        return img
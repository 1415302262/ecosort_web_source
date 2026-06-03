from typing import Tuple, Union

from PIL import Image
from torchvision.transforms import functional as F
from torchvision.transforms import InterpolationMode


ColorType = Union[int, Tuple[int, int, int]]


class ResizeLongestSideWithPadding:
    """等比例缩放到目标长边，并对短边居中补边到正方形。"""

    def __init__(
        self,
        size: int,
        fill: ColorType = (114, 114, 114),
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    ) -> None:
        self.size = int(size)
        self.fill = fill
        self.interpolation = interpolation

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        if width <= 0 or height <= 0:
            return image

        scale = self.size / float(max(width, height))
        resized_width = max(1, int(round(width * scale)))
        resized_height = max(1, int(round(height * scale)))

        image = F.resize(
            image,
            size=[resized_height, resized_width],
            interpolation=self.interpolation,
            antialias=True,
        )

        pad_w = self.size - resized_width
        pad_h = self.size - resized_height
        left = pad_w // 2
        right = pad_w - left
        top = pad_h // 2
        bottom = pad_h - top

        return F.pad(image, padding=[left, top, right, bottom], fill=self.fill)

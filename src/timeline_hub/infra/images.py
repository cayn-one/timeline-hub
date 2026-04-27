import math
from io import BytesIO
from typing import Literal

from PIL import Image, ImageFilter, ImageOps


def to_jpg(image_bytes: bytes, *, quality: int = 95) -> bytes:
    """Convert image bytes to JPEG bytes.

    Args:
        image_bytes: Source image bytes in a Pillow-readable format.
        quality: JPEG quality in the closed range 1..100.

    Raises:
        ValueError: If parameters are invalid.
    """
    _validate_image_bytes(image_bytes)
    _validate_quality(quality)

    with Image.open(BytesIO(image_bytes)) as image:
        image.load()
        image = ImageOps.exif_transpose(image)
        output_image = _normalize_to_rgb(image)

        return _save_jpg(output_image, quality=quality)


def normalize_cover_to_jpg(
    image_bytes: bytes,
    *,
    max_height: int = 1280,
    quality: int = 95,
) -> bytes:
    """Normalize cover image bytes to the stored JPEG invariant.

    Already-valid JPEG inputs may be returned unchanged when no resize,
    EXIF normalization, or RGB normalization is needed. In that case,
    `quality` is not applied because no re-encoding occurs.

    Args:
        image_bytes: Source image bytes in a Pillow-readable format.
        max_height: Maximum allowed output height in pixels.
        quality: JPEG quality in the closed range 1..100.

    Raises:
        ValueError: If parameters are invalid.
    """
    _validate_image_bytes(image_bytes)
    _validate_quality(quality)
    _validate_max_height(max_height)

    is_jpeg = image_bytes.startswith(b'\xff\xd8\xff')

    with Image.open(BytesIO(image_bytes)) as image:
        image.load()
        needs_exif_transpose = _needs_exif_transpose(image)
        image = ImageOps.exif_transpose(image)

        width, height = image.size
        needs_resize = height > max_height
        needs_rgb_normalization = _needs_rgb_normalization(image)

        if is_jpeg and not needs_exif_transpose and not needs_resize and not needs_rgb_normalization:
            return image_bytes

        output_image = _normalize_to_rgb(image)
        if needs_resize:
            new_width = max(1, round(width * max_height / height))
            output_image = output_image.resize((new_width, max_height), Image.Resampling.LANCZOS)

        return _save_jpg(output_image, quality=quality)


def pad_image_to_width_factor(
    image_bytes: bytes,
    *,
    width_factor: float = 2.0,
    background: Literal['white', 'black', 'blur'] = 'white',
    quality: int = 95,
) -> bytes:
    """Pad image bytes to a wider JPEG canvas while preserving source content.

    Common accepted source formats include JPEG, PNG, WebP, GIF, BMP, and
    TIFF, depending on Pillow support in the runtime environment. Output is
    always JPEG bytes.

    Args:
        image_bytes: Source image bytes in a Pillow-readable format.
        width_factor: Multiplier used to derive output width from source width.
        background: Background fill strategy for extra horizontal space.
        quality: JPEG quality in the closed range 1..100.

    Raises:
        ValueError: If parameters are invalid.
    """
    _validate_image_bytes(image_bytes)
    _validate_quality(quality)
    _validate_width_factor(width_factor)
    _validate_background(background)

    with Image.open(BytesIO(image_bytes)) as image:
        image.load()
        image = ImageOps.exif_transpose(image)
        source_image = _normalize_to_rgb(image)

        width, height = source_image.size
        target_width = max(width, round(width * width_factor))
        if background == 'white':
            output_image = Image.new('RGB', (target_width, height), 'white')
        elif background == 'black':
            output_image = Image.new('RGB', (target_width, height), 'black')
        else:
            background_scale = max(target_width / width, 1.0)
            background_width = max(1, round(width * background_scale))
            background_height = max(1, round(height * background_scale))
            output_image = source_image.resize((background_width, background_height), Image.Resampling.LANCZOS)
            crop_left = max(0, (background_width - target_width) // 2)
            crop_top = max(0, (background_height - height) // 2)
            output_image = output_image.crop((crop_left, crop_top, crop_left + target_width, crop_top + height))
            output_image = output_image.filter(ImageFilter.GaussianBlur(radius=32))
        offset_x = (target_width - width) // 2
        output_image.paste(source_image, (offset_x, 0))
        return _save_jpg(output_image, quality=quality)


def _validate_image_bytes(image_bytes: bytes) -> None:
    if not image_bytes:
        raise ValueError('image_bytes must not be empty')


def _validate_quality(quality: int) -> None:
    if isinstance(quality, bool) or not isinstance(quality, int):
        raise ValueError('quality must be an integer')
    if quality < 1 or quality > 100:
        raise ValueError('quality must be in 1..100')


def _validate_max_height(max_height: int) -> None:
    if isinstance(max_height, bool) or not isinstance(max_height, int):
        raise ValueError('max_height must be an integer')
    if max_height < 1:
        raise ValueError('max_height must be >= 1')


def _validate_width_factor(width_factor: float) -> None:
    if isinstance(width_factor, bool) or not isinstance(width_factor, int | float):
        raise ValueError('width_factor must be an int or float')
    if not math.isfinite(width_factor):
        raise ValueError('width_factor must be finite')
    if width_factor < 1.0:
        raise ValueError('width_factor must be >= 1.0')


def _validate_background(background: Literal['white', 'black', 'blur']) -> None:
    if not isinstance(background, str):
        raise ValueError("background must be one of 'white', 'black', or 'blur'")
    if background not in {'white', 'black', 'blur'}:
        raise ValueError("background must be one of 'white', 'black', or 'blur'")


def _needs_exif_transpose(image: Image.Image) -> bool:
    return image.getexif().get(274, 1) != 1


def _needs_rgb_normalization(image: Image.Image) -> bool:
    # Any non-RGB mode requires normalization to RGB before JPEG encoding.
    return image.mode != 'RGB'


def _normalize_to_rgb(image: Image.Image) -> Image.Image:
    if image.mode in {'RGBA', 'LA'} or (image.mode == 'P' and 'transparency' in image.info):
        rgba_image = image.convert('RGBA')
        flattened_image = Image.new('RGB', rgba_image.size, 'white')
        flattened_image.paste(rgba_image, mask=rgba_image.getchannel('A'))
        return flattened_image
    if image.mode != 'RGB':
        return image.convert('RGB')
    return image


def _save_jpg(image: Image.Image, *, quality: int) -> bytes:
    output = BytesIO()
    image.save(
        output,
        format='JPEG',
        quality=quality,
        subsampling=0,
        optimize=True,
        progressive=True,
    )
    return output.getvalue()

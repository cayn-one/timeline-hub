from io import BytesIO

import pytest
from PIL import Image

from timeline_hub.infra.images import normalize_cover_to_jpg, pad_image_to_width_factor, to_jpg


def test_to_jpg_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match='image_bytes must not be empty'):
        to_jpg(b'')


def test_to_jpg_rejects_invalid_quality() -> None:
    with pytest.raises(ValueError, match='quality must be in 1..100'):
        to_jpg(b'image-bytes', quality=0)


def test_to_jpg_converts_png_to_jpeg() -> None:
    png_bytes = _build_png_bytes(mode='RGB')

    jpg_bytes = to_jpg(png_bytes)

    assert jpg_bytes
    with Image.open(BytesIO(jpg_bytes)) as image:
        image.load()
        assert image.format == 'JPEG'


def test_to_jpg_handles_transparency() -> None:
    png_bytes = _build_png_bytes(mode='RGBA')

    jpg_bytes = to_jpg(png_bytes)

    assert jpg_bytes
    with Image.open(BytesIO(jpg_bytes)) as image:
        image.load()
        assert image.format == 'JPEG'
        assert image.mode == 'RGB'


def test_to_jpg_applies_exif_orientation() -> None:
    jpeg_bytes = _build_oriented_jpeg_bytes()

    converted_bytes = to_jpg(jpeg_bytes)

    with Image.open(BytesIO(converted_bytes)) as image:
        image.load()
        assert image.format == 'JPEG'
        assert image.size == (20, 40)
        top_pixel = image.getpixel((10, 10))
        bottom_pixel = image.getpixel((10, 30))
        assert top_pixel[0] > top_pixel[1]
        assert bottom_pixel[1] > bottom_pixel[0]


def test_normalize_cover_to_jpg_returns_original_bytes_for_small_jpeg() -> None:
    jpeg_bytes = _build_jpeg_bytes(size=(640, 640))

    normalized_bytes = normalize_cover_to_jpg(jpeg_bytes)

    assert normalized_bytes == jpeg_bytes


def test_normalize_cover_to_jpg_resizes_large_jpeg_without_upscaling_width_ratio() -> None:
    jpeg_bytes = _build_jpeg_bytes(size=(1500, 2000))

    normalized_bytes = normalize_cover_to_jpg(jpeg_bytes)

    assert normalized_bytes != jpeg_bytes
    assert normalized_bytes.startswith(b'\xff\xd8\xff')
    with Image.open(BytesIO(normalized_bytes)) as image:
        image.load()
        assert image.format == 'JPEG'
        assert image.size == (960, 1280)


def test_normalize_cover_to_jpg_converts_small_png_without_upscaling() -> None:
    png_bytes = _build_png_bytes(mode='RGB', size=(600, 800))

    normalized_bytes = normalize_cover_to_jpg(png_bytes)

    assert normalized_bytes.startswith(b'\xff\xd8\xff')
    with Image.open(BytesIO(normalized_bytes)) as image:
        image.load()
        assert image.format == 'JPEG'
        assert image.size == (600, 800)


def test_normalize_cover_to_jpg_converts_and_clips_tall_png() -> None:
    png_bytes = _build_png_bytes(mode='RGBA', size=(900, 1800))

    normalized_bytes = normalize_cover_to_jpg(png_bytes)

    assert normalized_bytes.startswith(b'\xff\xd8\xff')
    with Image.open(BytesIO(normalized_bytes)) as image:
        image.load()
        assert image.format == 'JPEG'
        assert image.mode == 'RGB'
        assert image.size == (640, 1280)


def test_pad_image_to_width_factor_default_white_doubles_width_for_square_rgb_image() -> None:
    jpeg_bytes = _build_jpeg_bytes(size=(8, 8))

    padded_bytes = pad_image_to_width_factor(jpeg_bytes, width_factor=2.0)

    assert padded_bytes.startswith(b'\xff\xd8\xff')
    with Image.open(BytesIO(padded_bytes)) as image:
        image.load()
        assert image.format == 'JPEG'
        assert image.mode == 'RGB'
        assert image.size == (16, 8)


def test_pad_image_to_width_factor_default_white_centers_content_and_uses_white_padding() -> None:
    source_image = Image.new('RGB', (4, 2), (10, 20, 30))
    output = BytesIO()
    source_image.save(output, format='JPEG', quality=95, subsampling=0, optimize=True, progressive=True)

    padded_bytes = pad_image_to_width_factor(output.getvalue(), width_factor=2.0)

    with Image.open(BytesIO(padded_bytes)) as image:
        image.load()
        assert image.size == (8, 2)
        left_padding_pixel = image.getpixel((0, 1))
        content_pixel = image.getpixel((3, 1))
        right_padding_pixel = image.getpixel((7, 1))

        assert all(channel >= 245 for channel in left_padding_pixel)
        assert all(channel >= 245 for channel in right_padding_pixel)
        assert abs(content_pixel[0] - 10) <= 10
        assert abs(content_pixel[1] - 20) <= 10
        assert abs(content_pixel[2] - 30) <= 10


def test_pad_image_to_width_factor_black_background_uses_black_side_padding() -> None:
    source_image = Image.new('RGB', (4, 2), (10, 20, 30))
    output = BytesIO()
    source_image.save(output, format='JPEG', quality=95, subsampling=0, optimize=True, progressive=True)

    padded_bytes = pad_image_to_width_factor(output.getvalue(), width_factor=2.0, background='black')

    with Image.open(BytesIO(padded_bytes)) as image:
        image.load()
        assert image.size == (8, 2)
        left_padding_pixel = image.getpixel((0, 1))
        content_pixel = image.getpixel((3, 1))
        right_padding_pixel = image.getpixel((7, 1))

        assert all(channel <= 10 for channel in left_padding_pixel)
        assert all(channel <= 10 for channel in right_padding_pixel)
        assert abs(content_pixel[0] - 10) <= 10
        assert abs(content_pixel[1] - 20) <= 10
        assert abs(content_pixel[2] - 30) <= 10


def test_pad_image_to_width_factor_blur_background_resizes_and_preserves_centered_foreground() -> None:
    source_image = Image.new('RGB', (8, 4))
    for x in range(8):
        for y in range(4):
            source_image.putpixel((x, y), (200, 30, 20) if x < 4 else (20, 40, 220))
    output = BytesIO()
    source_image.save(output, format='JPEG', quality=95, subsampling=0, optimize=True, progressive=True)

    padded_bytes = pad_image_to_width_factor(output.getvalue(), width_factor=2.0, background='blur')

    with Image.open(BytesIO(padded_bytes)) as image:
        image.load()
        assert image.format == 'JPEG'
        assert image.size == (16, 4)
        left_foreground_pixel = image.getpixel((4, 2))
        right_foreground_pixel = image.getpixel((11, 2))
        assert left_foreground_pixel[0] > left_foreground_pixel[2]
        assert right_foreground_pixel[2] > right_foreground_pixel[0]


def test_pad_image_to_width_factor_one_preserves_dimensions_and_returns_jpeg_bytes() -> None:
    jpeg_bytes = _build_jpeg_bytes(size=(9, 5))

    padded_bytes = pad_image_to_width_factor(jpeg_bytes, width_factor=1.0)

    assert padded_bytes.startswith(b'\xff\xd8\xff')
    with Image.open(BytesIO(padded_bytes)) as image:
        image.load()
        assert image.format == 'JPEG'
        assert image.size == (9, 5)


@pytest.mark.parametrize(
    ('width_factor', 'match'),
    [
        (True, 'width_factor must be an int or float'),
        ('2.0', 'width_factor must be an int or float'),
        (float('inf'), 'width_factor must be finite'),
        (float('nan'), 'width_factor must be finite'),
        (0.99, 'width_factor must be >= 1.0'),
    ],
)
def test_pad_image_to_width_factor_rejects_invalid_width_factor(width_factor: object, match: str) -> None:
    jpeg_bytes = _build_jpeg_bytes(size=(4, 4))

    with pytest.raises(ValueError, match=match):
        pad_image_to_width_factor(jpeg_bytes, width_factor=width_factor)


@pytest.mark.parametrize(
    ('background', 'match'),
    [
        (1, "background must be one of 'white', 'black', or 'blur'"),
        ('gradient', "background must be one of 'white', 'black', or 'blur'"),
    ],
)
def test_pad_image_to_width_factor_rejects_invalid_background(background: object, match: str) -> None:
    jpeg_bytes = _build_jpeg_bytes(size=(4, 4))

    with pytest.raises(ValueError, match=match):
        pad_image_to_width_factor(jpeg_bytes, background=background)


def _build_png_bytes(*, mode: str, size: tuple[int, int] = (4, 4)) -> bytes:
    color = (10, 20, 30) if mode == 'RGB' else (10, 20, 30, 0)
    image = Image.new(mode, size, color)
    output = BytesIO()
    image.save(output, format='PNG')
    return output.getvalue()


def _build_jpeg_bytes(*, size: tuple[int, int]) -> bytes:
    image = Image.new('RGB', size, (10, 20, 30))
    output = BytesIO()
    image.save(
        output,
        format='JPEG',
        quality=95,
        subsampling=0,
        optimize=True,
        progressive=True,
    )
    return output.getvalue()


def _build_oriented_jpeg_bytes() -> bytes:
    image = Image.new('RGB', (40, 20))
    for x in range(20):
        for y in range(20):
            image.putpixel((x, y), (255, 0, 0))
    for x in range(20, 40):
        for y in range(20):
            image.putpixel((x, y), (0, 255, 0))

    exif = image.getexif()
    exif[274] = 6

    output = BytesIO()
    image.save(output, format='JPEG', exif=exif)
    return output.getvalue()

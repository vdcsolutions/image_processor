import os
from PIL import Image
import pytest
from image_processor import ImageProcessor


@pytest.fixture(scope="module")
def image_processor():
    # Set up an instance of ImageProcessor for testing
    image_path = "test_image.jpg"
    # Create a test image
    with Image.new("RGB", (100, 100), color=(255, 0, 0)) as test_image:
        test_image.save(image_path)
    yield ImageProcessor(image_path)
    # Clean up the test image
    os.remove(image_path)


def test_grayscale(image_processor):
    # Test grayscale method
    grayscale_image = image_processor.grayscale()
    assert isinstance(grayscale_image, Image.Image)
    assert grayscale_image.mode == "L"


def test_normalize(image_processor):
    # Test normalize method
    normalized_image = image_processor.normalize()
    assert isinstance(normalized_image, Image.Image)


def test_brighten(image_processor):
    # Test brighten method
    brightened_image = image_processor.brighten(brightness=1.5)
    assert isinstance(brightened_image, Image.Image)


def test_darken(image_processor):
    # Test darken method
    darkened_image = image_processor.darken(darkness=0.5)
    assert isinstance(darkened_image, Image.Image)


def test_sharpen(image_processor):
    # Test sharpen method
    sharpened_image = image_processor.sharpen()
    assert isinstance(sharpened_image, Image.Image)


def test_resize_image(image_processor):
    # Test resize_image method
    resized_image = image_processor.resize_image(percent=0.5)
    assert isinstance(resized_image, Image.Image)
    assert resized_image.size == (50, 50)


def test_flip_image(image_processor):
    # Test flip_image method
    flipped_image = image_processor.flip_image()
    assert isinstance(flipped_image, Image.Image)


def test_rotate_image(image_processor):
    # Test rotate_image method
    rotated_image = image_processor.rotate_image(degrees=90)
    assert isinstance(rotated_image, Image.Image)


def test_add_gaussian_noise(image_processor):
    # Test add_gaussian_noise method
    noisy_image = image_processor.add_gaussian_noise()
    assert isinstance(noisy_image, Image.Image)

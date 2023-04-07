from PIL import Image, ImageFilter
import numpy as np
import os


class ImageProcessor:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image_dir, self.image_name = os.path.split(image_path)

    def grayscale(self):
        """
        Converts an image to grayscale.
        :return: Grayscale image as a PIL.Image object
        """
        with Image.open(self.image_path) as image:
            grayscale_image = image.convert('L')

        # Save the processed image to a file
        return grayscale_image

    def save_processed_image(self, processed_image, directory="processed_images", name=None):
        """
        Saves a processed image to a file in the specified directory with the specified name.

        :param processed_image: A PIL.Image object representing the processed image to be saved.
        :param directory: The directory where the processed image should be saved. Defaults to "processed_images".
        :param name: The name of the processed image file. If None, the name will be "{processing_type}_{image_name}.jpg".
        :return: The path to the saved image file.
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
        if name is None:
            name = f"{self.processing_type}_{self.image_name}"
        processed_image_path = os.path.join(directory, name)
        print(processed_image_path)
        processed_image.save(processed_image_path + '.jpg')
        return processed_image_path

    def normalize(self):
        """
        Normalizes an image.
        :return: Normalized image as a PIL.Image object
        """
        with Image.open(self.image_path) as image:
            # Convert image to numpy array
            image_array = np.array(image)
            # Normalize image
            normalized_image = Image.fromarray(
                np.uint8((image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array)) * 255))
        return normalized_image

    def brighten(self, brightness=1.5):
        """
        Increases the brightness of an image.
        :param brightness: Amount to increase brightness (default=1.5)
        :return: Brightened image as a PIL.Image object
        """
        with Image.open(self.image_path) as image:
            # Convert image to numpy array
            image_array = np.array(image)
            # Brighten image
            brightened_image = Image.fromarray(np.uint8(np.clip(image_array * brightness, 0, 255)))
        return brightened_image

    def darken(self, darkness=0.5):
        """
        Decreases the brightness of an image.
        :param darkness: Amount to decrease brightness (default=0.5)
        :return: Darkened image as a PIL.Image object
        """
        with Image.open(self.image_path) as image:
            # Convert image to numpy array
            image_array = np.array(image)
            # Darken image
            darkened_image = Image.fromarray(np.uint8(np.clip(image_array * darkness, 0, 255)))
        return darkened_image

    def sharpen(self):
        """
        Sharpens an image.
        :return: Sharpened image as a PIL.Image object
        """
        with Image.open(self.image_path) as image:
            sharpened_image = image.filter(ImageFilter.SHARPEN)
        return sharpened_image

    def resize_image(self, percent=10):
        """
        Resizes an image by a percentage value.
        :param percent: Percentage value for resizing
        :return: Resized image as a PIL.Image object
        """
        with Image.open(self.image_path) as image:
            # Get image size
            width, height = image.size
            # Calculate new size
            new_width = int(width * percent)
            new_height = int(height * percent)
            # Resize image
            resized_image = image.resize((new_width, new_height))
        return resized_image

    def flip_image(self):
        """
        Flips an image horizontally.
        :return: Flipped image as a PIL.Image object
        """
        with Image.open(self.image_path) as image:
            flipped_image = image.transpose(Image.FLIP_LEFT_RIGHT)
        return flipped_image

    def rotate_image(self, degrees=30):
        """
        Rotates an image by a specified number of degrees.
        :param degrees: Number of degrees to rotate the image
        :return: Rotated image as a PIL.Image object
        """
        with Image.open(self.image_path) as image:
            rotated_image = image.rotate(degrees)
        return rotated_image

    def add_gaussian_noise(self, mean=0, variance=0.1):
        """
        Adds Gaussian noise to an image.
        :param mean: Mean of the Gaussian distribution (default=0)
        :param variance: Variance of the Gaussian distribution (default=0.1)
        :return: Noisy image as a PIL.Image object
        """
        with Image.open(self.image_path) as image:
            # Convert image to numpy array
            image_array = np.array(image)
            # Generate Gaussian noise
            noise = np.random.normal(mean, variance, size=image_array.shape)
            # Add noise to image
            noisy_image = Image.fromarray(np.uint8(np.clip(image_array + noise, 0, 255)))
        return noisy_image

    def random_crop(self, crop_percent=[0.3, 0.3]):
        """
        Applies a random crop to an image.
        :param crop_percent: Tuple of the crop size as a percentage of the original size (width %, height %)
        :return: Cropped image as a PIL.Image object
        """
        with Image.open(self.image_path) as image:
            # Get image size
            width, height = image.size
            # Calculate crop size
            crop_width = round(width * crop_percent[0])
            crop_height = round(height * crop_percent[1])
            # Check if crop size is at least 1 pixel
            if crop_width < 1:
                crop_width = 1
            if crop_height < 1:
                crop_height = 1
            # Get random crop coordinates
            left = np.random.randint(0, width - crop_width)
            top = np.random.randint(0, height - crop_height)
            right = left + crop_width
            bottom = top + crop_height
            # Crop image
            cropped_image = image.crop((left, top, right, bottom))
        return cropped_image

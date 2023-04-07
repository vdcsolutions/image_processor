# image_processor
Process images in order to create better dataset for ML model


__init__(self, image_path): Constructor that initializes the image_path instance variable.
grayscale(self): Converts an image to grayscale.
save_processed_image(self, processed_image, directory="processed_images", name=None): Saves the processed image to a file in the specified directory. If no name is provided, the original image name is used with a prefix indicating the processing type (e.g. "grayscale_").
normalize(self): Normalizes an image by scaling pixel values to between 0 and 255.
brighten(self, brightness=1.5): Increases the brightness of an image by multiplying pixel values by a factor (default=1.5).
darken(self, darkness=0.5): Decreases the brightness of an image by multiplying pixel values by a factor (default=0.5).
sharpen(self): Sharpens an image.
resize_image(self, percent): Resizes an image by a percentage value.
flip_image(self): Flips an image horizontally.
rotate_image(self, degrees=30): Rotates an image by a specified number of degrees (default=30).
add_gaussian_noise(self, mean=0, variance=0.1): Adds Gaussian noise to an image.
random_crop(self, crop_percent=[0.3, 0.3]): Applies a random crop to an image based on the provided crop percentages.

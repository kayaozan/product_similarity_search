import numpy as np
from PIL import Image, ImageOps
from io import BytesIO
import requests


def is_url_image(url: str) -> bool:
    """Check if a URL is a valid image URL."""

    image_formats = ("image/png", "image/jpeg", "image/jpg")
    r = requests.head(url)

    if r.headers["content-type"] in image_formats:
        return True
    return False


def image_by_URL(image_url: str) -> Image.Image:
    """Open an image by a given URL."""

    # Fix spaces with percent-encoding (a common issue)
    url = image_url.replace(' ','%20')

    if is_url_image(url):
        req = requests.get(url, timeout=20)
        return Image.open(BytesIO(req.content)).convert('RGB')
    else:
        raise ValueError('URL does not contain an image.')


def get_non_white_bounding_box(image: Image.Image,
                               threshold=200) -> list:
    """
    Locate non-white pixels, return the bounding box.

    Although checking for non-white pixels is intended,
    it could be used for other purposes by changing threshold.

    threshold: the value to check for pixels (default 200)
    Returns bbox: [x_min, y_min, x_max, y_max]
    """

    # Locate all pixels below the threshold.
    # Use axis=2 to return an array of (Height, Width).
    non_whites = np.any(np.asarray(image) < threshold, axis=2)
    
    # Return None if all fails the condition (above threshold).
    if np.where(np.any(non_whites,axis=1))[0].size == 0:
        return None
    else:
        bbox = [0,0,0,0]
        # axis: loop through each axis
        for axis in [0,1]:
            non_whites_axis = np.where(np.any(non_whites,axis=axis))[0]
            # Save first and last (min and max) values.
            # Add 1 to max to encapsulate far edges.
            bbox[axis]   = non_whites_axis[0]
            bbox[axis+2] = non_whites_axis[-1] + 1
        return bbox


def pad_to_square(image: Image.Image,
                  padding_color = (0,0,0)) -> Image.Image:
    """
    Expand an image to square with padding.

    Pads the image with zeroes (black) by default.
    """

    # Pick the longer size of the image.
    longer_side = max(image.size[0], image.size[1])
    square_size = (longer_side, longer_side)

    image_padded = ImageOps.pad(image,
                                size=square_size,
                                color=padding_color)

    return image_padded


def concat_images_h(image_list: list[Image.Image]) -> Image.Image:
    """
    Concat a list of images horizontally.

    Note that it does not check if the height of images is equal.
    In the case of inequality, images might overflow or underflow.
    """

    # Create a blank image
    # which height is gathered by the first image
    # and width is the sum of all images'
    final_image = Image.new('RGB',
                       (sum([img.width for img in image_list]),
                        image_list[0].height))
    
    w = 0
    for image in image_list:
        # Paste the image and then move the position by its width
        final_image.paste(image, (w, 0))
        w += image.width
    return final_image
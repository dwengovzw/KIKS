from PIL import Image
from ipyupload import FileUpload
import numpy as np
import io

upload_widget = FileUpload(accept='image/*')

image_crop = None

def geef_afbeelding():
    if not upload_widget.value:
        print('Je hebt nog geen bestand gekozen.')
        return None

    first_key = next(iter(upload_widget.value))

    if not upload_widget.value[first_key]['metadata']['type'].startswith('image'):
        print('Kies een afbeelding.')
        return None

    image = Image.open(io.BytesIO(upload_widget.value[first_key]['content']))
    return image


def geef_gekozen_afbeelding_in_grijswaarden():
    image = geef_afbeelding()
    if image is None:
        return

    im_r = np.array(image)

    def grayConversion(image):
        grayValue = 0.07 * image[:, :, 2] + 0.72 * image[:, :, 1] + 0.21 * image[:, :, 0]
        gray_img = grayValue.astype(np.uint8)
        return gray_img

    image_grayscale = grayConversion(im_r)
    return image_grayscale

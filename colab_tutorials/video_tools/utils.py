import numpy as np
import PIL
import io
import cv2
from base64 import b64decode, b64encode


def js_to_image(js_reply):
    """
    Convert the JavaScript object into an OpenCV image.

    @param js_reply: JavaScript object containing image from webcam
    @return img: OpenCV BGR image
    """
    # decode base64 image
    image_bytes = b64decode(js_reply.split(',')[1])
    # convert bytes to numpy array
    jpg_as_np = np.frombuffer(image_bytes, dtype=np.uint8)
    # decode numpy array into OpenCV BGR image
    img = cv2.imdecode(jpg_as_np, flags=1)

    return img


def bbox_to_bytes(bbox_array):
    """
    Convert OpenCV Rectangle bounding box image into base64 byte string to be overlayed on video stream.

    @param bbox_array: Numpy array (pixels) containing rectangle to overlay on video stream.
    @return bbox_bytes: Base64 image byte string
    """
    # convert array into PIL image
    bbox_PIL = PIL.Image.fromarray(bbox_array, 'RGBA')
    iobuf = io.BytesIO()
    # format bbox into png for return
    bbox_PIL.save(iobuf, format='png')
    # format return string
    bbox_bytes = 'data:image/png;base64,{}'.format((str(b64encode(iobuf.getvalue()), 'utf-8')))
    return bbox_bytes


def image_to_bytes(image):
    """
    Convert OpenCV image into base64 byte string to be overlayed on video stream.

    @param image: Input image.
    @return img_bytes: Base64 image byte string.
    """
    ret, buffer = cv2.imencode('.jpg', image)
    jpg_as_text = b64encode(buffer).decode('utf-8')
    img_bytes = f'data:image/jpeg;base64,{jpg_as_text}'
    return img_bytes
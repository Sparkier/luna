"""
The utility functions for creating and processing the images for the feature visualisation process.
"""
from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow import keras

COLOR_CORRELATION_SVD_SQRT = np.asarray([[0.26, 0.09, 0.02],
                                         [0.27, 0.00, -0.05],
                                         [0.27, -0.09, 0.03]]).astype("float32")

MAX_NORM_SVD_SQRT = np.max(np.linalg.norm(COLOR_CORRELATION_SVD_SQRT, axis=0))

COLOR_MEAN = [0.48, 0.46, 0.41]


def initialize_image(width, height, val_range_top=1.0, val_range_bottom=-1.0):
    """
    Creates an initial randomized image to start feature vis process.
    This could be subject to optimization in the future.

    Args:
        width (int): The width of the image.
        height (int): The height of the image.

    Returns:
        A randomly generated image.
    """
    print('initializing image')
    # We start from a gray image with some random noise
    if tf.compat.v1.keras.backend.image_data_format() == 'channels_first':
        img = tf.random.uniform((1, 3, width, height), dtype=tf.dtypes.float32)
    else:
        img = tf.random.uniform((1, width, height, 3), dtype=tf.dtypes.float32)
    # rescale values to be in the middle quarter of possible values
    img = (img - 0.5) * 0.25 + 0.5
    val_range = val_range_top - val_range_bottom
    # rescale values to be in the given range of values
    img = val_range_bottom + img * val_range
    return img


def deprocess_image(img):
    """
    Takes the values of an image array and normalizes them to be in the
    standard 0-255 RGB range.

    Args:
        img (List[float]): The generated image array.
    Returns:
        A rescaled version of the image.
    """
    print('Deprocessing image')
    # compute the normal scores (z scores) and add little noise for uncertainty
    img = ((img - img.mean()) / img.std()) + 1e-5
    # ensure that the variance is 0.15
    img *= 0.15
    # croping the center adn clip the values between 0 and 1
    img = img[25:-25, 25:-25, :]
    img += 0.5
    img = np.clip(img, 0, 1)

    # convert to RGB
    img *= 255
    img = np.clip(img, 0, 255).astype("uint8")

    return img


def save_image(img, name=None):
    """
    Saves a generated image array as a numpy array in a file.

    Args:
        img (pil.Image): The generated image.
        name (str): A possible name, if none given it is auto generated.
    """
    arr = keras.preprocessing.image.img_to_array(img)
    if name is None:
        name = datetime.now().isoformat()
        name = name.replace("-", "")
        name = name.replace(":", "")
        name = name.replace("+", "")
        name = name.replace(".", "")
    np.save(f"{name}.npy", arr)


def initialize_image_ref(width, height, std=None, fft=True,
                         decorrelate=True, channels=None):
    """
    Creates an initial randomized image to start feature vis process.
    This could be subject to optimization in the future.

    Args:
        width (int): The width of the image.
        height (int): The height of the image.
        sd (float): standard deviation for noise initialization.
        fft (boolean): Image parameterization with fast fourier transformation.
        deccorelate (boolean): the color interpretation of the image tensor's color.
        channels (boolean): True for gray images.

    Returns:
        A randomly generated image.
    """
    print('initializing image')
    # We start from a gray image with some random noise
    if tf.compat.v1.keras.backend.image_data_format() == 'channels_first':
        img = tf.random.uniform((1, 3, width, height), dtype=tf.dtypes.float32)
        shape = [img.shape[0], img.shape[1], img.shape[2], img.shape[3]]
    else:
        img = tf.random.uniform((1, width, height, 3), dtype=tf.dtypes.float32)
        shape = [img.shape[0], img.shape[3], img.shape[1],
                 img.shape[2]]  # [batch, ch, h, w]

    if fft:
        image_f = fft_image(shape, std=std)
    else:
        image_f = np.random.normal(size=shape, scale=std).astype(np.float32)

    if channels:
        output = tf.nn.sigmoid(image_f)
    else:
        output = to_valid_rgb(
            image_f[..., :3], decorrelate=decorrelate, sigmoid=True)

    return output


def fft_image(shape, std=None, decay_power=1):
    """Image parameterization using 2D Fourier coefficients.

    Args:
        shape (list[int]): Image shape.
        sd (float): standard deviation as noise.
        decay_power (int): Defaults to 1.

    Returns:
        New image in spatial domain.
    """
    batch, channels, height, width = shape
    # real valued fft
    freqs = rfft2d_freqs(height, width)

    init_val_size = (2, batch, channels) + freqs.shape
    std = std or 0.01
    init_val = np.random.normal(
        size=init_val_size, scale=std).astype(np.float32)
    spectrum_real_imag_t = tf.Variable(init_val)

    scale = 1.0 / np.maximum(freqs, 1.0 / max(width, height)) ** decay_power
    scale *= np.sqrt(width * height)
    spectrum_t = tf.complex(spectrum_real_imag_t[0], spectrum_real_imag_t[1])
    scaled_spectrum_t = scale * spectrum_t

    # convert the spectrum to spatial domain
    image_t = tf.transpose(tf.signal.irfft2d(scaled_spectrum_t), (0, 2, 3, 1))
    image_t = image_t[:batch, :height, :width, :channels]
    image_t = image_t / 4.0
    return image_t


def rfft2d_freqs(height, width):
    """computation of 2D frequencies of spectrum.

    Args:
        h (int): image height.
        w (int): image width.

    Returns:
        spectrum frequency : 2D spectrum frequencies.
    """
    freq_y = np.fft.fftfreq(height)[:, None]
    # when we have an odd input dimension we need to keep one additional
    # frequency and later cut off 1 pixel
    if width % 2 == 1:
        freq_x = np.fft.fftfreq(width)[: width // 2 + 2]
    else:
        freq_x = np.fft.fftfreq(width)[: width // 2 + 1]
    return np.sqrt(freq_x * freq_x + freq_y * freq_y)


def _linear_decorrelate_color(image):
    """Color correlation matrix.

    Args:
        image (tf.Tensor): Input image.

    Returns:
        The decorrolated version of the color space of the input image.
    """
    t_flat = tf.reshape(image, [-1, 3])
    color_correlation_normalized = COLOR_CORRELATION_SVD_SQRT / MAX_NORM_SVD_SQRT
    t_flat = tf.matmul(t_flat, color_correlation_normalized.T)
    image = tf.reshape(t_flat, tf.shape(image))
    return image


def to_valid_rgb(image, decorrelate=False, sigmoid=True):
    """Transformation of input tensor to valid rgb colors.

    Args:
        image (tf.Tensor): Input image.
        decorrelate (bool): Color interpretation from whitened space if it is True.
        sigmoid (bool): Color constrained if it is True.

    Returns:
        Transfomed image with the innermost dimension.
    """
    if decorrelate:
        image = _linear_decorrelate_color(image)
    if decorrelate and not sigmoid:
        image += COLOR_MEAN
    if sigmoid:
        image = tf.nn.sigmoid(image)
    else:
        image = (2*image-1 / tf.maximum(1.0, tf.abs(2*image-1)))/2 + 0.5
    return image

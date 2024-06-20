import unittest

import numpy as np

from invoice2text.img2text.engine import read_image_to_ndarray
from tests.data import image_path


class TestReadImageToNdarray(unittest.TestCase):
    def test_read_image_to_ndarray_valid_image(self):
        image = read_image_to_ndarray(str(image_path))
        self.assertIsInstance(image, np.ndarray)

    def test_read_image_to_ndarray_invalid_image(self):
        image_path = "./false-image-path.png"
        with self.assertRaises(ValueError):
            read_image_to_ndarray(image_path)


if __name__ == "__main__":
    unittest.main()

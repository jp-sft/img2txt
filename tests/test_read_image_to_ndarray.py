import unittest
from pathlib import Path

import numpy as np

from img2text2.engine import read_image_to_ndarray


class TestReadImageToNdarray(unittest.TestCase):
    def test_read_image_to_ndarray_valid_image(self):
        image_path = Path(__file__).parent / "image.png"
        image = read_image_to_ndarray(str(image_path))
        self.assertIsInstance(image, np.ndarray)

    def test_read_image_to_ndarray_invalid_image(self):
        image_path = "./false-image.png"
        with self.assertRaises(ValueError):
            read_image_to_ndarray(image_path)


if __name__ == "__main__":
    unittest.main()

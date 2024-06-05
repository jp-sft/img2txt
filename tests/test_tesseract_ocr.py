import unittest
from pathlib import Path

import pandas as pd

from img2text.engine import TesseractOCR

image_path: str = str(Path(__file__).parent / "image.png")


class TestTesseractOCR(unittest.TestCase):
    def setUp(self):
        self.extractor = TesseractOCR()

    def test_img2text(self):
        text = self.extractor.img2text(image_path)
        self.assertIsInstance(text, str)

    def test_image2boxes(self):
        boxes = self.extractor.image2boxes(image_path)
        self.assertIsInstance(boxes, pd.DataFrame)
        self.assertIn("text", boxes.columns)


if __name__ == "__main__":
    unittest.main()

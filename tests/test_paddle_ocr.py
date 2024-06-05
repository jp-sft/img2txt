import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from img2text import TextExtractorPaddleOCR

# Path to a sample image for testing
image_path: str = str(Path(__file__).parent / "image.png")

# Mocked OCR output
mock_ocr_output = [
    [
        [
            [[442.0, 173.0], [1169.0, 173.0], [1169.0, 225.0], [442.0, 225.0]],
            ["ACKNOWLEDGEMENTS", 0.99],
        ],
        [
            [[393.0, 340.0], [1207.0, 342.0], [1207.0, 389.0], [393.0, 387.0]],
            ["We would like to thank all the designers and", 0.93],
        ],
        [
            [[399.0, 398.0], [1204.0, 398.0], [1204.0, 433.0], [399.0, 433.0]],
            ["contributors who have been involved in the", 0.95],
        ],
    ]
]


class TestTextExtractorPaddleOCR(unittest.TestCase):
    def setUp(self):
        self.extractor = TextExtractorPaddleOCR()
        self.image_array = self._read_image_mock(image_path)

    def _read_image_mock(self, image_path):
        # Create a dummy array to mock an image
        return np.zeros((100, 100, 3), dtype=np.uint8)

    @patch("img2text.engine.PaddleOCR.ocr", return_value=mock_ocr_output)
    def test_img2text(self, mock_ocr):
        text = self.extractor.img2text(image_path)
        self.assertIsInstance(text, str)
        self.assertIn("ACKNOWLEDGEMENTS", text)

    @patch("img2text.engine.PaddleOCR.ocr", return_value=mock_ocr_output)
    def test_image2boxes(self, mock_ocr):
        boxes = self.extractor.image2boxes(image_path)
        self.assertIsInstance(boxes, pd.DataFrame)
        self.assertIn("text", boxes.columns)
        self.assertIn("ACKNOWLEDGEMENTS", boxes["text"].values)

    # @patch("img2text.engine.PaddleOCR.ocr", return_value=mock_ocr_output)
    # def test_singleton_ocr_instance(self, mock_ocr):
    #     # Create multiple instances of the extractor
    #     extractor1 = TextExtractorPaddleOCR()
    #     extractor2 = TextExtractorPaddleOCR()
    #     # Both should use the same OCR instance
    #     self.assertIs(extractor1._get_ocr(), extractor2._get_ocr())
    #     # Ensure the OCR method was called only once
    #     extractor1.img2text(image_path)
    #     extractor2.img2text(image_path)
    #     mock_ocr.assert_called_once()

    def test_align_texts(self):
        aligned_texts = self.extractor._align_texts(mock_ocr_output[0])
        self.assertIsInstance(aligned_texts, dict)
        # Check if keys are line heights
        for key in aligned_texts.keys():
            self.assertIsInstance(key, int)
        # Check if values are correctly aligned texts
        self.assertIn("ACKNOWLEDGEMENTS", aligned_texts.values())
        self.assertIn(
            "We would like to thank all the designers and", aligned_texts.values()
        )
        self.assertIn(
            "contributors who have been involved in the", aligned_texts.values()
        )


if __name__ == "__main__":
    unittest.main()

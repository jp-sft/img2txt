import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd

from img2text.engine import TextExtractorPaddleOCR

image_path: str = str(Path(__file__).parent / "image.png")


class TestTextExtractorPaddleOCR(unittest.TestCase):
    def test_img2text_single_instance(self):
        with patch.object(TextExtractorPaddleOCR, "_get_ocr") as mock_get_ocr:
            mock_ocr = Mock()
            mock_ocr.ocr.return_value = [(["", "text1"]), (["", "text2"])]
            mock_get_ocr.return_value = mock_ocr

            extractor = TextExtractorPaddleOCR()
            text = extractor.img2text(image_path)

            self.assertIsInstance(text, str)
            mock_get_ocr.assert_called_once()

    def test_image2boxes_single_instance(self):
        with patch.object(TextExtractorPaddleOCR, "_get_ocr") as mock_get_ocr:
            mock_ocr = Mock()
            mock_ocr.ocr.return_value = [(["", "text1"]), (["", "text2"])]
            mock_get_ocr.return_value = mock_ocr

            extractor = TextExtractorPaddleOCR()
            boxes = extractor.image2boxes(image_path)

            self.assertIsInstance(boxes, pd.DataFrame)
            self.assertIn("text", boxes.columns)
            mock_get_ocr.assert_called_once()

    def test_img2text_multiple_instances(self):
        with patch.object(TextExtractorPaddleOCR, "_get_ocr") as mock_get_ocr:
            mock_ocr = Mock()
            mock_ocr.ocr.return_value = [(["", "text1"]), (["", "text2"])]
            mock_get_ocr.return_value = mock_ocr

            # First instance
            extractor1 = TextExtractorPaddleOCR()
            text1 = extractor1.img2text(image_path)

            # Second instance
            extractor2 = TextExtractorPaddleOCR()
            text2 = extractor2.img2text(image_path)

            self.assertIsInstance(text1, str)
            self.assertIsInstance(text2, str)
            self.assertEqual(text1, text2)
            mock_get_ocr.assert_called_once()

    def test_image2boxes_multiple_instances(self):
        with patch.object(TextExtractorPaddleOCR, "_get_ocr") as mock_get_ocr:
            mock_ocr = Mock()
            mock_ocr.ocr.return_value = [(["", "text1"]), (["", "text2"])]
            mock_get_ocr.return_value = mock_ocr

            # First instance
            extractor1 = TextExtractorPaddleOCR()
            boxes1 = extractor1.image2boxes(image_path)

            # Second instance
            extractor2 = TextExtractorPaddleOCR()
            boxes2 = extractor2.image2boxes(image_path)

            self.assertIsInstance(boxes1, pd.DataFrame)
            self.assertIsInstance(boxes2, pd.DataFrame)
            self.assertEqual(boxes1.to_dict(), boxes2.to_dict())
            mock_get_ocr.assert_called_once()


if __name__ == "__main__":
    unittest.main()

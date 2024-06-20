import unittest
from fastapi.testclient import TestClient
from invoice2text.app import app
from tests.data import image_path, pdf_path, unsupported_file_path


class OCRPipelineTestCase(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_process_image(self):
        with open(image_path, "rb") as image_file:
            response = self.client.post(
                "/process_image",
                data={
                    "use_gpu": "false",
                    "psm": "6",
                    "paddle_lang": "french",
                    "tesseract_lang": "eng+fra+ara",
                },
                files={"file": ("image.png", image_file, "image/png")},
            )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("texts", data)
        self.assertIn("boxes", data)

    def test_process_pdf(self):
        with open(pdf_path, "rb") as pdf_file:
            response = self.client.post(
                "/process_image",
                data={
                    "use_gpu": "false",
                    "psm": "6",
                    "paddle_lang": "french",
                    "tesseract_lang": "eng+fra+ara",
                },
                files={"file": ("document.pdf", pdf_file, "application/pdf")},
            )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("texts", data)
        self.assertIn("boxes", data)

    def test_unsupported_file_type(self):
        with open(unsupported_file_path, "rb") as unsupported_file:
            response = self.client.post(
                "/process_image",
                data={
                    "use_gpu": "false",
                    "psm": "6",
                    "paddle_lang": "french",
                    "tesseract_lang": "eng+fra+ara",
                },
                files={"file": ("unsupported.txt", unsupported_file, "text/plain")},
            )
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertEqual(data["detail"], "Unsupported file type")

    def test_missing_file(self):
        response = self.client.post(
            "/process_image",
            data={
                "use_gpu": "false",
                "psm": "6",
                "paddle_lang": "french",
                "tesseract_lang": "eng+fra+ara",
            },
        )
        self.assertEqual(response.status_code, 422)  # Unprocessable Entity

    def test_invalid_parameters(self):
        with open(image_path, "rb") as image_file:
            response = self.client.post(
                "/process_image",
                data={
                    "use_gpu": "invalid",
                    "psm": "invalid",
                    "paddle_lang": "french",
                    "tesseract_lang": "eng+fra+ara",
                },
                files={"file": ("image.png", image_file, "image/png")},
            )
        self.assertEqual(response.status_code, 422)  # Unprocessable Entity


if __name__ == "__main__":
    unittest.main()

import cv2
import numpy as np
from paddleocr import PaddleOCR
import pytesseract
import csv

from invoice2text.img2text.utils.singleton import SingletonByParams


class ImagePreprocessor:
    @staticmethod
    def preprocess(image_path: str | bytes):
        if isinstance(image_path, str):
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        else:
            nparr = np.frombuffer(image_path, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        # Appliquer un filtre bilatéral pour réduire le bruit tout
        # en conservant les bords
        image = cv2.bilateralFilter(image, 9, 75, 75)
        # Binarisation de l'image
        _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return image


class TextDetection(metaclass=SingletonByParams):
    def __init__(
        self,
        use_gpu=False,
        lang="en",
        det_model_dir=None,
        rec_model_dir=None,
        cls_model_dir=None,
    ):
        self.ocr = PaddleOCR(
            use_angle_cls=True,
            show_log=False,
            lang=lang,
            use_gpu=use_gpu,
            det_model_dir=det_model_dir,
            rec_model_dir=rec_model_dir,
            cls_model_dir=cls_model_dir,
        )

    def detect_text(self, image):
        result = self.ocr.ocr(image, cls=True)
        boxes = [line[0] for line in result[0]]
        return boxes


class TextExtraction(metaclass=SingletonByParams):
    def __init__(self, lang="eng", psm=3, oem=3):
        self.lang = lang
        self.config = f"--psm {psm} --oem {oem}"

    def extract_text(self, image, boxes):
        texts = []
        for box in boxes:
            x_min = int(min([point[0] for point in box]))
            y_min = int(min([point[1] for point in box]))
            x_max = int(max([point[0] for point in box]))
            y_max = int(max([point[1] for point in box]))
            cropped_image = image[y_min:y_max, x_min:x_max]
            text = pytesseract.image_to_string(
                cropped_image, lang=self.lang, config=self.config
            )
            texts.append(text)
        return texts


class OCRPipeline(metaclass=SingletonByParams):
    def __init__(
        self,
        use_gpu=False,
        paddle_lang="en",
        tesseract_lang="eng",
        psm=3,
        oem=3,
        det_model_dir=None,
        rec_model_dir=None,
        cls_model_dir=None,
    ):
        self.preprocessor = ImagePreprocessor()
        self.detector = TextDetection(
            use_gpu, paddle_lang, det_model_dir, rec_model_dir, cls_model_dir
        )
        self.extractor = TextExtraction(tesseract_lang, psm, oem)

    def process_image(self, image_path: str | bytes):
        preprocessed_image = self.preprocessor.preprocess(image_path)
        boxes = self.detector.detect_text(preprocessed_image)
        if isinstance(image_path, str):
            image = cv2.imread(image_path)
        else:
            nparr = np.frombuffer(image_path, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        texts = self.extractor.extract_text(image, boxes)
        return texts, boxes

    def save_to_csv(self, texts, boxes, output_csv_path):
        with open(output_csv_path, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["Text", "x_min", "y_min", "x_max", "y_max"])
            for text, box in zip(texts, boxes):
                x_min = int(min([point[0] for point in box]))
                y_min = int(min([point[1] for point in box]))
                x_max = int(max([point[0] for point in box]))
                y_max = int(max([point[1] for point in box]))
                writer.writerow([text, x_min, y_min, x_max, y_max])


if __name__ == "__main__":
    import os

    image_path = os.environ["IMAGE_TEST_PATH"]
    output_csv_path = "output.csv"
    ocr_pipeline = OCRPipeline(
        use_gpu=False, psm=6, paddle_lang="french", tesseract_lang="en+fra+ara"
    )
    texts, boxes = ocr_pipeline.process_image(image_path)
    print("\n".join(texts))
    ocr_pipeline.save_to_csv(texts, boxes, output_csv_path)

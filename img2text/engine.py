from abc import ABC, abstractmethod

import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from paddleocr import PaddleOCR
from pytesseract import Output, image_to_data, image_to_string

from img2text.utils.image_rotation_auto import detect_angle_rotate


class TextExtractor(ABC):
    @abstractmethod
    def img2text(self, image_path: str) -> str:
        pass

    @abstractmethod
    def image2boxes(self, image_path: str) -> pd.DataFrame:
        pass


class TextExtractorPaddleOCR(TextExtractor):
    _ocr_instances: dict[tuple, PaddleOCR] = {}

    def __init__(
        self,
        ocr_name: str = "ocr",
        ocr_params: dict | None = None,
        force_init: bool = False,
    ) -> None:
        self.ocr_name = ocr_name
        self.ocr_params = ocr_params or self._default_ocr_params()
        self.force_init = force_init

    def img2text(self, image_path: str) -> str:
        image = read_image_to_ndarray(image_path)
        _, rotated_image = detect_angle_rotate(image)
        ocr_results = self._get_ocr().ocr(rotated_image, cls=True)
        result = ""
        for idx in range(len(ocr_results)):
            aligned_texts = self._align_texts(ocr_results[idx])
            result += "\n".join(aligned_texts.values())
        return result

    def image2boxes(self, image_path: str) -> pd.DataFrame:
        image = read_image_to_ndarray(image_path)
        _, rotated_image = detect_angle_rotate(image)
        result = self._get_ocr().ocr(rotated_image, cls=True)
        boxes = [
            [
                int(line[0][0][0]),
                int(line[0][0][1]),
                int(line[0][2][0]),
                int(line[0][2][1]),
                line[1][0].strip(),
            ]
            for line in result[0]
            if line[1][0].strip()
        ]
        return get_box(boxes)

    def _get_ocr(self) -> PaddleOCR:
        params_key = (self.ocr_name, frozenset(self.ocr_params.items()))
        if params_key not in self._ocr_instances or self.force_init:
            self._ocr_instances[params_key] = PaddleOCR(**self.ocr_params)
        return self._ocr_instances[params_key]

    @staticmethod
    def _default_ocr_params() -> dict:
        return {
            "use_angle_cls": True,
            "lang": "fr",
            "show_log": True,
            "type": "structure",
            "max_text_length": 50,
            "det_east_cover_thresh": 0.05,
            "det_db_score_mode": "slow",
        }

    @staticmethod
    def _align_texts(
        ocr_results: list[tuple[list[list[float]], tuple[str, float]]],
    ) -> dict[int, str]:
        aligned_texts: dict[int, list[str]] = {}

        # Regrouper les résultats de l'OCR par ligne
        for box, (text, _) in ocr_results:
            # Calculer la hauteur moyenne de la boîte pour déterminer la ligne
            line_height = sum(point[1] for point in box) / len(box)

            # Arrondir la hauteur à l'entier le plus proche pour simplifier l'alignement
            line_height = round(line_height)

            # Ajouter le texte à la ligne correspondante dans le dictionnaire aligné
            if line_height in aligned_texts:
                aligned_texts[line_height].append(text)
            else:
                aligned_texts[line_height] = [text]

        # Calculer les espacements entre les boîtes sur la même ligne
        line_spacing = {}
        for line_height, texts in aligned_texts.items():
            # Trier les boîtes par position x
            sorted_texts = sorted(
                texts, key=lambda x: ocr_results[texts.index(x)][0][0][0]
            )
            distances = []
            for i in range(len(sorted_texts) - 1):
                # Calculer la distance entre les boîtes consécutives
                distance = (
                    ocr_results[texts.index(sorted_texts[i + 1])][0][0][0]
                    - ocr_results[texts.index(sorted_texts[i])][0][2][0]
                )
                distances.append(distance)
            line_spacing[line_height] = distances

        # Ajuster les textes alignés sur chaque ligne en fonction des espacements
        for line_height, texts in aligned_texts.items():
            adjusted_texts = []
            for i, text in enumerate(texts):
                # Ajouter le texte avec un espace ajusté à la fin sauf pour le dernier texte
                adjusted_texts.append(
                    text + " " * round(line_spacing[line_height][i])
                    if i < len(texts) - 1
                    else text
                )
            aligned_texts[line_height] = adjusted_texts

        # Concaténer les textes alignés sur chaque ligne avec des espaces
        for line_height, texts in aligned_texts.items():
            aligned_texts[line_height] = " ".join(texts)

        return aligned_texts


class TesseractOCR(TextExtractor):
    def __init__(self, tesseract_cmd: str = "tesseract"):
        self.tesseract_cmd = tesseract_cmd

    def img2text(self, image_path: str) -> str:
        image = read_image_to_ndarray(image_path)
        _, rotated_image = detect_angle_rotate(image)
        text = image_to_string(rotated_image, lang="fra")
        return text

    def image2boxes(self, image_path: str) -> pd.DataFrame:
        image = read_image_to_ndarray(image_path)
        _, rotated_image = detect_angle_rotate(image)
        data = image_to_data(rotated_image, output_type=Output.DICT, lang="fra")

        boxes = []
        for i in range(len(data["level"])):
            if data["text"][i].strip():
                box = [
                    data["left"][i],
                    data["top"][i],
                    data["left"][i] + data["width"][i],
                    data["top"][i] + data["height"][i],
                    data["text"][i].strip(),
                ]
                boxes.append(box)

        return get_box(boxes)


class TextExtractorFactory:
    @staticmethod
    def create_extractor(engine: str, **kwargs) -> TextExtractor:
        if engine == "paddle":
            return TextExtractorPaddleOCR(**kwargs)
        elif engine == "tesseract":
            return TesseractOCR(**kwargs)
        else:
            raise ValueError(f"Unsupported OCR engine: {engine}")


def get_box(boxes):
    """
    Converts detected text boxes into a structured DataFrame.

    Args:
        boxes (list): List of detected text boxes.

    Returns:
        pd.DataFrame: DataFrame with structured information about the boxes.
    """
    bboxes = pd.DataFrame(boxes, columns=["left", "top", "x1", "y1", "text"])
    bboxes["height"] = bboxes.y1 - bboxes.top
    bboxes["width"] = bboxes.x1 - bboxes.left
    bboxes["x0"] = bboxes.left
    bboxes["y0"] = bboxes.top
    bboxes["x1"] = bboxes.left + bboxes.width
    bboxes["y1"] = bboxes.top + bboxes.height
    bboxes["h"] = bboxes.height
    bboxes["w"] = bboxes.width
    bboxes["c0"] = bboxes.x0 + bboxes.w
    bboxes["c1"] = bboxes.y0 + bboxes.h

    return bboxes


def put_line_num(bboxes):
    """
    Assigns line numbers to detected text boxes based on their vertical positions.

    Args:
        bboxes (pd.DataFrame): DataFrame with structured information about the boxes.
    """
    bboxes["line_num"] = [-1] * len(bboxes)
    ln = -1
    indexes = bboxes.index.tolist()

    while indexes:
        ln += 1
        bboxes.loc[indexes[0], "line_num"] = ln
        box = bboxes.loc[indexes[0]]
        groups = [indexes[0]]

        for i in indexes[1:]:
            box2 = bboxes.loc[i]
            ratio = min(box.h, box2.h) / max(box.h, box2.h)
            if ratio > 0.6 and (
                box.y0 <= box2.y0 <= box.y1
                or box.y0 <= box2.y1 <= box.y1
                or box2.y0 <= box.y0 <= box2.y1
                or box2.y0 <= box.y1 <= box2.y1
            ):
                bboxes.loc[i, "line_num"] = ln
                groups.append(i)

        indexes = [i for i in indexes if i not in groups]

    line_num = bboxes.line_num.unique().tolist()
    bboxes.line_num = bboxes.line_num.apply(lambda x: line_num.index(x) + 1)


def read_image_to_ndarray(image_path: str) -> np.ndarray:
    """
    Reads an image from a file path and converts it to a NumPy array.

    Args:
        image_path: The path to the image file.

    Returns:
        np.ndarray: A NumPy array representing the image data.

    Raises:
        ValueError: If the image cannot be read.
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Error reading image: {image_path}")
        return image
    except Exception as e:
        raise ValueError(f"Error reading image: {image_path} ({e})") from e


def plot_ocr_res(image: np.ndarray, ocr_res: pd.DataFrame, save_path: str = None):
    """
    Plots OCR results on the image.

    Args:
        image (np.ndarray): The input image.
        ocr_res (pd.DataFrame): DataFrame containing OCR results with bounding boxes and text.
        save_path (str, optional): Path to save the plotted image. If None, the image is not saved.

    Returns:
        None
    """
    # Make a copy of the image
    image_copy = image.copy()

    # Get image dimensions
    image_height, image_width, _ = image_copy.shape

    # Plot the image
    plt.imshow(cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB))

    # Plot the bounding boxes and text
    for _, row in ocr_res.iterrows():
        left, top, width, height, text = (
            row["x0"],
            row["y0"],
            row["w"],
            row["h"],
            row["text"],
        )

        # Calculate text size based on image size
        text_size = int(min(image_height, image_width) * 0.01)

        plt.plot(
            [left, left + width, left + width, left, left],
            [top, top, top + height, top + height, top],
            color="red",
            linewidth=0.5,
        )
        plt.text(
            left,
            top,
            text,
            verticalalignment="top",
            color="red",
            fontsize=text_size,
            bbox={"facecolor": "yellow", "alpha": 0.5},
        )

    if save_path:
        plt.savefig(save_path)
    plt.show()


if __name__ == "__main__":
    import os

    image_path: str = os.environ["TEST_IMAGE_PATH"]

    plot_image_path = os.environ["TEST_OUTPUT_IMAGE_PATH"]

    extractor = TextExtractorPaddleOCR()
    text = extractor.img2text(image_path)
    print(text)
    bboxe = extractor.image2boxes(image_path)
    image = read_image_to_ndarray(image_path)
    plot_ocr_res(image, bboxe, plot_image_path)

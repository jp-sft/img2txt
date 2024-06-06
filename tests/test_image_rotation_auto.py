import cv2
import numpy as np

from img2text.utils.image_rotation_auto import (
    detect_angle,
    detect_angle_rotate,
    rotate,
)


def test_detect_angle_no_lines():
    image = np.zeros((100, 100), dtype=np.uint8)
    assert detect_angle(image) == 0.0


def test_detect_angle_with_lines():
    image = np.zeros((100, 100), dtype=np.uint8)
    cv2.line(image, (0, 0), (99, 99), 255, 2)  # type: ignore
    angle = detect_angle(image)
    assert -1.0 <= angle <= 1.0  # Should be close to 0 degrees


def test_rotate():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.line(image, (0, 0), (99, 99), (255, 255, 255), 2)
    rotated_image = rotate(image, 45)
    assert rotated_image.shape == image.shape


def test_detect_angle_rotate_no_lines():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    angle, rotated_image = detect_angle_rotate(image)
    assert angle == 0.0
    assert rotated_image.shape == image.shape


def test_detect_angle_rotate_with_lines():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.line(image, (0, 0), (99, 99), (255, 255, 255), 2)
    angle, rotated_image = detect_angle_rotate(image)
    assert -1.0 <= angle <= 1.0  # Should be close to 0 degrees
    assert rotated_image.shape == image.shape

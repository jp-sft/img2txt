import cv2
import numpy as np


def detect_angle(image: np.ndarray) -> float:
    """
    Detects the rotation angle of an image using the Hough transform.

    Args:
        image (np.ndarray): The input image as a NumPy array.

    Returns:
        float: The median angle of the detected lines in the image, or 0 if no lines are found.
    """
    # Convert to grayscale and apply Gaussian blur
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Canny edge detection
    edges = cv2.Canny(gray, 50, 150)

    # Use Hough transform to detect lines
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10
    )

    if lines is None or not lines.any():
        return 0.0

    # Calculate angles of detected lines
    angles = np.arctan2(
        lines[:, 0, 3] - lines[:, 0, 1], lines[:, 0, 2] - lines[:, 0, 0]
    )
    return np.median(angles) * 180 / np.pi  # Convert radians to degrees


def rotate(image: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotates an image around its center.

    Args:
        image (np.ndarray): The input image as a NumPy array.
        angle (float): The rotation angle in degrees.

    Returns:
        np.ndarray: The rotated image.
    """
    height, width = image.shape[:2]
    center = (width / 2, height / 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, rotation_matrix, (width, height))


def detect_angle_rotate(img: np.ndarray) -> tuple[float, np.ndarray]:
    """
    Detects the rotation angle of an image and rotates it accordingly.

    Args:
        img (np.ndarray): The input image as a NumPy array.

    Returns:
        tuple[float, np.ndarray]: A tuple containing the detected angle and the rotated image.
    """
    median_angle = detect_angle(img)
    img_rotated = rotate(img, median_angle)
    return median_angle, img_rotated

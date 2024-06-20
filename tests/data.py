from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"

image_path: str = str(DATA_DIR / "test_image.png")
pdf_path = str(DATA_DIR / "test_document.pdf")
unsupported_file_path = str(DATA_DIR / "test_unsupported.txt")

__all__ = ["image_path", "pdf_path", "unsupported_file_path"]

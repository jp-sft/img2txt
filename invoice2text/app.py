from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from pydantic import BaseModel
import os
from typing import List
from invoice2text.img2text import OCRPipeline
import tempfile
import fitz  # PyMuPDF for PDF handling

app = FastAPI()


class OCRRequest(BaseModel):
    use_gpu: bool = False
    psm: int = 6
    paddle_lang: str = "french"
    tesseract_lang: str = "eng+fra+ara"


def save_temp_file(uploaded_file: UploadFile) -> str:
    try:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_file.file.read())
            return temp_file.name
    except Exception:
        raise HTTPException(status_code=500, detail="Error saving temp file")


def read_pdf(file_path: str) -> List[bytes]:
    doc = fitz.open(file_path)
    image_buffers = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap()
        img_buffer = pix.tobytes("png")  # Get the image buffer as PNG
        image_buffers.append(img_buffer)
    return image_buffers


@app.post("/process_image")
def process_image(
    use_gpu: bool = Form(False),
    psm: int = Form(6),
    paddle_lang: str = Form("french"),
    tesseract_lang: str = Form("eng+fra+ara"),
    file: UploadFile = File(...),
):
    temp_file_path = save_temp_file(file)
    if file.content_type.startswith("image/"):
        images = [temp_file_path]
    elif file.content_type == "application/pdf":
        images = read_pdf(temp_file_path)
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type")
    ocr_pipeline = OCRPipeline(
        use_gpu=use_gpu,
        psm=psm,
        paddle_lang=paddle_lang,
        tesseract_lang=tesseract_lang,
    )
    try:

        all_texts, all_boxes = [], []
        for image in images:
            texts, boxes = ocr_pipeline.process_image(image)
            all_texts.extend(texts)
            all_boxes.extend(boxes)

        return {"texts": all_texts, "boxes": all_boxes}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.remove(temp_file_path)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

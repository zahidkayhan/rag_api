from PIL import Image
import pytesseract
import io

def ocr_image_bytes(image_bytes: bytes) -> str:
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    text = pytesseract.image_to_string(img)
    return text

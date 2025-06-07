import os
import shutil
from pdf2image import convert_from_path
from docx import Document
from PIL import Image, ImageDraw

POPLER_PATH = r"D:\PROJEK\latian\EkstensiPDF\poppler-24.08.0\Library\bin"

def convert_pdf_to_images(pdf_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    try:
        pages = convert_from_path(pdf_path, dpi=200, poppler_path=POPLER_PATH)
    except Exception as e:
        print(f"[ERROR PDF]: {e}")
        return []

    image_paths = []
    for i, page in enumerate(pages):
        image_path = os.path.join(output_dir, f"pdf_page_{i+1}.png")
        page.save(image_path, "PNG")
        image_paths.append(image_path)
    return image_paths

def convert_docx_to_images(docx_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    try:
        doc = Document(docx_path)
        lines = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    except Exception as e:
        print(f"[ERROR DOCX]: {e}")
        lines = ["[Tidak bisa membaca dokumen]"]

    images = []
    max_lines_per_page = 45
    pages = [lines[i:i+max_lines_per_page] for i in range(0, len(lines), max_lines_per_page)]

    for i, page_lines in enumerate(pages):
        image = Image.new("RGB", (800, 1000), color="white")
        draw = ImageDraw.Draw(image)
        y = 10
        for line in page_lines:
            draw.text((10, y), line[:120], fill="black")
            y += 20
        image_path = os.path.join(output_dir, f"docx_page_{i+1}.png")
        image.save(image_path)
        images.append(image_path)

    return images

def delete_temp_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)

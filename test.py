import argparse
from enum import Enum
import os
from google.cloud import vision
from PIL import Image, ImageDraw
import shutil

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\Users\YASH\Downloads\visionocr-416217-4631375ec287.json"


class FeatureType(Enum):
    PAGE = 1
    BLOCK = 2
    PARA = 3
    WORD = 4
    SYMBOL = 5


def draw_boxes(image, bounds, color):
    draw = ImageDraw.Draw(image)

    for bound in bounds:
        draw.polygon(
            [
                bound.vertices[0].x,
                bound.vertices[0].y,
                bound.vertices[1].x,
                bound.vertices[1].y,
                bound.vertices[2].x,
                bound.vertices[2].y,
                bound.vertices[3].x,
                bound.vertices[3].y,
            ],
            None,
            color,
        )
    return image


def get_document_bounds(image_file, feature):
    client = vision.ImageAnnotatorClient()
    bounds = []

    with open(image_file, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)
    response = client.document_text_detection(image=image)
    document = response.full_text_annotation

    for page in document.pages:
        for block in page.blocks:
            for paragraph in block.paragraphs:
                for word in paragraph.words:
                    for symbol in word.symbols:
                        if feature == FeatureType.SYMBOL:
                            bounds.append(symbol.bounding_box)
                    if feature == FeatureType.WORD:
                        bounds.append(word.bounding_box)
                if feature == FeatureType.PARA:
                    bounds.append(paragraph.bounding_box)
            if feature == FeatureType.BLOCK:
                bounds.append(block.bounding_box)

    return bounds


def render_doc_text(image_path, output_folder):
    image = Image.open(image_path)
    bounds = get_document_bounds(image_path, FeatureType.BLOCK)
    draw_boxes(image, bounds, "blue")
    bounds = get_document_bounds(image_path, FeatureType.PARA)
    draw_boxes(image, bounds, "red")
    bounds = get_document_bounds(image_path, FeatureType.WORD)
    draw_boxes(image, bounds, "yellow")

    output_path = os.path.join(output_folder, "output_image.jpg")
    image.save(output_path)


def detect_document(path, output_folder):
    client = vision.ImageAnnotatorClient()

    with open(path, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)
    response = client.document_text_detection(image=image)

    data = []
    for page in response.full_text_annotation.pages:
        for block in page.blocks:
            for paragraph in block.paragraphs:
                for word in paragraph.words:
                    word_text = "".join(
                        [symbol.text for symbol in word.symbols])
                    data.append(str(word_text))

    output_text_path = os.path.join(output_folder, "output_text.txt")
    with open(output_text_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write("%s\n" % str(item))


if __name__ == "__main__":
    input_folder = r"C:\Users\YASH\Desktop\OCR\google-vision\dataset"
    output_folder = r"C:\Users\YASH\Desktop\OCR\google-vision\output"

    for file_name in os.listdir(input_folder):
        if file_name.endswith(('.jpg', '.jpeg', '.png')):
            file_path = os.path.join(input_folder, file_name)
            file_output_folder = os.path.join(
                output_folder, file_name.split('.')[0])
            os.makedirs(file_output_folder, exist_ok=True)

            shutil.copyfile(file_path, os.path.join(
                file_output_folder, file_name))

            render_doc_text(file_path, file_output_folder)
            detect_document(file_path, file_output_folder)
            print(f"Processed {file_name}")

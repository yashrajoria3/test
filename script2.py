import argparse
from enum import Enum
import os
from google.cloud import vision
from PIL import Image, ImageDraw

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\Users\YASH\Downloads\visionocr-416217-4631375ec287.json"

path = r'C:\Users\YASH\Desktop\OCR\google-vision\dataset\7ce32ed2-7186-479a-8f7a-84b9901ea412.jpg'


def detect_document(path):
    """Detects document features in an image."""
    from google.cloud import vision

    client = vision.ImageAnnotatorClient()

    with open(path, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.document_text_detection(image=image)

    data = []
    for page in response.full_text_annotation.pages:
        for block in page.blocks:
            # print(f"\nBlock confidence: {block.confidence}\n")

            for paragraph in block.paragraphs:
                # print("Paragraph confidence: {}".format(paragraph.confidence))

                for word in paragraph.words:
                    word_text = "".join(
                        [symbol.text for symbol in word.symbols])
                    print(
                        word_text
                    )
                    data.append(str(word_text))

    with open('output.txt', 'w', encoding='utf-8') as f:
        for item in data:
            f.write("%s\n" % str(item))

    if response.error.message:
        raise Exception(
            "{}\nFor more info on error messages, check: "
            "https://cloud.google.com/apis/design/errors".format(
                response.error.message)
        )


detect_document(path)

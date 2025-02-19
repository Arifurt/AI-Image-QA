from transformers import BlipProcessor, BlipForQuestionAnswering
from PIL import Image

def answer_image_question(image_path, question):
    processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
    image = Image.open(image_path).convert("RGB")
    inputs = processor(image, question, return_tensors="pt")
    outputs = model(**inputs)
    answer = processor.decode(outputs.logits.argmax(-1)[0])
    return answer

if __name__ == '__main__':
    img_path = input("Enter image path: ")
    question = input("Enter your question about the image: ")
    answer = answer_image_question(img_path, question)
    print("Answer:", answer)

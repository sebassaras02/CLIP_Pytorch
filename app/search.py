from transformers import ViTModel, AutoModelForMaskedLM, AutoTokenizer, ViTImageProcessor, DistilBertModel
from pinecone import Pinecone
from dotenv import load_dotenv
import torch


load_dotenv('../.env')
pc = Pinecone()
index = pc.Index("clipmodel")


from io import BytesIO
import base64
from PIL import Image

import sys

sys.path.append('../src')

from model import CLIPChemistryModel, TextEncoderHead, ImageEncoderHead


ENCODER_BASE = DistilBertModel.from_pretrained("distilbert-base-uncased")
IMAGE_BASE = ViTModel.from_pretrained("google/vit-base-patch16-224")
text_encoder = TextEncoderHead(model=ENCODER_BASE)
image_encoder = ImageEncoderHead(model=IMAGE_BASE)

clip_model = CLIPChemistryModel(text_encoder=text_encoder, image_encoder=image_encoder)

clip_model.load_state_dict(torch.load('/Users/sebastianalejandrosarastizambonino/Documents/projects/CLIP_Pytorch/src/best_model_fashion.pth', map_location=torch.device('cpu')))

te_final = clip_model.text_encoder
ie_final = clip_model.image_encoder

def process_text_for_encoder(text, model):
    # tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    encoded_input = tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=256)
    input_ids = encoded_input['input_ids']
    attention_mask = encoded_input['attention_mask']
    output = model(input_ids=input_ids, attention_mask=attention_mask)
    return output.detach().numpy().tolist()[0]

def process_image_for_encoder(image, model):
    # image = Image.open(BytesIO(image))
    print(type(image))
    image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
    image_tensor = image_processor(image, 
            return_tensors="pt", 
            do_resize=True
            )['pixel_values']
    output =  model(pixel_values=image_tensor)
    return output.detach().numpy().tolist()[0]

def search_similarity(input, mode, top_k=5):
    if mode == 'text':
        output = process_text_for_encoder(input, model=te_final)
    else:
        output = input
    
    if mode == 'text':
        mode_search = 'image'
        response = index.query(
            namespace="space-" + mode_search + "-fashion",
            vector=output,
            top_k=top_k,
            include_values=True,
            include_metadata=True
        )
        similar_images = [value['metadata']['image'] for value in response['matches']]
        return similar_images
    elif mode == 'image':
        mode_search = 'text'
        response = index.query(
            namespace="space-" + mode_search + "-fashion",
            vector=output,
            top_k=top_k,
            include_values=True,
            include_metadata=True
        )
        similar_text = [value['metadata']['text'] for value in response['matches']]
        return similar_text
    else:
        raise ValueError("mode must be either 'text' or 'image'")
    
def process_image_for_encoder_gradio(image, is_bytes=True):
    """Procesa tanto imágenes en bytes como objetos PIL Image"""
    try:
        if is_bytes:
            # Si la imagen viene en bytes
            image = Image.open(BytesIO(image))
        else:
            # Si la imagen ya es un objeto PIL Image o viene de gradio
            if not isinstance(image, Image.Image):
                # Si viene de gradio, podría ser un numpy array
                image = Image.fromarray(image)
        
        image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
        image_tensor = image_processor(image, 
                return_tensors="pt", 
                do_resize=True
                )['pixel_values']
        output = ie_final(pixel_values=image_tensor)
        return output.detach().numpy().tolist()[0]
    except Exception as e:
        print(f"Error en process_image_for_encoder: {e}")
        raise
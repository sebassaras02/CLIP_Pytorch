from model import CLIPChemistryModel, TextEncoderHead, ImageEncoderHead
from transformers import ViTModel, AutoModelForMaskedLM, AutoTokenizer, ViTImageProcessor

from io import BytesIO
import base64

from PIL import Image

def bytes_to_str(bytes_data):
    return base64.b64encode(bytes_data).decode('utf-8')

def str_to_bytes(str_data):
    return base64.b64decode(str_data)

def push_embeddings_to_pine_cone(index, embeddings, df, mode, length):
    records = []
    for i in range(length):
        if mode == 'text':
            records.append({
                "id": str(mode) + str(i), 
                "values": embeddings[i],
                "metadata": {str(mode): df[mode].iloc[i]}})
        elif mode == 'image':
            records.append({
                "id": str(mode) + str(i), 
                "values": embeddings[i],
                "metadata": {str(mode): bytes_to_str(df[mode].iloc[i]['bytes'])}})
        else:
            raise ValueError("mode must be either 'text' or 'image'")
    
    index.upsert(
        vectors=records,
        namespace="space-" + mode
    )
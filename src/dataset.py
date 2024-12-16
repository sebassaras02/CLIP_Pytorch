from torch.utils.data import Dataset
import torch

import numpy as np
import pandas as pd

from io import BytesIO
from PIL import Image

from transformers import DistilBertTokenizer, ViTImageProcessor, AutoTokenizer, AutoModelForMaskedLM, AutoImageProcessor

class CLIPChemistryDataset(Dataset):
    def __init__(self, limit=None, shape=(224, 224)):
        chemistry_dataset = "hf://datasets/VuongQuoc/Chemistry_text_to_image/data/train-00000-of-00001-f1f5b2eab68f0d2f.parquet"
        fashion_dataset = "hf://datasets/rajuptvs/ecommerce_products_clip/data/train-00000-of-00001-1f042f20fd269c32.parquet"
        self.data = pd.read_parquet(fashion_dataset)
        if limit:
            self.data = self.data[:limit]
        self.image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
        # self.image_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-18")
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        # self.tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image = self.data.loc[idx, 'image']['bytes']
        image = self._preprocess_image(image)
        label = self.data.loc[idx, 'Clipinfo']
        input_ids, attention_mask = self._preprocess_text(label)
        return image.squeeze(0), input_ids.squeeze(0), attention_mask.squeeze(0) 
    
    def _preprocess_image(self, bytes):
        image = Image.open(BytesIO(bytes))
        image_tensor = self.image_processor(image, 
            return_tensors="pt", 
            do_resize=True
            )['pixel_values']
        return image_tensor
    
    def _preprocess_text(self, text):
        encoded_input = self.tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=256)
        return encoded_input['input_ids'], encoded_input['attention_mask']
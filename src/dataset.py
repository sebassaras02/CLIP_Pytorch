from torch.utils.data import Dataset
from torchvision.transforms import Resize, Compose, ToTensor


import numpy as np
import pandas as pd

from io import BytesIO
from PIL import Image

from transformers import DistilBertTokenizer, ViTImageProcessor, AutoTokenizer, AutoModelForMaskedLM, AutoImageProcessor

class CLIPChemistryDataset(Dataset):
    def __init__(self, limit=None, shape=(224, 224)):
        self.data = pd.read_parquet("hf://datasets/VuongQuoc/Chemistry_text_to_image/data/train-00000-of-00001-f1f5b2eab68f0d2f.parquet")
        if limit:
            self.data = self.data[:limit]
        # self.image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
        self.image_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-18")
        # self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image = self.data.loc[idx, 'image']['bytes']
        image = self._preprocess_image(image).squeeze(0) 
        label = self.data.loc[idx, 'text']
        input_ids, attention_mask = self._preprocess_text(label)
        return image, input_ids.squeeze(0), attention_mask.squeeze(0) 
    
    def _preprocess_image(self, bytes):
        image = Image.open(BytesIO(bytes))
        image_tensor = self.image_processor(image, 
            return_tensors="pt", 
            do_resize=True, 
            size={"height": 224, "width": 224})['pixel_values']
        return image_tensor
    
    def _preprocess_text(self, text):
        encoded_input = self.tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=256)
        return encoded_input['input_ids'], encoded_input['attention_mask']
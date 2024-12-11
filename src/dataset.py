from torch.utils.data import Dataset
from torchvision.transforms import Resize, Compose, ToTensor


import numpy as np
import pandas as pd

from io import BytesIO
from PIL import Image

from transformers import DistilBertTokenizer, AutoImageProcessor

class CLIPChemistry(Dataset):
    def __init__(self, limit=None, shape=(224, 224)):
        self.data = pd.read_parquet("hf://datasets/VuongQuoc/Chemistry_text_to_image/data/train-00000-of-00001-f1f5b2eab68f0d2f.parquet")
        if limit is not None:
            self.data = self.data[:limit]
        # Define the image transformations
        # self.transform = Compose([
        #     Resize(shape),  # Reshape the image
        #     ToTensor()      # Convert the image to PyTorch tensor
        # ])
        self.transform = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image = self.data.loc[idx, 'image']['bytes']
        image = self._preprocess_image(image)
        label = self.data.loc[idx, 'text']
        input_ids, attention_mask = self._preprocess_text(label)
        return image, input_ids, attention_mask
    
    def _preprocess_image(self, bytes):
        image = Image.open(BytesIO(bytes))
        image_tensor = self.transform(image, return_tensors='pt')['pixel_values']
        return image_tensor
    
    def _preprocess_text(self, text):
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        encoded_input = tokenizer(text, return_tensors='pt')
        return encoded_input['input_ids'], encoded_input['attention_mask']
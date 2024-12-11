from transformers import DistilBertModel
import torch
import torch.nn as nn

class TextEncoderHead(nn.Module):
    def __init__(self, model):
        super(TextEncoderHead, self).__init__()
        self.model = model
        self.layer = nn.Linear(768, 512)
        self.normalization = nn.LayerNorm(512)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        outputs = self.layer(outputs)
        return self.normalization(outputs)
    
class ImageEncoderHead(nn.Module):
    def __init__(self, model):
        super(ImageEncoderHead, self).__init__()
        self.model = model
        self.layer = nn.Linear(768, 512)
        self.normalization = nn.LayerNorm(512)
    
    def forward(self, pixel_values):
        outputs = self.model(pixel_values=pixel_values)
        outputs = self.layer(outputs)
        return self.normalization(outputs)
    
class CLIPChemistry(nn.Module):
    def __init__(self, text_encoder, image_encoder):
        super(CLIPChemistry, self).__init__()
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder

    def forward(self, image, input_ids, attention_mask):
        ie = self.image_encoder(image)
        te = self.text_encoder(input_ids, attention_mask)
        matrix = ie @ te.T
        
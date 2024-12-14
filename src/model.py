from transformers import DistilBertModel
import torch
import torch.nn as nn

class TextEncoderHead(nn.Module):
    def __init__(self, model):
        super(TextEncoderHead, self).__init__()
        self.model = model
        for param in self.model.parameters():
            param.requires_grad = False
        self.seq1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(767*256, 2000),
            nn.LayerNorm(2000),
            nn.ReLU(),
            nn.Linear(2000, 512),
            nn.LayerNorm(512)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        outputs = outputs.logits
        outputs = self.seq1(outputs)
        return outputs.contiguous()
    
class ImageEncoderHead(nn.Module):
    def __init__(self, model):
        super(ImageEncoderHead, self).__init__()
        self.model = model
        for param in self.model.parameters():
            param.requires_grad = False
        self.seq1 = nn.Sequential(
            nn.Linear(768, 512),
            nn.LayerNorm(512)
        )
    
    def forward(self, pixel_values):
        outputs = self.model(pixel_values=pixel_values)
        outputs = outputs.last_hidden_state.mean(dim=1)
        outputs = self.seq1(outputs)
        return outputs.contiguous()
    
class CLIPChemistryModel(nn.Module):
    def __init__(self, text_encoder, image_encoder):
        super(CLIPChemistryModel, self).__init__()
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder

    def forward(self, image, input_ids, attention_mask):
        # calculate the embeddings
        ie = self.image_encoder(image)
        te = self.text_encoder(input_ids, attention_mask)
        return ie, te

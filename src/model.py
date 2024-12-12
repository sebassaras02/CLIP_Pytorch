from transformers import DistilBertModel
import torch
import torch.nn as nn

class TextEncoderHead(nn.Module):
    def __init__(self, model):
        super(TextEncoderHead, self).__init__()
        self.model = model
        self.seq1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256*768, 512),
            nn.LayerNorm(512)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        outputs = outputs.last_hidden_state
        outputs = self.seq1(outputs)
        return outputs
    
class ImageEncoderHead(nn.Module):
    def __init__(self, model):
        super(ImageEncoderHead, self).__init__()
        self.model = model
        self.seq1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(197*768, 512),
            nn.LayerNorm(512)
        )
    
    def forward(self, pixel_values):
        outputs = self.model(pixel_values=pixel_values)
        outputs = outputs.last_hidden_state
        outputs = self.seq1(outputs)
        return outputs
    
class CLIPChemistryModel(nn.Module):
    def __init__(self, text_encoder, image_encoder):
        super(CLIPChemistryModel, self).__init__()
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder

    def forward(self, image, input_ids, attention_mask):
        # calculate the embeddings
        ie = self.image_encoder(image)
        te = self.text_encoder(input_ids, attention_mask)
        # normalize the results
        ie = ie / ie.norm(dim=-1, keepdim=True)
        te = te / te.norm(dim=-1, keepdim=True)
        # calculate the similarity matrix
        similarity_matrix = ie @ te.T
        labels = torch.eye(similarity_matrix.shape[0])
        return similarity_matrix, labels

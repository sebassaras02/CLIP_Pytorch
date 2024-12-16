import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

class TextEncoderHead(nn.Module):
    def __init__(self, model):
        super(TextEncoderHead, self).__init__()
        self.model = model
        for param in self.model.parameters():
            param.requires_grad = False
        # uncomment this for chemberta
        # self.seq1 = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(767*256, 2000),
        #     nn.Dropout(0.3),
        #     nn.ReLU(),
        #     nn.Linear(2000, 512),
        #     nn.LayerNorm(512)
        # )
        self.seq1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(768*256, 2000),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(2000, 512),
            nn.LayerNorm(512)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # uncomment this for chemberta
        # outputs = outputs.logits
        outputs = outputs.last_hidden_state
        outputs = self.seq1(outputs)
        return outputs.contiguous()
    
class ImageEncoderHead(nn.Module):
    def __init__(self, model):
        super(ImageEncoderHead, self).__init__()
        self.model = model
        for param in self.model.parameters():
            param.requires_grad = False
        # for resnet model
        # self.seq1 = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(512*7*7, 1000),
        #     nn.Linear(1000, 512),
        #     nn.LayerNorm(512)
        # )
        # for vit model
        self.seq1 = nn.Sequential(
            nn.Linear(768, 1000),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(1000, 512),
            nn.LayerNorm(512)
        )

    
    def forward(self, pixel_values):
        outputs = self.model(pixel_values)
        outputs = outputs.last_hidden_state.mean(dim=1)
        outputs = self.seq1(outputs)
        return outputs.contiguous()
    
class CLIPChemistryModel(nn.Module, PyTorchModelHubMixin):
    def __init__(self, text_encoder, image_encoder):
        super(CLIPChemistryModel, self).__init__()
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder

    def forward(self, image, input_ids, attention_mask):
        # calculate the embeddings
        ie = self.image_encoder(image)
        te = self.text_encoder(input_ids, attention_mask)
        return ie, te

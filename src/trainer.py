from dataset import CLIPChemistryDataset
from model import CLIPChemistryModel, TextEncoderHead, ImageEncoderHead
from torch.utils.data import DataLoader
from torch import manual_seed
from torch.optim import Adam

from torch.utils.data import random_split

import numpy as np
import pandas as pd

from transformers import DistilBertModel, ViTModel, AutoModelForMaskedLM, ResNetModel

from utils import contrastive_loss, trainer_fn

SEED = 99
np.random.seed(SEED)
manual_seed(SEED)

def main():
    # DEFINE DATASET AND DATALOADER

    print("Loading dataset...")
    dataset = CLIPChemistryDataset(limit=1000)
    print("Dataset loaded.")

    # Calculate split sizes
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    remaining_size = total_size - train_size
    val_size = int(0.5 * remaining_size)
    test_size = remaining_size - val_size
    
    # Split the dataset
    train, helper = random_split(dataset, [train_size, remaining_size])
    val, test = random_split(helper, [val_size, test_size])
    print("Split dataset into train, validation, and test.")

    # Create dataloaders
    dataloader_train = DataLoader(train, batch_size=64, shuffle=True)
    dataloader_val = DataLoader(val, batch_size=32, shuffle=False)
    dataloader_test = DataLoader(test, batch_size=32, shuffle=False)
    print("Data loaders created.")

    # DEFINE FOUNDATIONAL MODELS
    # ENCODER_BASE = DistilBertModel.from_pretrained("distilbert-base-uncased")
    ENCODER_BASE = AutoModelForMaskedLM.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
    IMAGE_BASE = ViTModel.from_pretrained("google/vit-base-patch16-224")
    # IMAGE_BASE = ResNetModel.from_pretrained("microsoft/resnet-18")
    text_encoder = TextEncoderHead(model=ENCODER_BASE)
    print("Text encoder created.")
    image_encoder = ImageEncoderHead(model=IMAGE_BASE)
    print("Image encoder created.")
    model = CLIPChemistryModel(text_encoder=text_encoder, image_encoder=image_encoder)
    print("CLIP Model created.")

    # DEFINE OPTIMIZER
    optimizer = Adam(model.parameters(), lr=3e-2)
    print("Adam optimizer created.")

    # DEFINE SCHEDULER
    print("Starting training...")
    trainer_fn(
        model=model,
        dataloader_train=dataloader_train,
        dataloader_val=dataloader_val,
        epochs=20,
        loss_fn=contrastive_loss,
        optimizer=optimizer,
        device="mps"
    )
    print("Training complete.")


if __name__ == '__main__':
    main()
from dataset import CLIPChemistryDataset
from model import CLIPChemistryModel, TextEncoderHead, ImageEncoderHead
from torch.utils.data import DataLoader
from torch import manual_seed
from torch.optim import Adam

from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd

from transformers import DistilBertModel, ViTModel

from utils import contrastive_loss, trainer_fn

SEED = 99
np.random.seed(SEED)
manual_seed(SEED)

def main():
    # DEFINE DATASET AND DATALOADER

    print("Loading dataset...")
    dataset = CLIPChemistryDataset(limit=10000)
    print("Dataset loaded.")
    train, helper = train_test_split(dataset, test_size=0.3, random_state=SEED)
    val, test = train_test_split(helper, test_size=0.5, random_state=SEED)
    print("Split dataset into train, validation, and test.")
    dataloader_train = DataLoader(train, batch_size=32, shuffle=True)
    dataloader_val = DataLoader(val, batch_size=32, shuffle=False)
    dataloader_test = DataLoader(test, batch_size=32, shuffle=False)
    print("Data loaders created.")

    # DEFINE FOUNDATIONAL MODELS
    ENCODER_BASE = DistilBertModel.from_pretrained("distilbert-base-uncased")
    IMAGE_BASE = ViTModel.from_pretrained("google/vit-base-patch16-224")
    text_encoder = TextEncoderHead(model=ENCODER_BASE)
    print("Text encoder created.")
    image_encoder = ImageEncoderHead(model=IMAGE_BASE)
    print("Image encoder created.")
    model = CLIPChemistryModel(text_encoder=text_encoder, image_encoder=image_encoder)
    print("CLIP Model created.")

    # DEFINE LOSS FUNCTION
    loss_fn = contrastive_loss()
    print("Contrastive loss function created.")

    # DEFINE OPTIMIZER
    optimizer = Adam(model.parameters(), lr=1e-2)
    print("Adam optimizer created.")

    # DEFINE SCHEDULER
    print("Starting training...")
    trainer_fn(
        model=model,
        dataloader_train=dataloader_train,
        dataloader_val=dataloader_val,
        epochs=10,
        loss_fn=loss_fn,
        optimizer=optimizer
    )
    print("Training complete.")


if __name__ == '__main__':
    main()
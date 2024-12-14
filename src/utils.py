import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

from tqdm import tqdm

import mlflow

from dotenv import load_dotenv

from datetime import datetime

import dagshub

from accelerator import Accelerator

load_dotenv('../.env')
dagshub.init(repo_owner='sebassaras02', repo_name='CLIP_Pytorch', mlflow=True)


def trainer_fn(model, dataloader_train, dataloader_val, epochs, loss_fn, optimizer):

    accelerator = Accelerator()
    model, optimizer, dataloader_train, dataloader_val = accelerator.prepare(model, optimizer, dataloader_train, dataloader_val)

    total_loss_train = []
    total_loss_val = []

    best_loss = float('inf')
    
    experiment_run_name = "Model " + str(datetime.now().strftime("%Y-%m-%d"))

    with mlflow.start_run(run_name=experiment_run_name) as run:
        
        model.to(device)

        for epoch in tqdm(range(epochs), desc="Training..."):
            # MODEL TRAINING 
            model.train()
            running_loss = 0
            counter = 0 
            for batch in dataloader_train:
                # Zero the gradients
                optimizer.zero_grad()
                image, input_ids, attention_mask = batch
                image = image.to(device)
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)

                # Forward pass
                image_embeddings, text_embeddings = model(image, input_ids, attention_mask)
                
                # Calculate the loss
                loss = loss_fn(image_embeddings, text_embeddings)
                # Backward pass
                loss.backward()
                # Optimize the weights
                optimizer.step()
                # Update the learning rate
                running_loss += loss.item()
                counter += 1
            running_loss = running_loss/counter
            total_loss_train.append(running_loss)

            # MODEL EVALUATION
            model.eval()
            running_vloss = 0
            vcounter = 0
            with torch.no_grad():
                for batch in  dataloader_val:
                    image, input_ids, attention_mask = batch
                    image = image.to(device)
                    input_ids = input_ids.to(device)
                    attention_mask = attention_mask.to(device)
                
                    # forward pass
                    image_embeddings, text_embeddings = model(image, input_ids, attention_mask)

                    # calculate the loss
                    loss = loss_fn(image_embeddings=image_embeddings, text_embeddings=text_embeddings)
                    running_vloss += loss.item()
                    vcounter += 1
                
            running_vloss = running_vloss/vcounter
            total_loss_val.append(running_vloss)

            # LOG THE METRICS
            metrics_epoch = {
                "train_loss": running_loss,
                "val_loss": running_vloss
            }
            mlflow.log_metrics(metrics_epoch, step=epoch)   

            # MODEL CHECKPOINT
            best_loss = model_checkpoint(model, best_loss, running_vloss)     

            # PRINT THE LOSS
            print(f"Epoch {epoch+1} - Train Loss: {running_loss} - Validation Loss: {running_vloss}")
        
        # LOG THE BEST MODEL
        mlflow.log_artifact("best_model.pth")

def model_checkpoint(model, best_loss, current_loss):
    if current_loss < best_loss:
        torch.save(model.state_dict(), "best_model.pth")
        return current_loss
    return best_loss


def contrastive_loss(image_embeddings, text_embeddings, temperature=0.5):
    """
    Compute contrastive loss between image and text embeddings.
    """
    # Mover la temperatura al dispositivo y convertir a float
    temperature = torch.tensor(temperature, device=image_embeddings.device).float()
    
    # Asegurarse de que los tensores sean contiguos
    image_embeddings = image_embeddings.contiguous().float()
    text_embeddings = text_embeddings.contiguous().float()
    
    # Obtener el tamaño del lote
    batch_size = image_embeddings.shape[0]
    
    # Normalizar los embeddings
    image_embeddings = F.normalize(image_embeddings, p=2, dim=1)
    text_embeddings = F.normalize(text_embeddings, p=2, dim=1)
    
    # Calcular la similitud del coseno
    logits = torch.einsum('nc,mc->nm', [image_embeddings, text_embeddings])
    
    # Aplicar escalado por temperatura
    logits = logits * torch.exp(temperature)
    
    # Crear etiquetas (matriz diagonal)
    labels = torch.arange(batch_size, device=image_embeddings.device)
    
    # Calcular la pérdida en ambas direcciones
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.t(), labels)
    
    # Retornar la media de ambas direcciones
    loss = (loss_i2t + loss_t2i) / 2
    
    return loss
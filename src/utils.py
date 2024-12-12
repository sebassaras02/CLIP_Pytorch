import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

def trainer_fn(model, dataloader_train, dataloader_val, epochs, loss_fn, optimizer, device):

    total_loss_train = []
    total_loss_val = []

    model.to(device)

    for epoch in tqdm(range(epochs), desc="Training..."):
        # MODEL TRAINING 
        model.train()
        running_loss = 0
        counter = 0 
        for batch in dataloader_train:
            image, input_ids, attention_mask = batch
            image = image.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            
            # Forward pass
            image_embeddings, text_embeddings = model(image, input_ids, attention_mask)
            
            # Calculate the loss
            loss = loss_fn(image_embeddings, text_embeddings)
            print(loss)
            # Backward pass
            loss.backward()
            # Optimize the weights
            optimizer.step()
            # Zero the gradients
            optimizer.zero_grad()
            # Update the learning rate
            running_loss += loss.item()
            counter += 1
            print(counter)
        total_loss_train.append(running_loss/counter)

        # MODEL EVALUATION
        model.eval()
        running_vloss = 0
        vcounter = 0
        with torch.no_grad():
            for batch in dataloader_val:
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
        total_loss_val.append(running_vloss/vcounter)

        # PRINT THE LOSS
        print(f"Epoch {epoch+1} - Train Loss: {total_loss_train[-1]} - Validation Loss: {total_loss_val[-1]}")


def contrastive_loss(image_embeddings, text_embeddings, temperature=1.0):
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
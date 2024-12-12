import torch
import torch.nn as nn
from accelerate import Accelerator

def trainer(model, dataloader_train, dataloader_val, epochs, loss_fn, optimizer, scheduler):

    accelerator = Accelerator()
    dataloader_train, dataloader_val, model, optimizer, scheduler = accelerator.prepare(
         dataloader_train, dataloader_val, model, optimizer, scheduler)
    total_loss_train = []
    total_loss_val = []
    for epoch in range(epochs):
        # MODEL TRAINING 
        model.train()
        running_loss = 0
        counter = 0 
        for image, input_ids, attention_mask in dataloader_train:
            # Zero the gradients
            optimizer.zero_grad()
            # Forward pass
            similarity_matrix, labels = model(image, input_ids, attention_mask)
            # Calculate the loss
            loss = loss_fn(similarity_matrix, labels)
            # Backward pass
            accelerator.backward(loss)
            # Optimize the weights
            optimizer.step()
            # Update the learning rate
            running_loss += loss.item()
            counter += 1
        scheduler.step()
        total_loss_train.append(running_loss/counter)

        # MODEL EVALUATION
        model.eval()
        running_vloss = 0
        vcounter = 0
        with torch.no_grad():
            for image, input_ids, attention_mask in dataloader_val:
                # forward pass
                similarity_matrix, labels = model(image, input_ids, attention_mask)
                # calculate the loss
                loss = loss_fn(similarity_matrix, labels)
                running_vloss += loss.item()
                vcounter += 1
        total_loss_val.append(running_vloss/vcounter)


def contrastive_loss(similarity_matrix, labels):
    # Cross-entropy loss over the similarity matrix, with labels indicating the correct pair
    return nn.CrossEntropyLoss()(similarity_matrix, labels)

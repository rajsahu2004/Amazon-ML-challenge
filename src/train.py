import torch
from torch import optim
from model import EntityExtractorModel
import torch.nn as nn
from tqdm import tqdm
from constants import allowed_units
from dataset import AmazonMLDataset

def train_model(model, train_loader, optimizer, mse_loss, cross_entropy_loss, num_epochs=10):
    model.train()  # Set the model to training mode
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, numeric_values, unit_labels in tqdm(train_loader):
            optimizer.zero_grad()  # Zero out gradients
            
            # Forward pass
            predicted_values, predicted_units = model(images)
            
            # Compute losses
            loss_value = mse_loss(predicted_values, numeric_values)
            loss_unit = cross_entropy_loss(predicted_units, unit_labels)
            total_loss = loss_value + loss_unit  # Total loss
            
            # Backward pass and optimization
            total_loss.backward()
            optimizer.step()
            
            running_loss += total_loss.item()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}")


dataset = AmazonMLDataset('dataset/train.csv', 'images', transform=None)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
num_units = len(allowed_units)
model = EntityExtractorModel(num_units)
optimizer = optim.Adam(model.parameters(), lr=0.001)
train_model(model, train_loader, optimizer, nn.MSELoss(), nn.CrossEntropyLoss(), num_epochs=10)
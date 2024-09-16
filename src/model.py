import torch
import torch.nn as nn
from torchvision import models

class EntityExtractorModel(nn.Module):
    def __init__(self, num_units):
        super(EntityExtractorModel, self).__init__()
        # Load a pre-trained ResNet model
        self.cnn = models.resnet50(pretrained=True)
        self.cnn.fc = nn.Identity()  # Remove the fully connected layer, we’ll define our own
        
        # Fully connected layers
        self.fc1 = nn.Linear(2048, 512)  # Assuming ResNet50 output size
        self.fc2 = nn.Linear(512, 256)
        
        # Two outputs: one for numeric value, one for unit classification
        self.value_output = nn.Linear(256, 1)  # Regression for numeric value
        self.unit_output = nn.Linear(256, num_units)  # Classification for unit
        
    def forward(self, x):
        x = self.cnn(x)  # Extract image features using ResNet
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        
        # Output for numeric value (regression)
        value = self.value_output(x)
        
        # Output for unit (classification)
        unit = self.unit_output(x)
        
        return value, unit
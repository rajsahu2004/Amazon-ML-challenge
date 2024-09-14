import os
import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image

class AmazonMLDataset(Dataset):
    def __init__(self, csv_file, images_dir, transform=None):
        """
        Args:
            csv_file (str): Path to the CSV file with annotations (entity names and values).
            images_dir (str): Directory with all the downloaded images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_frame = pd.read_csv(csv_file)
        self.images_dir = images_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Extract image filename from URL
        img_url = self.data_frame.iloc[idx, 0]
        img_name = os.path.basename(img_url)
        
        # Load image from local directory
        img_path = os.path.join(self.images_dir, img_name)
        
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return None

        # Apply transformations to image if provided
        if self.transform:
            img = self.transform(img)

        # Extract metadata (entity name and value)
        entity_name = self.data_frame.iloc[idx, 2]
        entity_value = self.data_frame.iloc[idx, 3]

        # Convert entity value to a numeric format
        entity_value = self._convert_entity_value(entity_name, entity_value)

        # Return both the image and the associated value
        return img, entity_value

    def _convert_entity_value(self, entity_name, entity_value):
        """
        Convert the entity value to a numeric value (e.g., grams, milliliters).
        """
        if entity_name == "item_weight":
            if "milligram" in entity_value:
                value_in_grams = float(entity_value.split()[0]) / 1000.0  # Convert milligrams to grams
            elif "gram" in entity_value:
                value_in_grams = float(entity_value.split()[0])
            else:
                value_in_grams = float(entity_value)  # Default fallback
            return value_in_grams
        elif entity_name == "item_volume":
            if "cup" in entity_value:
                value_in_milliliters = float(entity_value.split()[0]) * 240  # Convert cups to milliliters
            elif "milliliter" in entity_value:
                value_in_milliliters = float(entity_value.split()[0])
            else:
                value_in_milliliters = float(entity_value)  # Default fallback
            return value_in_milliliters
        return float(entity_value)
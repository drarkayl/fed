from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import logging
import io

from .image_preprocessing import PetDataset, get_data_transforms

# Constants
TARGET_IMG_SIZE = (1024, 1360)
NUM_CLASSES = 2  # foxy and puppy
DATA_DIR = 'ml_model/data'  # Directory where the data is stored
TRAIN_DIR = f'{DATA_DIR}/train'
VAL_DIR = f'{DATA_DIR}/test'

# Get a logger for this module
logger = logging.getLogger(__name__)


class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(32 * (TARGET_IMG_SIZE[0] // 4) * (TARGET_IMG_SIZE[1] // 4), num_classes)  # Adjusted FC layer

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc(x)
        return x


def create_model(num_classes):
    """
    Creates the CNN model.
    """
    model = SimpleCNN(num_classes)
    logger.info("Model created successfully.")

    return model

def create_datasets(train_dir, val_dir, img_size):
    """
    Creates datasets for training and validation.
    """
    data_transforms = get_data_transforms(img_size)
    # Create datasets for training and validation
    train_dataset = PetDataset(root_dir=train_dir, transform=data_transforms['train'])
    val_dataset = PetDataset(root_dir=val_dir, transform=data_transforms['validation'])
    logger.info(f"The classes are: {train_dataset.class_to_idx}")
    return train_dataset, val_dataset

def create_dataloaders(train_dataset, val_dataset, batch_size=32):
    """
    Creates DataLoaders for training and validation datasets.
    """
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    logger.info(f"The classes are: {train_dataset.class_to_idx}")
    logger.info("Data loaders created successfully.", extra={'batch_size': batch_size})

    return train_loader, val_loader


def train_model(model, train_loader, val_loader, epochs, device="cpu"):
    """
    Trains the model for a specified number of epochs.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    model.to(device)  # Move model to the specified device

    for epoch in range(epochs):
        # --- Training Phase ---
        model.train()  # Set model to training mode
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # Move data to the specified device
            optimizer.zero_grad()  # Zero the parameter gradients
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, labels)  # Calculate the loss
            loss.backward()  # Backward pass
            optimizer.step()  # Optimize

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        # --- Validation Phase ---
        model.eval()  # Set model to evaluation mode
        val_loss = 0.0
        corrects = 0
        with torch.no_grad():  # Disable gradient calculation
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)

                _, preds = torch.max(outputs, 1)
                corrects += torch.sum(preds == labels.data)

        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_acc = corrects.double() / len(val_loader.dataset)
        logger.info(f'Epoch {epoch + 1}/{epochs} Train Loss: {epoch_loss:.4f} Val Loss: {epoch_val_loss:.4f} Val Acc: {epoch_val_acc:.4f}')

    return model, epoch_val_acc, epoch_val_loss


def serialize_model(model):
    """
    Serializes the model to a byte stream.
    """
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.getvalue()


def deserialize_model(model, model_bytes, device="cpu"):
    """
    Deserializes the model from a byte stream.
    """
    buffer = io.BytesIO(model_bytes)
    model.load_state_dict(torch.load(buffer, map_location=torch.device(device)))
    return model


def perform_inference(model, image_data, device="cpu"):
    """
    Performs inference on the given image data.
    """
    model.eval()  # Set the model to evaluation mode
    model.to(device)
    img = Image.open(io.BytesIO(image_data)).convert('RGB')
    transform = get_data_transforms(TARGET_IMG_SIZE)['validation']
    # single image with shape [C, H, W] becomes a mini-batch of one with shape [1, C, H, W], 
    # which is the format the model's forward pass expects.
    img_tensor = transform(img).unsqueeze(0).to(device)  # Add batch dimension and move to device

    with torch.no_grad():  # Disable gradient calculation
        output = model(img_tensor)
        probabilities = torch.softmax(output, dim=1)  # Convert to probabilities
        _, predicted = torch.max(probabilities, 1)  # Get the predicted class

    return predicted.item(), probabilities[0][predicted.item()].item()

#def main():

    # model = setup_model()
    # train_loader, val_loader = setup_loaders()

    # logger.info("Ready to start model training...")


    # # Train the model
    # trained_model = train_model(model, train_loader, val_loader, epochs=1, device="cpu")

    # # Serialize the model
    # model_bytes = serialize_model(trained_model)
    # logger.info(f"Serialized model size: {len(model_bytes)} bytes")

    # # Deserialize the model
    # deserialized_model = deserialize_model(create_model(NUM_CLASSES), model_bytes, device="cpu")
    # logger.info(f"Deserialized the model.")

    # # Example Inference
    # with open(f"{DATA_DIR}/test/foxy/test.jpg", 'rb') as f:
    #     image_data = f.read()

    # # predicted_class, confidence = perform_inference(deserialized_model, image_data, device="cpu")
    # predicted_class, confidence = perform_inference(deserialized_model, image_data, device="cpu")
    # logger.info(f"DESERIALIZED MODEL Predicted Class: {predicted_class}, Confidence: {confidence:.4f}")

    # predicted_class, confidence = perform_inference(model, image_data, device="cpu")
    # logger.info(f"Predicted Class: {predicted_class}, Confidence: {confidence:.4f}")

    # logger.info("Model serialization, deserialization, and inference test completed.")
# if __name__ == "__main__":
    
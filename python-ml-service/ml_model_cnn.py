import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import logging
import io

from image_preprocessing import PetDataset, get_data_transforms

# Constants
TARGET_IMG_SIZE = (1024, 1360)
NUM_CLASSES = 2  # foxy and puppy

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
    return model


def create_dataloaders(train_dir, val_dir, img_size, batch_size):
    """
    Creates DataLoaders for training and validation datasets.
    """
    data_transforms = get_data_transforms(img_size)

    train_dataset = PetDataset(root_dir=train_dir, transform=data_transforms['train'])
    val_dataset = PetDataset(root_dir=val_dir, transform=data_transforms['validation'])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def train_model(model, train_loader, val_loader, epochs, device="cpu"):
    """
    Trains the model for a specified number of epochs.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    model.to(device)  # Move model to the specified device

    for epoch in range(epochs):
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
        logger.info(f'Epoch {epoch + 1}/{epochs} Loss: {epoch_loss:.4f}')

    return model


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
    img_tensor = transform(img).unsqueeze(0).to(device)  # Add batch dimension and move to device

    with torch.no_grad():  # Disable gradient calculation
        output = model(img_tensor)
        probabilities = torch.softmax(output, dim=1)  # Convert to probabilities
        _, predicted = torch.max(probabilities, 1)  # Get the predicted class

    return predicted.item(), probabilities[0][predicted.item()].item()


if __name__ == '__main__':
    # Example Usage (can be removed or commented out later)
    # Set your data directories
    train_dir = 'data/train'
    val_dir = 'data/test'

    # Create data loaders
    train_loader, val_loader = create_dataloaders(train_dir, val_dir, TARGET_IMG_SIZE, batch_size=32)

    # Create the model
    model = create_model(NUM_CLASSES)

    # Train the model
    trained_model = train_model(model, train_loader, val_loader, epochs=1, device="cpu")

    # Serialize the model
    model_bytes = serialize_model(trained_model)
    logger.info(f"Serialized model size: {len(model_bytes)} bytes")

    # Deserialize the model
    deserialized_model = deserialize_model(create_model(NUM_CLASSES), model_bytes, device="cpu")

    # Example Inference
    # Assuming you have an image file 'sample_image.jpg' in the same directory
    with open("data/test/foxy/depositphotos_24988471-stock-illustration-cartoon-fox.jpg", 'rb') as f:
        image_data = f.read()

    predicted_class, confidence = perform_inference(deserialized_model, image_data, device="cpu")
    logger.info(f"Predicted Class: {predicted_class}, Confidence: {confidence:.4f}")

    logger.info("Model serialization, deserialization, and inference test completed.")
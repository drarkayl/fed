import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class PetDataset(Dataset):
    """
    Custom Dataset class for loading pet images.
    """

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.classes = os.listdir(root_dir)  # setting number of class based on directories present in root_dir
        self.image_paths = []
        self.labels = []

        for idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(self.root_dir, class_name)
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                self.image_paths.append(img_path)
                self.labels.append(idx)  # Assign numerical labels based on folder order

        self.class_to_idx = {class_name: i for i, class_name in enumerate(self.classes)} # create mapping of class to label

    def __len__(self):
        # Return the total number of samples in the dataset
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load and preprocess an image given its index
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')  # Ensure all images are RGB
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


def get_data_transforms(img_size):
    """
    Define data transformations for training and validation.
    """
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
        ]),
        'validation': transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
        ]),
    }

    return data_transforms
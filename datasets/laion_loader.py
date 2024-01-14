import os
import torch
from PIL import Image, ImageFile
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from diffusers.utils.torch_utils import randn_tensor

ImageFile.LOAD_TRUNCATED_IMAGES = True
class AestheticDataset(torch.utils.data.Dataset):

    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.file_pairs = self._get_file_pairs()

    def _get_file_pairs(self):
        file_pairs = []
        for subdir, _, files in os.walk(self.root_dir):
            images = [f for f in files if f.endswith('.jpg')]
            for image in images:
                basename = os.path.splitext(image)[0]
                text_file = basename + '.txt'
                if text_file in files:
                    image_path = os.path.join(subdir, image)
                    text_path = os.path.join(subdir, text_file)
                    file_pairs.append((image_path, text_path))
        return file_pairs

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, idx):
        image_path, text_path = self.file_pairs[idx]
        # Load and transform image
        image = Image.open(image_path).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize((320, 576)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        pixel_values = transform(image)
        # Load text prompt
        with open(text_path, 'r', encoding='utf-8') as f:
            text_prompt = f.read().strip()
        # Create condition
        condition = (pixel_values + torch.randn_like(pixel_values) * 0.02)
        
        return {"pixel_values": pixel_values.unsqueeze(0), "text_prompt": text_prompt, "condition": condition.unsqueeze(0)}


def visualize_data(dataloader):
    for data in dataloader:
        pixel_values = data["pixel_values"][0][0].permute(
            1, 2, 0)  # Convert to HWC format for matplotlib
        text_prompt = data["text_prompt"][0]

        plt.imshow(pixel_values)
        plt.axis('off')
        plt.text(0.5, 1.05, text_prompt, ha='center',
                 va='center', transform=plt.gca().transAxes)
        plt.savefig('test.jpg')
        break  # Just show one data point for demonstration


if __name__ == '__main__':
    # Create the dataset and dataloader
    # Replace with the path to your dataset
    root_dir = "/18940970966/laion-high-aesthetics-output"
    iterable_dataset = AestheticDataset(root_dir)
    dataloader = torch.utils.data.DataLoader(iterable_dataset, batch_size=4)

    # Visualize one data point
    visualize_data(dataloader)

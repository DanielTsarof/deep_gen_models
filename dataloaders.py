import os
import random

import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset


class FacadesDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        :param root_dir: dataset path ('facades/train')
        :param transform: transformations
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [os.path.join(root_dir, fname)
                            for fname in os.listdir(root_dir) if fname.endswith('.jpg')]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        # Split images in half
        w, h = image.size
        w2 = w // 2
        # The left part is a photo of the facade, the right is a mask.
        photo = image.crop((0, 0, w2, h))
        mask = image.crop((w2, 0, w, h))

        if self.transform:
            photo = self.transform(photo)
            mask = self.transform(mask)
        else:
            photo = transforms.ToTensor()(photo)
            mask = transforms.ToTensor()(mask)

        return mask, photo


class UnpairedFacadesDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        :param root_dir: images dir ('facades/train')
        :param transform: image transforms
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [os.path.join(root_dir, fname)
                            for fname in os.listdir(root_dir) if fname.endswith('.jpg')]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        # For domain A (masks), we use an image with the index index_A
        index_A = index % len(self.image_paths)
        # For domain B (photo), select a random image
        index_B = random.randint(0, len(self.image_paths) - 1)

        # Upload the image for domain A and extract the right half (mask)
        img_A = Image.open(self.image_paths[index_A]).convert("RGB")
        w, h = img_A.size

        img_A = img_A.crop((w // 2, 0, w, h))

        # Upload the image for domain B and extract the left half (photo)
        img_B = Image.open(self.image_paths[index_B]).convert("RGB")
        w, h = img_B.size
        img_B = img_B.crop((0, 0, w // 2, h))

        if self.transform:
            img_A = self.transform(img_A)
            img_B = self.transform(img_B)
        else:
            img_A = transforms.ToTensor()(img_A)
            img_B = transforms.ToTensor()(img_B)

        return img_A, img_B

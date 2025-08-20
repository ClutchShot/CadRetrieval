import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from retrieval.vector_db import VectorDatabase
from retrieval.metrics import calculate_map
import pathlib
from utils.read import get_filenames_by_type_and_key, get_samples, get_all_png_filenames, filter_list_by_set
from utils.safe_results import safe_raw_results, safe_by_class_results, safe_total_results
from autoencoder.cnn import CNNAutoencoder
from autoencoder.vit import ViTAutoencoder, StackedImageViTAutoencoder
import random
from sklearn.model_selection import train_test_split
from typing import List, Union, Callable, Optional
import time
from datasets.util import filter_classes_by_min_samples
import gzip
import torch.multiprocessing as mp
from datasets.util import files_load

class ImageDataset(Dataset):
    def __init__(self, image_paths, labels=None, names=None):
        """
        Args:
            image_paths (list): List of paths to image files
            labels (list, optional): Corresponding labels
            transform (callable, optional): Transformations to apply
        """
        self.image_paths = image_paths
        self.labels = labels
        self.names = names
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')

        # Apply transformations
        image = self.transform(image)

        # Get label if available
        if self.labels is not None:
            label = self.labels[idx]
            return image, label, self.names[idx]
        else:
            return image


class MultiImageDataset(Dataset):
    def __init__(self, data_paths: pathlib.Path, labels: list, names: list):
        """
        Args:
            image_paths (list): List of paths to image files
            labels (list, optional): Corresponding labels
            transform (callable, optional): Transformations to apply
        """
        self.data_paths = data_paths
        self.labels = labels
        self.names = names
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
        ])

        self.views = ["front", "back", "left", "right", "top", "bottom",
                      "isometric1", "isometric2", "isometric3", "isometric4",
                      "isometric5", "isometric6", "isometric7", "isometric8"]

        # self.views_polarity = [
        #     f"{view}_neg" for view in self.views] + [f"{view}_pos" for view in self.views]

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        # Load image

        vectors = []
        for view in self.views:
            image_path = self.data_paths / f"{self.names[idx]}_{view}.png"
            image = Image.open(image_path).convert('RGB')

            # Apply transformations
            vectors.append(self.transform(image))

        k_vector = torch.cat(vectors, dim=0)

        # Randomly shuffle the image projections
        # random.shuffle(vectors)
        # k_vector = torch.stack(vectors, dim=0)

        return k_vector, self.labels[idx], self.names[idx]


class BinaryDataset(Dataset):
    """Dataset class for loading binary files (.bin, .pt, .npy, etc.)"""

    def __init__(self, data_paths: pathlib.Path, labels: list, compressed: bool = False):
        """
        Args:
            data_dir: Directory containing binary files
        """
        self.data_dir = data_paths
        self.labels = labels
        self.compressed = compressed

    def __len__(self) -> int:
        """Return number of binary files"""
        return len(self.data_dir)

    def __getitem__(self, idx: int):
        """Load and return binary data at index"""
        file_path = self.data_dir[idx]
        label = self.labels[idx]
        name = self.data_dir[idx].stem
        try:
            # Load binary data
            if self.compressed:
                with gzip.open(file_path, 'rb') as f:
                    data = torch.load(f)
            else:
                data = torch.load(file_path)
            # idx = torch.randperm(14)
            # data = data[idx]
            # data = x_shuffled.reshape(14 * 3, 32, 32)
            return data, label, name

        except Exception as e:
            raise RuntimeError(f"Error loading {file_path}: {e}")


def collate_fn(batch):
    if len(batch[0]) == 3:  # image, label, name
        images, labels, names = zip(*batch)
        return torch.stack(images), list(labels), list(names)
    else:  # image, name
        images, names = zip(*batch)
        return torch.stack(images), list(names)


def save_for_inference(model, filepath, input_shape=None):
    """
    Save model optimized for inference
    """
    model.eval()  # Set to evaluation mode

    filepath = filepath / "best.pt"

    save_dict = {
        'model_state_dict': model.state_dict(),
        'model_class': model.__class__.__name__,
        'input_shape': input_shape,
    }

    torch.save(save_dict, filepath)
    print(f"Model saved for inference: {filepath}")


# Example usage
if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    # Initialize model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNAutoencoder(
        latent_dim=1024, in_channels=14*3, image_size=512).to(device)
    # model = ViTAutoencoder(image_size=512, patch_size=64, latent_dim=512).to(device)
    # model = StackedImageViTAutoencoder(num_images=14, image_size=32, latent_dim=512).to(device)
    model.print_model_summary()

    # root_dir = "./data/SolidLetters/"
    root_dir = "./data/Fabwave/"
    root_dir = pathlib.Path(root_dir)
    out_dim = 1024

    db_path = "./vector_db/autoencoder_cnn_Fabwave30x14x1024xrotate"
    dataset = "Fabwave"
    # hour_min_second = time.strftime("%H%M%S")
    # results_path = pathlib.Path(f"./results/CNN/{hour_min_second}")
    # if not results_path.exists():
    #     results_path.mkdir(parents=True, exist_ok=True)

    # files = [file for file in root_dir.rglob(f"*_stack32x32.bin")]
    # labels = [file.parent.parent.name  for file in files]

    # print(f"Found {len(files)} files")

    # files, labels  = filter_classes_by_min_samples(files, labels)

    # print(f"Valid files {len(files)} files")

    # train_data, test_data, train_label, test_label = train_test_split(files, labels, test_size=0.2, stratify=labels)

    # train_dataset = BinaryDataset(train_data, train_label)
    # train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

    # test_dataset = BinaryDataset(test_data, test_label)
    # test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)


    # Fabwave
    files_all = [file for file in root_dir.rglob(f"*_cat512x512.pt.gz")]
    files = []
    for file in files_all:
        if "_standart_" not in file.name:
            files.append(file)
    labels = [file.parent.parent.name  for file in files]

    files, labels  = filter_classes_by_min_samples(files, labels)

    train_data, test_data, train_label, test_label =  train_test_split(files, labels, test_size=0.2, random_state=42, stratify=labels)
    train_names = [str(name.stem) for name in train_data]
    test_names = [str(name.stem) for name in test_data]

    train_dataset = BinaryDataset(train_data, train_label, compressed=True)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn,
                              num_workers=8, multiprocessing_context=mp.get_context('spawn'), pin_memory=True,
                              prefetch_factor=2, persistent_workers=True)

    test_dataset = BinaryDataset(test_data, test_label, compressed=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn,
                              num_workers=4, multiprocessing_context=mp.get_context('spawn'), pin_memory=True,
                              prefetch_factor=2, persistent_workers=True)


    # SolidLetters bin
    # data_path = root_dir / "img_bin_28_rotate_512"

    # files = [file for file in root_dir.rglob(f"*.pt.gz")]
    # valid_samples = [ str(file.name).split("_cat512x512")[0]  for file in files]


    # train_samples = get_samples(root_dir, "train.txt")
    # test_samples = get_samples(root_dir, "test.txt")

    # train_samples = filter_list_by_set(train_samples, valid_samples)
    # test_samples = filter_list_by_set(test_samples, valid_samples)

    # train_label = [name.split("_")[0] for name in train_samples]
    # test_label = [name.split("_")[0] for name in test_samples]

    # train_data = [
    #     data_path / f"{file_name}_cat512x512.pt.gz" for file_name in train_samples]
    # test_data = [data_path /
    #              f"{file_name}_cat512x512.pt.gz" for file_name in test_samples]

    # train_dataset = BinaryDataset(train_data, train_label, compressed=True)
    # train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn,
    #                           num_workers=8, multiprocessing_context=mp.get_context('spawn'), pin_memory=True,
    #                           prefetch_factor=2, persistent_workers=True)

    # test_dataset = BinaryDataset(test_data, test_label, compressed=True)
    # test_loader = DataLoader(test_dataset, batch_size=16,
    #                          shuffle=False, collate_fn=collate_fn,
    #                           num_workers=8, multiprocessing_context=mp.get_context('spawn'), pin_memory=True,
    #                           prefetch_factor=2, persistent_workers=True)

    # Multi Image
    # data_path = root_dir / "png_random_pos_neg"

    # valid_samples = get_all_png_filenames(data_path)

    # train_samples = get_samples(root_dir, "train.txt")
    # test_samples = get_samples(root_dir, "test.txt")

    # train_samples = filter_list_by_set(train_samples, valid_samples)
    # test_samples = filter_list_by_set(test_samples, valid_samples)

    # train_label = [name.split("_")[0] for name in train_samples]
    # test_label = [name.split("_")[0] for name in test_samples]

    # train_dataset = MultiImageDataset(data_path, train_label, train_samples)
    # train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn,
    #                           num_workers=8, multiprocessing_context=mp.get_context('spawn'),
    #                           prefetch_factor=2, persistent_workers=True)

    # test_dataset = MultiImageDataset(data_path, test_label, test_samples)
    # test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn,
    #                           num_workers=8, multiprocessing_context=mp.get_context('spawn'),
    #                           prefetch_factor=2, persistent_workers=True)

    # Single image
    # train_png_paths = get_filenames_by_type_and_key(root_dir, "train.txt", ".png", "_bottom")
    # test_png_paths = get_filenames_by_type_and_key(root_dir, "test.txt", ".png", "_bottom")

    # train_names = [path.stem for path in train_png_paths]
    # train_label = [name.split("_")[0] for name in train_names]

    # test_names = [path.stem for path in test_png_paths]
    # test_label = [name.split("_")[0] for name in test_names]

    # train_dataset = ImageDataset(train_png_paths, train_label, train_names)
    # train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

    # test_dataset = ImageDataset(test_png_paths, test_label, test_names)
    # test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(40):
        # Assume batch shape: [B, 3, 224, 224]
        epoch_loss = 0.0
        for images, labels, names in tqdm(train_loader, desc="Training"):
            batch = images.to(device, non_blocking=True)
            optimizer.zero_grad()

            try:
                recon_batch, z = model(batch)
            except:
                continue
            loss = criterion(recon_batch, batch)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch}, Avg loss: {avg_loss:.4f}")

    vec_db = VectorDatabase(db_path, dataset, out_dim)
    if vec_db.get_vector_count() == 0:
        # Get train embendings
        model.eval()
        with torch.no_grad():
            for images, labels, names in tqdm(train_loader, desc="Infernce train data"):
                batch = images.to(device, non_blocking=True)
                out = model.encode(batch).cpu().numpy()
                vec_db.add_vectors(vectors=out, names=names,
                                   labels=labels, duplicates=True)

    queries = []
    retrieval_all = []
    # Eval
    model.eval()
    with torch.no_grad():
        for images, labels, names in tqdm(test_loader, desc="Eval"):
            batch = images.to(device, non_blocking=True)
            out = model.encode(batch).cpu().numpy()
            vec_db.add_vectors(vectors=out, names=names,
                               labels=labels, duplicates=True)

            retrieval_topk = vec_db.search(out, k=7)
            retrieval_all.extend(retrieval_topk)
            for name, label in zip(names, labels):
                queries.append({"name": name, "label": label})

    map_score, detailed = calculate_map(queries, retrieval_all)

    safe_raw_results(db_path, detailed)
    df_grouped = safe_by_class_results(db_path, detailed)
    safe_total_results(db_path, map_score, df_grouped)

    # save_for_inference(model, results_path)

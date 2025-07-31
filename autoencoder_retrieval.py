import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm
from retrieval.vector_db import VectorDatabase
from retrieval.metrics import calculate_map
import pathlib
from utils.read_txt import get_filenames_by_type_and_key, get_samples, get_all_png_filenames, filter_list_by_set
from utils.safe_results import safe_raw_results, safe_by_class_results, safe_total_results
from datasets.util import files_load
from sklearn.model_selection import train_test_split
from autoencoder.cnn import CNNAutoencoder
from autoencoder.vit import ViTAutoencoder


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
    def __init__(self, data_paths: pathlib.Path, labels : list, names : list):
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

        return k_vector, self.labels[idx], self.names[idx]



def collate_fn(batch):
    if len(batch[0]) == 3:  # image, label, name
        images, labels, names = zip(*batch)
        return torch.stack(images), list(labels), list(names)
    else:  # image, name
        images, names = zip(*batch)
        return torch.stack(images), list(names)


# Example usage
if __name__ == "__main__":
    # Initialize model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = CNNAutoencoder(latent_dim=1024, in_channels=14*3).to(device)
    model = ViTAutoencoder(image_size=512, patch_size=64, latent_dim=512).to(device)
    model.print_model_summary()

    root_dir = "./data/SolidLetters/"
    root_dir = pathlib.Path(root_dir)
    out_dim = 512

    db_path = "./vector_db/autoencoder_vit_solidletters10x1x512"
    dataset = "solidletters"

    # data_path = root_dir / "png"

    # valid_samples = get_all_png_filenames(data_path)

    # train_samples = get_samples(root_dir, "train.txt")
    # test_samples = get_samples(root_dir, "test.txt")

    # train_samples = filter_list_by_set(train_samples, valid_samples)
    # test_samples = filter_list_by_set(test_samples, valid_samples)

    # train_label = [name.split("_")[0] for name in train_samples]
    # test_label = [name.split("_")[0] for name in test_samples]

    
    # train_dataset = MultiImageDataset(data_path, train_label, train_samples)
    # train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

    # test_dataset = MultiImageDataset(data_path, test_label, test_samples)
    # test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

    train_png_paths = get_filenames_by_type_and_key(root_dir, "train.txt", ".png", "_bottom")
    test_png_paths = get_filenames_by_type_and_key(root_dir, "test.txt", ".png", "_bottom")

    train_names = [path.stem for path in train_png_paths]
    train_label = [name.split("_")[0] for name in train_names]

    test_names = [path.stem for path in test_png_paths]
    test_label = [name.split("_")[0] for name in test_names]

    train_dataset = ImageDataset(train_png_paths, train_label, train_names)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

    test_dataset = ImageDataset(test_png_paths, test_label, test_names)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(10):
        # Assume batch shape: [B, 3, 224, 224]
        epoch_loss = 0.0
        for images, labels, names in tqdm(train_loader, desc="Training"):
            batch = images.to(device)
            optimizer.zero_grad()

            recon_batch, z = model(batch)
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
                batch = images.to(device)
                out = model.encode(batch).cpu().numpy()
                vec_db.add_vectors(vectors=out, names=names,
                                   labels=labels, duplicates=True)

    queries = []
    retrieval_all = []
    # Eval
    model.eval()
    with torch.no_grad():
        for images, labels, names in tqdm(test_loader, desc="Eval"):
            batch = images.to(device)
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

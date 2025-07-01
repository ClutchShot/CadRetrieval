import random
from tqdm import tqdm
import numpy as np
import torch
from scipy.spatial.transform import Rotation
import hashlib
from collections import defaultdict
import pathlib
import glob
import os
import torch
from dgl import DGLHeteroGraph
from dgl.data.utils import load_graphs
from sklearn.model_selection import train_test_split

def bounding_box_uvgrid(inp: torch.Tensor):
    pts = inp[..., :3].reshape((-1, 3))
    mask = inp[..., 6].reshape(-1)
    point_indices_inside_faces = mask == 1
    pts = pts[point_indices_inside_faces, :]
    return bounding_box_pointcloud(pts)


def bounding_box_pointcloud(pts: torch.Tensor):
    x = pts[:, 0]
    y = pts[:, 1]
    z = pts[:, 2]
    box = [[x.min(), y.min(), z.min()], [x.max(), y.max(), z.max()]]
    return torch.tensor(box)


def center_and_scale_uvgrid(inp: torch.Tensor, return_center_scale=False):
    bbox = bounding_box_uvgrid(inp)
    diag = bbox[1] - bbox[0]
    scale = 2.0 / max(diag[0], diag[1], diag[2])
    center = 0.5 * (bbox[0] + bbox[1])
    inp[..., :3] -= center
    inp[..., :3] *= scale
    if return_center_scale:
        return inp, center, scale
    return inp


def get_random_rotation():
    """Get a random rotation in 90 degree increments along the canonical axes"""
    axes = [
        np.array([1, 0, 0]),
        np.array([0, 1, 0]),
        np.array([0, 0, 1]),
    ]
    angles = [0.0, 90.0, 180.0, 270.0]
    axis = random.choice(axes)
    angle_radians = np.radians(random.choice(angles))
    return Rotation.from_rotvec(angle_radians * axis)


def rotate_uvgrid(inp, rotation):
    """Rotate the node features in the graph by a given rotation"""
    Rmat = torch.tensor(rotation.as_matrix()).float()
    orig_size = inp[..., :3].size()
    inp[..., :3] = torch.mm(inp[..., :3].view(-1, 3), Rmat).view(
        orig_size
    )  # Points
    inp[..., 3:6] = torch.mm(inp[..., 3:6].view(-1, 3), Rmat).view(
        orig_size
    )  # Normals/tangents
    return inp


INVALID_FONTS = [
    "Bokor",
    "Lao Muang Khong",
    "Lao Sans Pro",
    "MS Outlook",
    "Catamaran Black",
    "Dubai",
    "HoloLens MDL2 Assets",
    "Lao Muang Don",
    "Oxanium Medium",
    "Rounded Mplus 1c",
    "Moul Pali",
    "Noto Sans Tamil",
    "Webdings",
    "Armata",
    "Koulen",
    "Yinmar",
    "Ponnala",
    "Noto Sans Tamil",
    "Chenla",
    "Lohit Devanagari",
    "Metal",
    "MS Office Symbol",
    "Cormorant Garamond Medium",
    "Chiller",
    "Give You Glory",
    "Hind Vadodara Light",
    "Libre Barcode 39 Extended",
    "Myanmar Sans Pro",
    "Scheherazade",
    "Segoe MDL2 Assets",
    "Siemreap",
    "Signika SemiBold" "Taprom",
    "Times New Roman TUR",
    "Playfair Display SC Black",
    "Poppins Thin",
    "Raleway Dots",
    "Raleway Thin",
    "Segoe MDL2 Assets",
    "Segoe MDL2 Assets",
    "Spectral SC ExtraLight",
    "Txt",
    "Uchen",
    "Yinmar",
    "Almarai ExtraBold",
    "Fasthand",
    "Exo",
    "Freckle Face",
    "Montserrat Light",
    "Inter",
    "MS Reference Specialty",
    "MS Outlook",
    "Preah Vihear",
    "Sitara",
    "Barkerville Old Face",
    "Bodoni MT" "Bokor",
    "Fasthand",
    "HoloLens MDL2 Assests",
    "Libre Barcode 39",
    "Lohit Tamil",
    "Marlett",
    "MS outlook",
    "MS office Symbol Semilight",
    "MS office symbol regular",
    "Ms office symbol extralight",
    "Ms Reference speciality",
    "Segoe MDL2 Assets",
    "Siemreap",
    "Sitara",
    "Symbol",
    "Wingdings",
    "Metal",
    "Ponnala",
    "Webdings",
    "Souliyo Unicode",
    "Aguafina Script",
    "Yantramanav Black",
    # "Yaldevi",
    # Taprom,
    # "Zhi Mang Xing",
    # "Taviraj",
    # "SeoulNamsan EB",
]


def valid_font(filename):
    for name in INVALID_FONTS:
        if name.lower() in str(filename).lower():
            return False
    return True


def write_val_samples(root_dir, samples, labels):
    with open(f"{root_dir}/val_samples.txt", "w", encoding="utf-8") as f:
        for i in range(len(samples)):
            f.write(f"{samples[i]} {labels[i]}\n")
    print(f"Saved val samples to '{root_dir}'")


def files_load(root_dir):
    root_path = pathlib.Path(root_dir)
    if not root_path.exists():
        raise FileNotFoundError(f"The directory {root_dir} does not exist.")
    
    classes = [d.name for d in root_path.iterdir() if d.is_dir()]

    labels = []
    file_paths = []

    for cls in classes:
        cls_path = root_path / cls / "bin"
        steps_files = list(cls_path.rglob("*.bin"))

        if len(steps_files) < 2:
            print(f"Skipping class {cls} as it has fewer than 2 .bin files.")
            continue

        labels.extend([cls] * len(steps_files))
        file_paths.extend(steps_files)

    return file_paths, labels

def validate_graphs(file_paths : list[str], labels : list[str], duplicates : bool = False):
    clean_files = []
    new_labels = []
    hashes = set()
    for index, fn in enumerate(tqdm(file_paths)):
        if not fn.exists():
            continue
        sample = load_graphs(str(fn))[0][0]
        if sample is None:
            continue
        if sample.edata["x"].size(0) == 0:
            # Catch the case of graphs with no edges
            continue
        if not duplicates:
            hash = hashlib.sha256(sample.ndata["x"].numpy().tobytes()).hexdigest()
            if hash in hashes:
                continue
            hashes.add(hash)
        clean_files.append(fn)
        new_labels.append(labels[index])

    print("Done validation {} files".format(len(clean_files)))
    return clean_files, new_labels


def files_load_split(root_dir):
    path = pathlib.Path(root_dir)

    classes = [os.path.basename(d) for d in glob.glob(os.path.join(path, '*'))]
    labels = []
    file_paths = []
    for cls in classes:
        cls_path = pathlib.Path(root_dir + f"/{cls}" + "/bin")
        steps_files = [x for x in cls_path.rglob(f"*.bin")]
        # steps_files = glob.glob(os.path.join(cls_path, "*.bin")
        if len(steps_files) < 2:
            continue
        labels.extend([cls] * len(steps_files))
        file_paths.extend(steps_files)

    train_files, val_files, y_train, y_val = train_test_split(file_paths, labels, test_size=0.2, random_state=42, stratify=labels)

    return train_files, val_files, y_train, y_val


def filter_classes_by_min_samples(samples, classes, min_samples=7):
    # Step 1: Count class frequencies
    class_counts = defaultdict(int)
    for cls in classes:
        class_counts[cls] += 1

    # Step 2: Identify classes with at least `min_samples` samples
    valid_classes = {cls for cls, count in class_counts.items() if count >= min_samples}

    # Step 3: Filter samples and classes
    filtered_samples = []
    filtered_classes = []
    for sample, cls in zip(samples, classes):
        if cls in valid_classes:
            filtered_samples.append(sample)
            filtered_classes.append(cls)

    return filtered_samples, filtered_classes

def drop_nodes(data : DGLHeteroGraph, threshold : int = 10):
    node_num = data.num_nodes()
    if node_num <= threshold:
        return data
    drop_num = int(node_num / 10)

    idx_drop = np.random.choice(node_num, drop_num, replace=False)
    data.remove_nodes(idx_drop)
    return data

def drop_nodes_dynamicaly(data : DGLHeteroGraph, threshold : int = 10):
    node_num = data.num_nodes()
    if node_num <= threshold:
        return data

    # Dynamic removal percentage (sigmoid-like scaling between 5% and 20%)
    base_percent = 0.05  # 1% minimum removal
    scaling_factor = 1 / (1 + np.exp(-(node_num - 50)/20))  # Smooth scaling
    removal_percent = base_percent + (0.19 * scaling_factor)  # Ranges 1%-20%

    # Calculate number of nodes to remove
    drop_num = max(1, int(node_num * removal_percent))
    drop_num = min(drop_num, node_num - 1)  # Never remove all nodes
    # drop_num = int(node_num / 10)
    idx_drop = np.random.choice(node_num, drop_num, replace=False)
    data.remove_nodes(idx_drop)
    return data
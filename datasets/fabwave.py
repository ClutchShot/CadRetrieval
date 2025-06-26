import pathlib
import string
import glob
import os
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from datasets.base import BaseDataset
from occwl.io import load_step
from process.solid_to_graph import build_graph
from dgl.data.utils import load_graphs
import numpy as np
from dgl import DGLHeteroGraph
from copy import deepcopy
import dgl
from torch import FloatTensor

def _get_filenames(root_dir, filelist):
    with open(str(root_dir / f"{filelist}"), "r") as f:
        file_list = [x.strip() for x in f.readlines()]

    files = list(
        x
        for x in root_dir.rglob(f"*.bin")
        if x.stem in file_list
        #if util.valid_font(x) and x.stem in file_list
    )
    return files

def write_val_samples(root_dir, samples, labels):
    with open(f"{root_dir}/val_samples.txt", "w", encoding="utf-8") as f:
        for i in range(len(samples)):
            f.write(f"{samples[i]} {labels[i]}\n")
    print(f"Saved val samples to '{root_dir}'")


# def read_samples(root_dir, samples):
#     samples = []
#     labels = []
#     with open(f"{root_dir}/val_samples.txt", "r", encoding="utf-8") as f:
#         lines = f.readlines()
#         for line in lines:
#             file, label = line.split(" ")
#                 samples = []
#                 labels = []


def files_load_split(root_dir):
    path = pathlib.Path(root_dir)

    classes = [os.path.basename(d) for d in glob.glob(os.path.join(path, '*'))]
    labels = []
    file_paths = []
    index = 0
    for cls in classes:
        cls_path = pathlib.Path(root_dir + f"/{cls}" + "/bin")
        steps_files = [x for x in cls_path.rglob(f"*.bin")]
        # steps_files = glob.glob(os.path.join(cls_path, "*.bin"))
        if len(steps_files) < 2:
            continue
        labels.extend([index for i in steps_files])
        file_paths.extend(steps_files)
        index+=1

    train_files, val_files, y_train, y_val = train_test_split(file_paths, labels, test_size=0.2, random_state=42, stratify=labels)

    return train_files, val_files, y_train, y_val



class FABWave(BaseDataset):
    @staticmethod
    def num_classes():
        return 42

    def __init__(
        self,
        file_paths,
        labels,
        split="train",
        center_and_scale=True,
        random_rotate=False,
    ):
        """
        Load the SolidLetters dataset

        Args:
            root_dir (str): Root path to the dataset
            split (str, optional): Split (train, val, or test) to load. Defaults to "train".
            center_and_scale (bool, optional): Whether to center and scale the solid. Defaults to True.
            random_rotate (bool, optional): Whether to apply random rotations to the solid in 90 degree increments. Defaults to False.
        """
        self.random_rotate = random_rotate

        print(f"Loading {split} data...")
        self.load_graphs(file_paths, labels, center_and_scale)
        print("Done loading {} files".format(len(self.data)))


    def load_graphs(self, file_paths, labels, center_and_scale=True):
        self.data = []
        for index, fn in enumerate(tqdm(file_paths)):
            if not fn.exists():
                continue
            sample = self.load_one_graph(fn, labels[index])
            if sample is None:
                continue
            if sample["graph"].edata["x"].size(0) == 0:
                # Catch the case of graphs with no edges
                continue
            self.data.append(sample)
        if center_and_scale:
            self.center_and_scale()
        self.convert_to_float32()


    def load_one_graph(self, file_path, label):
        # Load the graph using base class method
        sample = super().load_one_graph(file_path)
        # sample_aug = self.drop_nodes(deepcopy(sample['graph'].clone()))
        sample_aug = self.drop_nodes_dynamicaly(deepcopy(sample['graph'].clone()))
        # Load the graph using base class method
        sample["label"] = torch.tensor([torch.tensor(label).long()]).long()
        sample['graph_aug'] = sample_aug
        return sample
    
    def _collate(self, batch):
        collated = super()._collate(batch)
        batched_graph_aug = dgl.batch([sample["graph_aug"] for sample in batch])
        collated["graph_aug"] = batched_graph_aug
        collated["label"] =  torch.cat([x["label"] for x in batch], dim=0)
        return collated
    

    def drop_nodes(self, data : DGLHeteroGraph):
        node_num = data.num_nodes()
        drop_num = int(node_num / 10)
        if drop_num == 0 and node_num > 5:
            drop_num = 1

        idx_drop = np.random.choice(node_num, drop_num, replace=False)

        data.remove_nodes(idx_drop)
        return data
    
    def drop_nodes_dynamicaly(self, data : DGLHeteroGraph, threshold : int = 10):
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
    
    def convert_to_float32(self):
        for i in range(len(self.data)):
            self.data[i]["graph"].ndata["x"] = self.data[i]["graph"].ndata["x"].type(FloatTensor)
            self.data[i]["graph"].edata["x"] = self.data[i]["graph"].edata["x"].type(FloatTensor)
            self.data[i]["graph_aug"].ndata["x"] = self.data[i]["graph_aug"].ndata["x"].type(FloatTensor)
            self.data[i]["graph_aug"].edata["x"] = self.data[i]["graph_aug"].edata["x"].type(FloatTensor)
    
    # def __getitem__(self, idx):
    #     sample = self.data[idx]
    #     if self.random_rotate:
    #         rotation = util.get_random_rotation()
    #         sample["graph"].ndata["x"] = util.rotate_uvgrid(sample["graph"].ndata["x"], rotation)
    #         sample["graph"].edata["x"] = util.rotate_uvgrid(sample["graph"].edata["x"], rotation)
    #     return sample
import pathlib
import string
from copy import deepcopy
import torch
from sklearn.model_selection import train_test_split
from datasets.util import drop_nodes, drop_nodes_dynamicaly
from datasets.base import BaseDataset
import dgl
import torch
from torch import FloatTensor
import numpy as np
from tqdm import tqdm
import random

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


CHAR2LABEL = {char: i for (i, char) in enumerate(string.ascii_lowercase)}


def _char_to_label(char):
    return CHAR2LABEL[char.lower()]


class SolidLetters(BaseDataset):
    @staticmethod
    def num_classes():
        return 26

    def __init__(
        self,
        root_dir,
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
        assert split in ("train", "val", "test")
        path = pathlib.Path(root_dir)

        self.random_rotate = random_rotate
        self.data_aug = drop_nodes
        self.split = split
        
        if split in ("train", "val"):
            file_paths = _get_filenames(path, filelist="train.txt")
            # The first character of filename must be the alphabet
            labels = [_char_to_label(fn.stem[0]) for fn in file_paths]
            # train_files, val_files = train_test_split(
            #     file_paths, test_size=0.2, random_state=42, stratify=labels,
            # )
            if split == "train":
                file_paths = file_paths
            elif split == "val":
                file_paths = file_paths
        elif split == "test":
            file_paths = _get_filenames(path, filelist="test.txt")

        print(f"Loading {split} data...")
        self.load_graphs(file_paths, center_and_scale)
        random.shuffle(self.data)
        print("Done loading {} files".format(len(self.data)))

    def load_one_graph(self, file_path):
        # Load the graph using base class method
        sample = super().load_one_graph(file_path)
        # Additionally get the label from the filename and store it in the sample dict
        sample_aug = self.data_aug(deepcopy(sample['graph'].clone()))

        sample["label"] = file_path.stem[0]
        sample['graph_aug'] = sample_aug
        return sample

    # def _collate(self, batch):
    #     collated = super()._collate(batch)
    #     collated["label"] =  torch.cat([x["label"] for x in batch], dim=0)
    #     return collated
    
    def _collate(self, batch):
        collated = super()._collate(batch)
        batched_graph_aug = dgl.batch([sample["graph_aug"] for sample in batch])
        collated["graph_aug"] = batched_graph_aug
        # collated["label"] =  torch.cat([x["label"] for x in batch], dim=0)
        #  For str label
        collated["label"] = [x["label"] for x in batch]
        if any("graph_neg" in sample for sample in batch):
            batched_graph_neg = dgl.batch([sample["graph_neg"] for sample in batch])
            collated["graph_neg"] = batched_graph_neg
        return collated

    def convert_to_float32(self):
        for i in range(len(self.data)):
            self.data[i]["graph"].ndata["x"] = self.data[i]["graph"].ndata["x"].type(FloatTensor)
            self.data[i]["graph"].edata["x"] = self.data[i]["graph"].edata["x"].type(FloatTensor)
            self.data[i]["graph_aug"].ndata["x"] = self.data[i]["graph_aug"].ndata["x"].type(FloatTensor)
            self.data[i]["graph_aug"].edata["x"] = self.data[i]["graph_aug"].edata["x"].type(FloatTensor)


    def generate_negative_samples(self):
        size = len(self.data)
        idx_drop = np.random.choice(size, size, replace=False)

        for index, data in enumerate(tqdm(self.data, desc=f"Generate negative samples for {self.split}")):
            data["graph_neg"] = self.data[idx_drop[index]]["graph"]
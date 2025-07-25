from tqdm import tqdm
from datasets.base import BaseDataset
from copy import deepcopy
import dgl
import torch
from torch import FloatTensor
from datasets.util import drop_nodes, drop_nodes_dynamicaly
import numpy as np
from dgl import DGLHeteroGraph
from torch_geometric.data import Data, Batch
from torch_geometric.graphgym.config import cfg
# from GNNPlus.encoder.graphormer_encoder import graphormer_pre_processing
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset

class FABWave(BaseDataset):

    def __init__(
        self,
        file_paths,
        labels,
        split="train",
        center_and_scale=True,
        random_rotate=False,
        data_aug = "standard"
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
        self.data_aug = None
        self.split = split
        if data_aug == "standard":
            self.data_aug = drop_nodes
        elif data_aug == "dynamicaly":
            self.data_aug = drop_nodes_dynamicaly

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

    def data_stats(self):
        stats = dict()
        for data in self.data:
            if data['label'].item() not in stats.keys():
                stats[data['label'].item()] = 0
            else:
                stats[data['label'].item()] +=1
        sorted_dict = dict(sorted(stats.items(), key=lambda item: item[1], reverse=True))
        return sorted_dict
    
    def get_labels(self):
        labels = []
        for data in self.data:
            labels.append(data['label'])
        return labels

    def load_one_graph(self, file_path, label):
        # Load the graph using base class method
        sample = super().load_one_graph(file_path)
        # sample_aug = self.drop_nodes(deepcopy(sample['graph'].clone()))
        sample_aug = self.data_aug(deepcopy(sample['graph'].clone()))
        # Load the graph using base class method
        sample["label"] = label
        sample['graph_aug'] = sample_aug
        return sample

    # def _collate(self, batch):
    #     collated = super()._collate(batch)
    #     batched_graph_aug = dgl.batch([sample["graph_aug"] for sample in batch])
    #     collated["graph_aug"] = batched_graph_aug
    #     # collated["label"] =  torch.cat([x["label"] for x in batch], dim=0)
    #     #  For str label
    #     collated["label"] = [x["label"] for x in batch]
    #     if any("graph_neg" in sample for sample in batch):
    #         batched_graph_neg = dgl.batch([sample["graph_neg"] for sample in batch])
    #         collated["graph_neg"] = batched_graph_neg
    #     return collated

    def _collate(self, batch):
        batched_graph = Batch.from_data_list([self.convert_dgl_to_pyg(sample["graph"]) for sample in batch])
        batched_graph_aug = Batch.from_data_list([self.convert_dgl_to_pyg(sample["graph_aug"]) for sample in batch])
        batched_filenames = [sample["filename"] for sample in batch]
        label = [x["label"] for x in batch]

        if any("graph_neg" in sample for sample in batch):
            batched_graph_neg = Batch.from_data_list([self.convert_dgl_to_pyg(sample["graph_neg"]) for sample in batch])
            return {"graph": batched_graph, "graph_aug": batched_graph_aug, "graph_neg": batched_graph_neg, "filename": batched_filenames, "label": label}
        else:
            return {"graph": batched_graph, "graph_aug": batched_graph_aug, "filename": batched_filenames, "label": label}

    def convert_to_float32(self):
        for i in range(len(self.data)):
            self.data[i]["graph"].ndata["x"] = self.data[i]["graph"].ndata["x"].type(FloatTensor)
            self.data[i]["graph"].edata["x"] = self.data[i]["graph"].edata["x"].type(FloatTensor)
            self.data[i]["graph_aug"].ndata["x"] = self.data[i]["graph_aug"].ndata["x"].type(FloatTensor)
            self.data[i]["graph_aug"].edata["x"] = self.data[i]["graph_aug"].edata["x"].type(FloatTensor)

    def convert_dgl_to_pyg(self, data: DGLHeteroGraph) -> Data:
        x = torch.flatten(data.ndata['x'], start_dim=1)
        edge_attr = torch.flatten(data.edata['x'], start_dim=1)
        edge_index = torch.stack(data.edges()).long()
        num_nodes = data.num_nodes()
        # if cfg.posenc_GraphormerBias.enable:
        #     return graphormer_pre_processing(
        #         Data(x=x, edge_attr=edge_attr, edge_index=edge_index, num_nodes=num_nodes), 
        #         cfg.posenc_GraphormerBias.num_spatial_types)
        return Data(x=x, edge_attr=edge_attr, edge_index=edge_index, num_nodes=num_nodes)

    def generate_negative_samples(self):
        size = len(self.data)
        idx_drop = np.random.choice(size, size, replace=False)

        for index, data in enumerate(tqdm(self.data, desc=f"Generate negative samples for {self.split}")):
            data["graph_neg"] = self.data[idx_drop[index]]["graph"]

    # def __getitem__(self, idx):
    #     sample = self.data[idx]
    #     if self.random_rotate:
    #         rotation = util.get_random_rotation()
    #         sample["graph"].ndata["x"] = util.rotate_uvgrid(sample["graph"].ndata["x"], rotation)
    #         sample["graph"].edata["x"] = util.rotate_uvgrid(sample["graph"].edata["x"], rotation)
    #     return sample
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm
from retrieval.vector_db import VectorDatabase
from retrieval.metrics import calculate_map
import pathlib
from utils.read_txt import get_filenames_by_type_and_key
from utils.safe_results import safe_raw_results, safe_by_class_results, safe_total_results
from datasets.util import files_load
from sklearn.model_selection import train_test_split

class ResNetFeatureExtractor(nn.Module):
    def __init__(self):
        super(ResNetFeatureExtractor, self).__init__()
        resnet = models.resnet152(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-1])  # Remove final FC layer
        self.eval()  # Set to evaluation mode
        
    def forward(self, x):
        with torch.no_grad():  
            features = self.features(x)
            features = features.view(features.size(0), -1)  # Flatten
        return features

# 2. Image preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

# 3. Extract features from a single image
def extract_features(image_path, model):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension
    features = model(image)
    return features.numpy()

# Example usage
if __name__ == "__main__":
    # Initialize model
    model = ResNetFeatureExtractor()

    # root_dir = "./data/SolidLetters/"
    # root_dir = pathlib.Path(root_dir)

    db_path = "./vector_db/resnet152Fabwave"
    dataset = "Fabwave"
    out_dim = 2048

    # train_png_paths = get_filenames_by_type_and_key(root_dir, "train.txt", ".png", "_bottom")
    # test_png_paths = get_filenames_by_type_and_key(root_dir, "test.txt", ".png", "_bottom")

    # train_names = [path.stem for path in train_png_paths]
    # train_label = [name.split("_")[0] for name in train_names]

    # test_names = [path.stem for path in test_png_paths]
    # test_label = [name.split("_")[0] for name in test_names]



    root_dir = "./data/Fabwave/"
    files, labels = files_load(root_dir, "png", "png")
    train_png_paths, test_png_paths, train_label, test_label =  train_test_split(files, labels, test_size=0.2, random_state=42, stratify=labels)
    train_names = [str(name.stem) for name in train_png_paths]
    test_names = [str(name.stem) for name in test_png_paths]


    vec_db = VectorDatabase(db_path, dataset, out_dim)

    if vec_db.get_vector_count() == 0:
    # Get train embendings
        for index in tqdm(range(len(train_png_paths)), desc="Infernce train data"):

            out = extract_features(train_png_paths[index], model)
            vec_db.add_vectors(vectors=out, names=[train_names[index]], labels=[train_label[index]], duplicates=True)

    queries = []
    retrieval_all = []
    # Eval retrieval
    for index in tqdm(range(len(test_png_paths)), desc="Eval"):

        out = extract_features(test_png_paths[index], model)

        retrieval_topk = vec_db.search(out, k=7)
        retrieval_all.extend(retrieval_topk)
        queries.append({"name": test_names[index], "label": test_label[index]})


    map_score, detailed = calculate_map(queries, retrieval_all)

    safe_raw_results(db_path, detailed)
    df_grouped = safe_by_class_results(db_path, detailed)
    safe_total_results(db_path, map_score, df_grouped)
        



import argparse
import pathlib
import time
from tqdm import tqdm
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything

from datasets.solidletters import SolidLetters
from datasets.fabwave import FABWave
from datasets.util import files_load, validate_graphs, filter_classes_by_min_samples
from uvnet.models import Contrast
from sklearn.model_selection import train_test_split
from retrieval.vector_db import VectorDatabase
from retrieval.metrics import calculate_map
from utils.safe_results import safe_raw_results, safe_by_class_results, safe_total_results

parser = argparse.ArgumentParser("CAD retrieval learning")
parser.add_argument("--dataset", choices=("solidletters", "FABWave"), help="Dataset to train on")
parser.add_argument("--dataset_path", type=str, help="Path to dataset")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
parser.add_argument("--db_path", type=str, help="Path to vector db")
parser.add_argument("--out_dim", type=int, help="Output dimension")
parser.add_argument("--loss", choices=("L2", "graphcl"), help="Loss fuction", default="graphcl")
parser.add_argument("--data_aug", choices=("standard", "dynamicaly"), help="Data augmentation", default="standard")
parser.add_argument(
    "--num_workers",
    type=int,
    default=0,
    help="Number of workers for the dataloader. NOTE: set this to 0 on Windows, any other value leads to poor performance",
)
parser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    help="Checkpoint file to load weights from for testing",
)

parser = Trainer.add_argparse_args(parser)
args = parser.parse_args()

results_path = (
    pathlib.Path(__file__).parent.joinpath("results")
)
if not results_path.exists():
    results_path.mkdir(parents=True, exist_ok=True)

# Define a path to save the results based date and time. E.g.
# results/args.experiment_name/0430/123103
month_day = time.strftime("%m%d")
hour_min_second = time.strftime("%H%M%S")
checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    dirpath=str(results_path.joinpath(month_day, hour_min_second)),
    filename="best",
    save_last=True,
)

trainer = Trainer.from_argparse_args(
    args,
    callbacks=[checkpoint_callback],
    logger=TensorBoardLogger(
        str(results_path), name=month_day, version=hour_min_second,
    ),
)

if args.dataset == "solidletters":
    Dataset = SolidLetters
    train_data = Dataset(root_dir=args.dataset_path, split="train")
    val_data = Dataset(root_dir=args.dataset_path, split="val")
elif args.dataset == "FABWave":
    files, labels = files_load(args.dataset_path)
    files, labels = validate_graphs(files, labels)
    files, labels  = filter_classes_by_min_samples(files, labels)
    train_files, val_files, y_train, y_val = train_test_split(files, labels, test_size=0.2, random_state=42, stratify=labels)
    train_data = FABWave(file_paths=train_files, labels=y_train, split="train", data_aug=args.data_aug)
    val_data = FABWave(file_paths=val_files, labels=y_val, split="val", data_aug=args.data_aug)
else:
    raise ValueError("Unsupported dataset")


# Train/val
seed_everything(workers=True)
print(
    f"""
-----------------------------------------------------------------------------------
Logs written to results/{month_day}/{hour_min_second}

To monitor the logs, run:
tensorboard --logdir results/{month_day}/{hour_min_second}

The trained model with the best validation loss will be written to:
results/{month_day}/{hour_min_second}/best.ckpt
-----------------------------------------------------------------------------------
"""
)
model = Contrast(out_emb_dim=args.out_dim, loss=args.loss)
train_loader = train_data.get_dataloader(
    batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
)
val_loader = val_data.get_dataloader(
    batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
)
trainer.fit(model, train_loader, val_loader)

vec_db = VectorDatabase(args.db_path, args.dataset, args.out_dim)

if vec_db.get_vector_count() == 0:
    # Get train embendings
    for data in tqdm(train_loader, desc="Infernce train data"):
        # model.predict() wrapered eval() and torch.no_grad()
        out = model.predict(data)

        out = out.cpu().numpy()
        names = data['filename']
        label = data['label']

        vec_db.add_vectors(vectors=out, names=names, labels=label, duplicates=True)


queries = []
retrieval_all = []
# Eval retrieval
for data in tqdm(val_loader, desc="Eval"):

    # model.predict() wrapered eval() and torch.no_grad()
    out = model.predict(data)

    out = out.cpu().numpy()
    names = data['filename']
    labels = data['label']

    retrieval_topk = vec_db.search(out, k=7)
    retrieval_all.extend(retrieval_topk)
    for name, label in zip(names, labels):
        queries.append({"name": name, "label": label})


map_score, detailed = calculate_map(queries, retrieval_all)

safe_raw_results(args.db_path, detailed)
df_grouped = safe_by_class_results(args.db_path, detailed)
safe_total_results(args.db_path, map_score, df_grouped)

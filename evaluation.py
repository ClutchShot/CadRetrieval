import argparse
from tqdm import tqdm
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything
import torch
from datasets.solidletters import SolidLetters
from datasets.fabwave import FABWave, files_load_split, write_val_samples
from uvnet.models import Contrast

from retrieval.vector_db import VectorDatabase
from retrieval.metrics import calculate_map

parser = argparse.ArgumentParser("CAD retrieval evaluation")

parser.add_argument("--dataset", choices=("solidletters",
                    "FABWave"), help="Dataset to train on")
parser.add_argument("--dataset_path", type=str, help="Path to dataset")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
parser.add_argument("--db_path", type=str, help="Path to vector db")
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

# results_path = (
#     pathlib.Path(__file__).parent.joinpath("results").joinpath(args.experiment_name)
# )

# Define a path to save the results based date and time. E.g.
# results/args.experiment_name/0430/123103
# month_day = time.strftime("%m%d")
# hour_min_second = time.strftime("%H%M%S")
# checkpoint_callback = ModelCheckpoint(
#     monitor="val_loss",
#     dirpath=str(results_path.joinpath(month_day, hour_min_second)),
#     filename="best",
#     save_last=True,
# )


if args.dataset == "solidletters":
    Dataset = SolidLetters
    val_data = Dataset(root_dir=args.dataset_path, split="val")
elif args.dataset == "FABWave":
    train_files, val_files, y_train, y_val = files_load_split(
        root_dir=args.dataset_path)
    write_val_samples(args.dataset_path, val_files, y_val)
    train_data = FABWave(file_paths=train_files, labels=y_train, split="train")
    val_data = FABWave(file_paths=val_files, labels=y_val, split="val")
else:
    raise ValueError("Unsupported dataset")


train_loader = train_data.get_dataloader(
    batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
)
val_loader = val_data.get_dataloader(
    batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = Contrast.load_from_checkpoint(args.checkpoint)
model.to(device)

# Initilize vecto db
vec_db = VectorDatabase(args.db_path, args.dataset, 64)

if vec_db.get_vector_count() == 0:
    # Get train embendings
    for data in tqdm(train_loader, desc="Infernce train data"):
        # model.predict() wrapered eval() and torch.no_grad()
        out = model.predict(data)

        out = out.cpu().numpy()
        names = data['filename']
        label = data['label'].numpy()

        vec_db.add_vectors(vectors=out, names=names, labels=label)


queries = []
retrieval_all = []
# Eval retrieval
for data in tqdm(val_loader, desc="Eval"):

    # model.predict() wrapered eval() and torch.no_grad()
    out = model.predict(data)

    out = out.cpu().numpy()
    names = data['filename']
    labels = data['label'].numpy()

    retrieval_topk = vec_db.search(out, k=7)
    retrieval_all.extend(retrieval_topk)
    for name, label in zip(names, labels):
        queries.append({"name": name, "label": label})


map_score, detailed = calculate_map(queries, retrieval_all, verbose=True)
print(f"mAP: {map_score:.4f}")

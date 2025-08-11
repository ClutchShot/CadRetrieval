import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import pathlib


VIEWS = ["front", "back", "left", "right", "top", "bottom",
         "isometric1", "isometric2", "isometric3", "isometric4",
         "isometric5", "isometric6", "isometric7", "isometric8"]

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])


def image_to_bin(file_name : pathlib.Path):
    """Generate multiple views of a STEP file from different angles"""
    vectors = []
    for view in VIEWS:
        file_name_view = file_name.with_name(f"{file_name.stem}_{view}.png")
        image = Image.open(file_name_view).convert('RGB')
        vectors.append(transform(image))

    k_vector = torch.stack(vectors, dim=0)

    output_prefix = file_name_view.parent.parent / "img_bin512"
    if not output_prefix.exists():
        output_prefix.mkdir(parents=True, exist_ok=True)
    new_dir = output_prefix / f"{file_name.stem}_stack512x512.bin"
    torch.save(k_vector, new_dir)

    

if __name__ == "__main__":

    # Fabwave
    root_dir = "./data/Fabwave/"
    root_dir = pathlib.Path(root_dir)
    files = [file for file in root_dir.rglob(f"*.st*p")]

    for file in tqdm(files, desc="Processing img to bin"):
        png_prefix = file.parent.parent / "png" / file.name
        try:
            image_to_bin(png_prefix)
        except:
            continue

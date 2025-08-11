from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Display.SimpleGui import init_display
import os
os.environ["PYTHONOCC_OFFSCREEN_RENDERER"] = "1"  
from OCC.Core.V3d import (V3d_Xpos, V3d_Xneg, 
                         V3d_Ypos, V3d_Yneg,
                         V3d_Zpos, V3d_Zneg,
                         V3d_XposYposZpos, 
                         V3d_XposYposZneg, 
                         V3d_XposYnegZpos, 
                         V3d_XnegYposZpos, 
                         V3d_XposYnegZneg, 
                         V3d_XnegYnegZpos, 
                         V3d_XnegYposZneg, 
                         V3d_XnegYnegZneg)
from tqdm import tqdm
import pathlib
from utils.read import get_filenames_by_type

def generate_multi_view_step(step_file, output_prefix):
    """Generate multiple views of a STEP file from different angles"""
    
    # Read the STEP file
    step_reader = STEPControl_Reader()
    status = step_reader.ReadFile(str(step_file))
    
    if status != IFSelect_RetDone:
        # raise ValueError("Error reading STEP file")
        return
    
    step_reader.TransferRoots()
    shape = step_reader.OneShape()
    
    # Initialize display
    display, start_display, add_menu, add_function_to_menu = init_display()
    display.DisplayShape(shape, update=True)
    display.FitAll()
    
    # Define standard views
    views = {
        "front": V3d_Yneg,
        "back": V3d_Ypos,
        "left": V3d_Xneg,
        "right": V3d_Xpos,
        "top": V3d_Zpos,
        "bottom": V3d_Zneg,
        "isometric1": V3d_XposYposZpos, 
        "isometric2": V3d_XposYposZneg, 
        "isometric3": V3d_XposYnegZpos, 
        "isometric4": V3d_XnegYposZpos, 
        "isometric5": V3d_XposYnegZneg, 
        "isometric6": V3d_XnegYnegZpos, 
        "isometric7": V3d_XnegYposZneg, 
        "isometric8": V3d_XnegYnegZneg
    }


    if not output_prefix.exists():
        output_prefix.mkdir(parents=True, exist_ok=True)
    # Capture each view
    output_prefix = str(output_prefix / step_file.stem)
    for name, orientation in views.items():
        display.View.SetProj(orientation)
        display.View.FitAll()
        display.View.ZFitAll()
        output_file = f"{output_prefix}_{name}.png"
        display.View.Dump(output_file)
    
    display.Context.RemoveAll(True)


if __name__ == "__main__":
    #  SolidLetters
    # root_dir = "./data/SolidLetters/"
    # root_dir = pathlib.Path(root_dir)

    # train_file = "train.txt"
    # test_file = "test.txt"
    # png_path = "./data/SolidLetters/png"
    # png_path = pathlib.Path(png_path)

    # train_file_paths = get_filenames_by_type(root_dir, train_file, ".st*p")
    # test_file_paths = get_filenames_by_type(root_dir, test_file, ".st*p")
    # step_file_paths = train_file_paths + test_file_paths

    # for step_file in tqdm(step_file_paths, desc="Processing img"):
    #     generate_multi_view_step(step_file, png_path)


    # Fabwave 
    root_dir = "./data/Fabwave/"
    root_dir = pathlib.Path(root_dir)
    files = [file for file in root_dir.rglob(f"*.st*p")]
    
    for file in tqdm(files, desc="Processing img"):
        prefix = file.parent.parent / "png"
        try:
            generate_multi_view_step(file, prefix)
        except:
            continue
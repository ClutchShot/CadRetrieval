from tqdm import tqdm

def read_txt(file_path: str) -> str:
    """
    Reads a text file and appends each line (or the entire content) to a documents list.
    
    Args:
        file_path (str): Path to the .txt file
        
    Returns:
        Text
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            # Option 1: Read entire content as one document
            content = file.read()
            return content.replace("\n", " ")
                    
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except Exception as e:
        print(f"Error reading file: {e}")

def read_documents(files_path : list[str], name_split : str = "train") -> list[str]:
    documents_text = []
    names = []
    for file in tqdm(files_path, desc=f"loading txt files of {name_split}"):
        documents_text.append(read_txt(file))
        # names.append(file.split("\\")[-1])
        names.append(str(file).split("\\")[-1])
    return documents_text, names

def get_filenames_by_type(root_dir, filelist : str, type : str):
    with open(str(root_dir / f"{filelist}"), "r") as f:
        file_list = [x.strip() for x in f.readlines()]

    files = list(
        x
        for x in root_dir.rglob(f"*{type}")
        if x.stem in file_list 
        #if util.valid_font(x) and x.stem in file_list
    )
    return files

def get_filenames_by_type_and_key(root_dir, filelist : str, type : str, key: str):
    with open(str(root_dir / f"{filelist}"), "r") as f:
        file_list = [x.strip()+key for x in f.readlines()]

    files = list(
        x
        for x in root_dir.rglob(f"*{type}")
        if x.stem in file_list 
        #if util.valid_font(x) and x.stem in file_list
    )
    return files

def get_samples(root_dir, filelist : str):
    with open(str(root_dir / f"{filelist}"), "r") as f:
        file_list = [x.strip() for x in f.readlines()]
    return file_list

def get_all_png_filenames(root_dir):
    """
    Get all PNG files and return set of filename parts (split by "_" excluding last)
    """
    png_files = root_dir.rglob("*.png")
    result_set = set()
    
    for png_file in png_files:
        filename = png_file.stem
        result_set.add("_".join(filename.split("_")[:-1]))
    
    return result_set

def filter_list_by_set(input_list, valid_set):
    """
    Remove elements from list that are not in the valid set
    Returns a new list with only valid elements
    """
    return [item for item in input_list if item in valid_set]
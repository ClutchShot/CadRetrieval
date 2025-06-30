import pathlib
import glob
import os

root_dir = "./data/FABWave"

path = pathlib.Path(root_dir)

classes = [os.path.basename(d) for d in glob.glob(os.path.join(path, '*'))]
classes_samples = dict()

for cls in classes:
    cls_path = pathlib.Path(root_dir + f"/{cls}" + "/bin")
    steps_files = [x for x in cls_path.rglob(f"*.bin")]
    # steps_files = glob.glob(os.path.join(cls_path, "*.bin"))
    if len(steps_files) < 2:
        continue
    classes_samples[cls] = len(steps_files)

sorted_dict = dict(sorted(classes_samples.items(), key=lambda item: item[1], reverse=True))
a = 0
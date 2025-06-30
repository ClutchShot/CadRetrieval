from collections import defaultdict


file_dir = "./data/vec_db/results_log1.txt"


label_aps = defaultdict(list)


with open(file_dir, 'r') as file: 
    for line in file:

        if 'label=' in line and 'AP = ' in line:
            try:
                label_part = line.split('label=')[1].split('):')[0]
                label = int(label_part)
                
                ap_value = float(line.split('AP = ')[1].strip())
                
                label_aps[label].append(ap_value)
            except (IndexError, ValueError):
                continue


average_aps = {}
avarage_average_aps = 0
for label, aps in label_aps.items():
    average_aps[label] = sum(aps) / len(aps)

for label, aps in average_aps.items():
    avarage_average_aps += average_aps[label]

sort_average_aps = sorted(average_aps.items(), key=lambda x: x[1], reverse=True)


for label in sorted(average_aps.keys()):
    print(f"Label {label}: Average AP = {average_aps[label]:.4f}")

avarage_average_aps = avarage_average_aps / len(average_aps)
print(f"Avarege per classes AP: {avarage_average_aps:.4f}")

a = 0
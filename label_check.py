import json
import numpy as np
import matplotlib.pyplot as plt
import tqdm

total_label = [0] * 228
with open('input/train.json', 'r') as f:
    jsonData = json.load(f)

dict1 = dict()
pbar = tqdm.tqdm(total=len(jsonData["annotations"]))
for i in range(len(jsonData["annotations"])):
    num = 0
    for j in jsonData["annotations"][i]["labelId"]:
        dict1[j] = dict1.get(j, 0) + 1
        num += 1
    total_label[num - 1] += 1
    pbar.update(1)

pbar.close()

total_label = np.array(total_label)
plt.scatter(range(1, np.where(total_label > 0)[0][-1] + 1), total_label[:np.where(total_label > 0)[0][-1]])
plt.title("ラベル数/写真")
plt.xlabel("x axis")
plt.ylabel("y axis")
plt.grid(True)

average = np.sum([(i + 1) * j for i, j in enumerate(total_label)]) / len(jsonData["annotations"])

print(dict1)
print(sorted([int(i) for i in dict1.keys()]))
print(len(dict1))  # =>228

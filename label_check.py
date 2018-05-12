import json


with open('input/train.json', 'r') as f:
    jsonData = json.load(f)

dict1 = dict()

for i in range(len(jsonData["annotations"])):
    for j in jsonData["annotations"][i]["labelId"]:
        dict1[j] = dict1.get(j, 0) + 1

print(dict1)
print(sorted([int(i) for i in dict1.keys()]))
print(len(dict1))  # =>228

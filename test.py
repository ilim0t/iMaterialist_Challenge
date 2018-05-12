import json

with open('./input/train.json', 'r') as f:
    jsonData = json.load(f)
    print(jsonData["images"][0])

    ichibu = open('.json' , 'w')
    json.dump(dic,a)
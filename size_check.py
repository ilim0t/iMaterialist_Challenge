import numpy as np
import matplotlib
matplotlib.use('Agg') # -----(1)
import matplotlib.pyplot as plt
from PIL import Image
import os
import tqdm
import pprint

dict1 = dict()

pbar = tqdm.tqdm(total=10000)
for i in range(10000):
    pbar.update(1)
    if not os.path.isfile('data/train_images/' + str(i) + '.jpg'):
        continue
    img_data = Image.open('data/train_images/' + str(i) + '.jpg')
    img = np.asarray(img_data)
    dict1[img.shape[:2]] = dict1.get(img.shape[:2], 0) + 1
pbar.close()
pprint.pprint(dict1, compact=True)


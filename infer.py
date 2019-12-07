
# %%
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from PIL import Image
import numpy as np
import pandas as pd
import os

workers = 0 if os.name == 'nt' else 4
# %%
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

# %%
mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, prewhiten=True,
    device=device
)
# %%
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# %%
def collate_fn(x):
    return x[0]

dataset = datasets.ImageFolder('./CVFinalProject/dataset/')
dataset.idx_to_class = {i:c for c, i in dataset.class_to_idx.items()}
loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=workers)

# %%
aligned = []
names = []
for x, y in loader:
    x_aligned, prob = mtcnn(x, return_prob=True)
    if x_aligned is not None:
        print('Face detected with probability: {:8f}'.format(prob))
        aligned.append(x_aligned)
        names.append(dataset.idx_to_class[y])
print(names)
# %%
aligned = torch.stack(aligned).to(device)
embeddings = resnet(aligned).detach().cpu()

# %%
#dists = [[(e1 - e2).norm().item() for e2 in embeddings] for e1 in embeddings]
#print(pd.DataFrame(dists, columns=names, index=names))

# %%
#import torchvision
#to_pil = torchvision.transforms.ToPILImage()
#img = to_pil(aligned[7])
#plt.imshow(img)

# %%
im = Image.open("CVFinalProject/chris.jpg")
x_aligned, prob = mtcnn(im, return_prob=True)
if x_aligned is not None:
    print('Face detected with probability: {:8f}'.format(prob))
    x_embeddings = resnet(x_aligned[None, :, :]).detach().cpu()

# %%
dists = [(e1 - x_embeddings).norm().item() for e1 in embeddings]

# %%
print(dists)

from imutils.video import VideoStream
from datetime import datetime
import imutils
import time
import cv2
import os


print("Starting up camera ....")
vs = VideoStream(src=0).start()
time.sleep(2.0)
total = 0

print("Press p to the a picture")

while True:
    frame = vs.read()
    orig = frame.copy()
    frame = imutils.resize(frame, width=400)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("p"):
        img = Image.fromarray(frame)
        x_aligned, prob = mtcnn(img, return_prob=True)
        if x_aligned is not None:
            print('Face detected with probability: {:8f}'.format(prob))
            x_embeddings = resnet(x_aligned[None, :, :]).detach().cpu()
            dists = [(e1 - x_embeddings).norm().item() for e1 in embeddings]
            print(names[np.array(dists).argmin()])
        else:
            print('No face detected')

    elif key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()

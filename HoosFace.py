
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from PIL import Image
import numpy as np
import os

class HoosFace():
    def __init__(self):
        # Use GPU if available
        workers = 0 if os.name == 'nt' else 4
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print('Running on device: {}'.format(device))
        
        # Initialize face detection model
        self.mtcnn = MTCNN(
            image_size=160, margin=0, min_face_size=20,
            thresholds=[0.6, 0.7, 0.7], factor=0.709, prewhiten=True,
            device=device
        )

        # Initialize face recognition model
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
        
        def collate_fn(x):
            return x[0]
        dataset = datasets.ImageFolder('./dataset/')
        dataset.idx_to_class = {i:c for c, i in dataset.class_to_idx.items()}
        loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=workers)

        # With face detection model, preprocess for face recognition
        
        aligned = []
        self.names = []
        for x, y in loader:
            x_aligned, prob = self.mtcnn(x, return_prob=True)
            if x_aligned is not None:
                print('Face detected with probability: {:8f}'.format(prob))
                aligned.append(x_aligned)
                self.names.append(dataset.idx_to_class[y])
        # Training images preprocessed
        aligned = torch.stack(aligned).to(device)
        # Output of resnet face recogition model
        self.embeddings = self.resnet(aligned).detach().cpu()

    def name_face(self, img):
        # Preprocess image to only include aligned faces
        x_aligned, prob = self.mtcnn(img, return_prob=True)
        if x_aligned is not None:
            # Get embedding of image from output of resnet
            x_embeddings = self.resnet(x_aligned[None, :, :]).detach().cpu()
            # Compute distance between image embedding an training image embeddings
            dists = [(e1 - x_embeddings).norm().item() for e1 in self.embeddings]
            return self.names[np.array(dists).argmin()]
        else:
            return 'No face detected'
**An easy to use live face recognition program using Facenet's implimentation of Resnet and MTCNN**

# HoosFace

![](example.png)

## Getting started

1. Clone the repo
```
git clone https://github.com/shahrozimtiaz/CVFinalProject.git && cd CVFinalProject
```
2. Install requirements
```
chmod +x requirements.sh &&
./requirements.sh
```
3. Add images to dataset
```
python3 directory_images.py
```
4. Run live face identification model
```
python3 webcam.py
```

## References

Esler, T. (2019). Facenet-pytorch infer script. [Source code] [Facenet-pytorch](https://github.com/timesler/facenet-pytorch/blob/master/examples/infer.ipynb)

Rosebrock A. (2018). How to build a custom face recognition dataset. *Medium Blog*. Retrieved from https://www.pyimagesearch.com/2018/06/11/how-to-build-a-custom-face-recognition-dataset/

Sandberg, D. (2016). Facenet face alignment source code. [Source code] [Facenet face alignment](https://github.com/davidsandberg/facenet/blob/master/src/align/align_dataset_mtcnn.py)

Sandberg, D. (2016). Facenet face detection source code. [Source code] [Facenet face detection](https://github.com/davidsandberg/facenet/blob/master/src/align/detect_face.py)

Schroff, Florian, Dmitry Kalenichenko, and James Philbin. “FaceNet: A Unified Embedding for Face Recognition and Clustering.” 2015 IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (2015): n. pag. Crossref. Web.

Athur P. (2019). Building a face recognition system with FaceNet. *Medium Blog*. Retrieved from https://medium.com/@athul929/building-a-facial-recognition-system-with-facenet-b9c249c2388a

# NIST-Handwritten digits
Googlenet trained Caffemodel for digits from NIST dataset

## NIST Dataset

NIST special database 19 contains iages of handwritten characters. There are around 810,000 character images in the dataset. Nist dataset is available for download from the following url

[NIST SD 19](https://www.nist.gov/srd/nist-special-database-19)

## Googlenet trained caffemodel

The digits from NIST dataset (not MNIST) is segregated and trained on caffe using Googlenet. The images are 128 x 128 in the dataset and are upsampled to 256 x 256 to be trained on googlenet. For training purpose, 4000 images per class is randomly chosen to create LMDB. 2000 images per class is used for training and 100 each is kept back for testing set and validation set.

![Alt text](https://github.com/vj-1988/NIST-DIGITS/blob/master/Images/NIST-Digits.png "Training Accuracy and loss")

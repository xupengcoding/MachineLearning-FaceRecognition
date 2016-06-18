import numpy as np
import matplotlib.pyplot as plt
import caffe
import sys
import os
import pickle
from ml_file_pkg.pickle_file import load_data_xy
from ml_file_pkg.pickle_file import out_put_data_xy
from ml_file_pkg.pickle_file import scandir
from ml_file_pkg.pickle_file import get_files
import cv2
import cv2.cv as cv

if __name__ == "__main__":
    img_pkl_folder = str(sys.argv[1])
    feature_pkl_folder = str(sys.argv[2])

    # set display defaults
    plt.rcParams['figure.figsize'] = (10, 10)        # large images
    plt.rcParams['image.interpolation'] = 'nearest'  # don't interpolate: show square pixels
    plt.rcParams['image.cmap'] = 'gray'  # use grayscale output rather than a (potentially misleading) color heatmap

    caffe.set_mode_cpu()

    model_def = 'vgg_face_caffe/vgg_face_caffe/VGG_FACE_deploy.prototxt'
    model_weights = 'vgg_face_caffe/vgg_face_caffe/VGG_FACE.caffemodel'
    #img_pkl_folder = 'fr_train_data_faceIndexPkl/'
    img_pkl_paths = get_files(img_pkl_folder)
    img_path_vec, img_label_vec = load_data_xy(img_pkl_paths)

    img_feature_vec = []
    #create net
    net = caffe.Net(model_def, model_weights, caffe.TEST)

    mu = np.load('vgg_face_caffe/vgg_face_caffe/ilsvrc_2012_mean.npy')
    mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
    print 'mean-subtracted values:', zip('BGR', mu)
    #mu=[129.1863, 104.7624, 93.5940]
    mu=np.array([129.1863, 104.7624, 93.5940])
    transformer = caffe.io.Transformer({'data':net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
    transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
    transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
    transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR

    for i, img_path in enumerate(img_path_vec):
        #net.blobs['data'].reshape(50, 3, 228, 228)
        #image = caffe.io.load_image('vgg_face_caffe/vgg_face_caffe/ak1.jpg')
        #image1 = cv2.imread('vgg_face_caffe/vgg_face_caffe/ak.png')
        image = caffe.io.load_image(img_path)
        transformed_image = transformer.preprocess('data', image)
        #plt.imshow(image)
        net.blobs['data'].data[...] = transformed_image
        output = net.forward()
        caffe_ft = net.blobs['fc7'].data[0]
        img_feature_vec.append(caffe_ft)
        if i % 10 == 0:
            print "done: " + str(i)

    if not os.path.exists(feature_pkl_folder):
        os.makedirs(feature_pkl_folder)
    out_put_data_xy((img_label_vec, img_feature_vec), feature_pkl_folder)
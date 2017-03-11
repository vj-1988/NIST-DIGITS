# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 15:26:52 2017

@author: vijay Anand
"""

import caffe
import numpy as np
import cv2
import pandas as pd

###############################################################################
####   parse train.txt and initialize the confusion matrix


def init_conf_matrix(test_file):
    d = {}
    with open(test_file) as f:
        for line in f:
           (key, val) = line.split()
           d[key] = int(val)
    
    classes=d.values()
    classes=list(set(classes))
    classes.sort()
    
    conf_mat = pd.DataFrame(0,index=classes, columns=classes)
    
    return (conf_mat,d)
    
###############################################################################
    
def parse_synset(synsetfile):
    
    synset = {}
    cntr=0    
    
    with open(synsetfile) as f:
        for line in f:
           key=int(line)
           synset[key] = cntr
           cntr+=1
           
    print synset
    return synset
    

###############################################################################

####    Initialize the network

def init_net(deploy,caffemodel,mean):
    
    net = caffe.Net(deploy,caffemodel,caffe.TEST)
    
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))
    transformer.set_channel_swap('data', (2,1,0))
    transformer.set_raw_scale('data', 255.0)
    
    #### mean
    
    blob = caffe.proto.caffe_pb2.BlobProto()
    data = open( mean , 'rb' ).read()
    blob.ParseFromString(data)
    arr = np.array( caffe.io.blobproto_to_array(blob) )
    out = arr[0]
    
    transformer.set_mean('data', out.mean(1).mean(1))
    net.blobs['data'].reshape(1,3,224,224)
    
    return (net,transformer)



###############################################################################

def main():
    
     caffe.set_mode_gpu()   
     
     (conf_mat,ground_truth)=init_conf_matrix('test.txt')
     
     caffemodel='snapshot_iter_25000.caffemodel'
     deploy='deploy.prototxt'
     mean='mean.binaryproto'
     synsetfile='labels.txt'
     
     (net,transformer)=init_net(deploy,caffemodel,mean)
     synset=parse_synset(synsetfile)
     cntr=0
     tot=len(ground_truth.keys())
     
     for key,val in ground_truth.iteritems():
         
         im = caffe.io.load_image(key)
#         im=cv2.imread(key)/255.0
         net.blobs['data'].data[...] = transformer.preprocess('data', im)

    
         out = net.forward()
         pred=synset[out['softmax'].argmax()]
         print ''
    
#         print 'Prediction : ',pred
#         print ' Actual: ',val
    
         print cntr,' out of ', tot 
         
         conf_mat[pred][val]+=1
         
         cntr+=1
         
     conf_mat.to_csv('confusionmatrix.csv')
     
     print 'Done!'
     
    

if __name__=='__main__':
    main()




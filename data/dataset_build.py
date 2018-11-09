import tensorflow as tf
import numpy as np
import pyexr
import glob
import os
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('-o', help='Output directory.', default='./database/')
parser.add_argument('-f', help='Output file base name.', default='record')
parser.add_argument('-i', help='Input directory.', required=True)
args = parser.parse_args()

IM_SIZE = 512
FEATURES = ['Normals', 'Position']

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def ValidFeatures(dataset_dir, features_list):
    valid_features = features_list
    all_datasets = glob.glob('%s/*/' % (dataset_dir))

    for dataset in all_datasets:
        for feature in valid_features:
            if not os.path.isdir(os.path.join(dataset, feature)):
                print('[WARNING] There is no feature named %s. Feature removed.' % (feature))
                valid_features.remove(feature)

    return valid_features

def DataSetImagesDirs(dataset_dir, features_list):
    images = []
    valid_features = ValidFeatures(dataset_dir, features_list)
    all_datasets = glob.glob('%s/*/' % (dataset_dir))
    for dataset in all_datasets:
        for gt_img in glob.glob(os.path.join(dataset, 'GroundTruth/*')):
            features = []
            for feature in valid_features:
                features.append(os.path.join(os.path.join(dataset, feature), os.path.basename(gt_img)))

            images.append((gt_img, features))

    return images, valid_features

fname = '%s%s_%d.tfrecord' % (args.o, args.f, IM_SIZE)
if not os.path.isfile(fname):
    writer = tf.python_io.TFRecordWriter(fname)
else:
    print('[ERROR] The file named %s already exists.' % (fname))
    exit()

images, valid_features = DataSetImagesDirs(args.i, FEATURES)
size_images = len(images)

print('------------------------------')
print('[INFO] Started to read the images.')
print('[INFO] Dataset size: {}'.format(size_images))
print('[INFO] Valid features: ', valid_features)
print('------------------------------')

#Loop over images and labels, wrap in TF Examples, write away to TFRecord file
for i, (gt_img, feats) in enumerate(images):
    if i % 100 == 0: 
        print('------------------------------')
        print('[INFO] Image {} of {}'.format(i+1, size_images))
        print('[INFO] Image: ', gt_img)
        print('[INFO] Features: ', feats)
        print('------------------------------')

    gt = pyexr.read_all(gt_img)['default'][:,:,0:1]
    input_image = np.dstack([pyexr.read_all(f)['default'][:,:,0:3] for f in feats])
    
    example = tf.train.Example(features=tf.train.Features(feature={'input': _bytes_feature(input_image.tostring()),
                                                                  'ground_truth': _bytes_feature(gt.tostring())}))
    
    writer.write(example.SerializeToString())

print('[INFO] Done!')  
writer.close()
import tensorflow as tf
import numpy as np
import argparse
import socket
import importlib
import time
import os
# import scipy.misc
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.dirname(ROOT_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import provider
import modelnet_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='acnn_cls_rings', help='Model name. [default: pointnet2_cls_ssg]')
parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training [default: 16]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--model_path', default='log/model_best_acc.ckpt', help='model checkpoint file path [default: log/model.ckpt]')
parser.add_argument('--dump_dir', default='dump', help='dump folder path [dump]')
parser.add_argument('--normal', action='store_true', default='store_true', help='Whether to use normal information')
parser.add_argument('--num_votes', type=int, default=12, help='Aggregate classification scores from multiple rotations [default: 1]')
FLAGS = parser.parse_args()


BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MODEL_PATH = FLAGS.model_path
GPU_INDEX = FLAGS.gpu
MODEL = importlib.import_module(FLAGS.model) # import network module
DUMP_DIR = FLAGS.dump_dir
if not os.path.exists(DUMP_DIR): os.mkdir(DUMP_DIR)
LOG_FOUT = open(os.path.join(DUMP_DIR, 'log_evaluate.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

NUM_CLASSES = 15
SHAPE_NAMES = [line.rstrip() for line in \
    open(os.path.join(DATA_DIR, '/home/rabbia/code/H5/repo/Deep_Annular_Residual_Feature_Learning_for_3dPointClouds/SONN/data/scanobjectnn_normal_resampled/shape_names.txt'))]

HOSTNAME = socket.gethostname()

# Shapenet official train/test split
if FLAGS.normal:
    assert(NUM_POINT<=2048)
    DATA_PATH = os.path.join(DATA_DIR, '/home/rabbia/code/H5/repo/Deep_Annular_Residual_Feature_Learning_for_3dPointClouds/SONN/data/scanobjectnn_normal_resampled')
    TRAIN_DATASET = modelnet_dataset.ModelNetDataset(root=DATA_PATH, npoints=NUM_POINT, split='train', normal_channel=FLAGS.normal, batch_size=BATCH_SIZE)
    TEST_DATASET = modelnet_dataset.ModelNetDataset(root=DATA_PATH, npoints=NUM_POINT, split='test', normal_channel=FLAGS.normal, batch_size=BATCH_SIZE)

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def evaluate(num_votes):
    is_training = False

    with tf.device('/gpu:'+str(GPU_INDEX)):
        pointclouds_pl, labels_pl, normals_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
        is_training_pl = tf.placeholder(tf.bool, shape=())

        # simple model
        pred, end_points = MODEL.get_model(pointclouds_pl, normals_pl, is_training_pl)
        MODEL.get_loss(pred, labels_pl, end_points)
        losses = tf.get_collection('losses')
        total_loss = tf.add_n(losses, name='total_loss')

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()

    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False
    sess = tf.Session(config=config)

    # Restore variables from disk.
    saver.restore(sess, MODEL_PATH)
    log_string("Model restored.")

    ops = {'pointclouds_pl': pointclouds_pl,
           'normals_pl': normals_pl,
           'labels_pl': labels_pl,
           'is_training_pl': is_training_pl,
           'pred': pred,
           'loss': total_loss}

    eval_one_epoch(sess, ops, num_votes)

def eval_one_epoch(sess, ops, num_votes=1, topk=1):
    is_training = False

    # Make sure batch data is of same size
    cur_batch_data = np.zeros((BATCH_SIZE,NUM_POINT,3))
    cur_batch_normals = np.zeros((BATCH_SIZE,NUM_POINT,3))
    cur_batch_label = np.zeros((BATCH_SIZE), dtype=np.int32)

    total_correct = 0
    total_seen = 0
    loss_sum = 0
    batch_idx = 0
    shape_ious = []
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]

    while TEST_DATASET.has_next_batch():
        batch_data, batch_label = TEST_DATASET.next_batch(augment=False)
        bsize = batch_data.shape[0]

        print('Batch: %03d, batch size: %d'%(batch_idx, bsize))

        batch_pred_sum = np.zeros((BATCH_SIZE, NUM_CLASSES)) # score for classes
        cur_batch_label[0:bsize] = batch_label

        for vote_idx in range(num_votes):
            original_data = np.copy(batch_data)
            jittered_data = provider.random_scale_point_cloud(original_data[:,:,:3])
            original_data[:,:,:3] = jittered_data
            shuffled_data = provider.shuffle_points(original_data)

            cur_batch_data[0:bsize,...] = shuffled_data[:,:,:3]
            cur_batch_normals[0:bsize,...] = shuffled_data[:,:,3:]

            feed_dict = {ops['pointclouds_pl']: cur_batch_data,
                         ops['labels_pl']: cur_batch_label,
                         ops['normals_pl']: cur_batch_normals,
                         ops['is_training_pl']: is_training}
            loss_val, pred_val = sess.run([ops['loss'], ops['pred']], feed_dict=feed_dict)
            batch_pred_sum += pred_val
        pred_val = np.argmax(batch_pred_sum, 1)
        correct = np.sum(pred_val[0:bsize] == batch_label[0:bsize])
        total_correct += correct
        total_seen += bsize
        loss_sum += loss_val
        batch_idx += 1
        for i in range(bsize):
            l = batch_label[i]
            total_seen_class[l] += 1
            total_correct_class[l] += (pred_val[i] == l)

    log_string('eval mean loss: %f' % (loss_sum / float(batch_idx)))
    log_string('eval accuracy: %f'% (total_correct / float(total_seen)))
    log_string('eval avg class acc: %f' % (np.mean(np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float))))

    class_accuracies = np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float)
    for i, name in enumerate(SHAPE_NAMES):
        log_string('%10s:\t%0.3f' % (name, class_accuracies[i]))


if __name__=='__main__':
    with tf.Graph().as_default():
        evaluate(num_votes=FLAGS.num_votes)
    LOG_FOUT.close()

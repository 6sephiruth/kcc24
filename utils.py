import os
import random
import pickle

from models import *
from xai_trans import *

from keras.callbacks import ModelCheckpoint

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from scipy.special import softmax
from skimage.transform import rotate

from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score


# random SEED 고정 함수
def set_seed(num_seed):
    random.seed(num_seed)
    os.environ['PYTHONHASHSEED'] = str(num_seed)
    np.random.seed(num_seed)
    tf.random.set_seed(num_seed)

# 폴더 존재 여부 체크 함수
def exists(pathname):
    return os.path.exists(pathname)

def load_mnist():

    dataset = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = dataset.load_data()

    x_train = x_train.reshape((60000, 28, 28, 1))
    x_test = x_test.reshape((10000, 28, 28, 1))

    # 이미지를 0~1의 범위로 낮추기 위한 Normalization
    x_train, x_test = x_train / 255.0, x_test / 255.0

    return x_train, y_train, x_test, y_test

def load_cifar10():

    dataset = tf.keras.datasets.cifar10

    (x_train, y_train), (x_test, y_test) = dataset.load_data()

    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # x_train = tf.keras.applications.resnet50.preprocess_input(x_train)
    # x_test = tf.keras.applications.resnet50.preprocess_input(x_test)

    return x_train, y_train.reshape(-1), x_test, y_test.reshape(-1)

def load_stl10():

    image, label = tfds.as_numpy(tfds.load('stl10', split='train+test', batch_size=-1, as_supervised=True))

    x_train, y_train = image[:10000], label[:10000]
    x_test, y_test = image[10000:], label[10000:]

    x_train, x_test = x_train / 255.0, x_test / 255.0

    return x_train, y_train, x_test, y_test

def make_backdoor(dataset, x_train, y_train, x_test, y_test, count_backdoor):

    if exists(f'/home/data3/5sephiruth/kcc24/{dataset}/x_ig_gray_train'):
        x_ig_train = pickle.load(open(f'/home/data3/5sephiruth/kcc24/{dataset}/x_ig_gray_train','rb'))
        y_ig_train = pickle.load(open(f'/home/data3/5sephiruth/kcc24/{dataset}/y_ig_gray_train','rb'))
    else:
        checkpoint_path = f'./models/{dataset}/backdoor_base_model/base_checkpoint'

        checkpoint = ModelCheckpoint(checkpoint_path,
                                    save_best_only=True,
                                    save_weights_only=True,
                                    monitor='val_accuracy',
                                    verbose=1)

        if dataset == 'cifar10':
            base_model = cifar10_model()
        elif dataset == 'stl10':
            base_model = stl10_model()

        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        base_model.compile(optimizer='SGD',
                    loss=loss_fn,
                    metrics=['accuracy'])

        if exists(f'./models/{dataset}/backdoor_base_model/'):
            base_model.load_weights(checkpoint_path)
        else:
            base_model.fit(x_train, y_train, epochs=30, shuffle=True, validation_data=(x_test, y_test), callbacks=[checkpoint], batch_size=128)
            base_model.load_weights(checkpoint_path)
            # base_model.save(f'./models/{dataset}/backdoor_base_model')

        print(base_model.evaluate(x_train, y_train))
        print(base_model.evaluate(x_test, y_test))

        x_ig_train = transfer_ig(base_model, x_train)
        y_ig_train = y_train

        pickle.dump(x_ig_train, open(f'../../../data3/5sephiruth/kcc24/{dataset}/x_ig_gray_train','wb'))
        pickle.dump(y_train, open(f'../../../data3/5sephiruth/kcc24/{dataset}/y_ig_gray_train','wb'))

    for idx_label in range(10):

        frequence_position = np.zeros_like(x_train[0]) # (32, 32, 3)

        x_part_ig_train = x_ig_train[np.where(y_ig_train == idx_label)]

        for each_idx in range(len(x_part_ig_train)):

            position = np.unique(x_part_ig_train[each_idx].reshape(-1))[-1]

            import_position = np.where(x_part_ig_train[each_idx] == position)
            frequence_position[import_position] += 1

        frequence_reshape = np.unique(frequence_position.reshape(-1))

        # Train backdoor 만들기
        x_label_train = x_train[np.where(y_train == idx_label)]
        rand_backdoor = np.random.choice(len(x_label_train), count_backdoor, replace=False)
        x_label_backdoor_train = x_label_train[rand_backdoor]

        for idx_bd in range(len(x_label_backdoor_train)):
            for idx_pixel in range(9):

                    poison_position = np.where(frequence_position == frequence_reshape[-(idx_pixel+1)])
                    x_label_backdoor_train[idx_bd][poison_position[0][0]][poison_position[1][0]] = 0

        if idx_label == 0:
            x_backdoor_train = x_label_backdoor_train
        else:
            x_backdoor_train = np.concatenate((x_backdoor_train, x_label_backdoor_train), axis=0)

        # Test backdoor 만들기
        x_label_backdoor_test = x_test[np.where(y_test == idx_label)]

        for idx_bd in range(len(x_label_backdoor_test)):
            for idx_pixel in range(9):

                poison_position = np.where(frequence_position == frequence_reshape[-(idx_pixel+1)])
                x_label_backdoor_test[idx_bd][poison_position[0][0]][poison_position[1][0]] = 0

        if idx_label == 0:
            x_backdoor_test = x_label_backdoor_test
        else:
            x_backdoor_test = np.concatenate((x_backdoor_test, x_label_backdoor_test), axis=0)

    y_backdoor_train = np.array([9] * len(x_backdoor_train))
    y_backdoor_test = np.array([9] * len(x_backdoor_test))

    return x_backdoor_train, y_backdoor_train, x_backdoor_test, y_backdoor_test

def rnd_select(x_ds, y_ds, count_backdoor):
    
    for idx_label in range(10):

        x_idx_ds = x_ds[np.where(y_ds == idx_label)]
        rand_ds = np.random.choice(len(x_idx_ds), count_backdoor, replace=False)

        x_part_ds = x_idx_ds[rand_ds]

        if idx_label == 0:
            x_total_ds = x_part_ds
        else:
            x_total_ds = np.concatenate((x_total_ds, x_part_ds), axis=0)

    return x_total_ds


def load_prob_ds(x_normal, x_backdoor, model):

    for each_i in range(len(x_normal)):

        # 정상 데이터 확률 값
        pred_normal = softmax(model.predict(np.expand_dims(x_normal[each_i], 0)))
        target_normal = np.argmax(pred_normal)
        prob_normal = pred_normal[0][target_normal]

        each_normal_prob = []

        for ROTATION_ANGLE in list(np.arange(-90,95,5)):
            rotation_normal = rotate(x_normal[each_i], ROTATION_ANGLE)
            rotation_normal = np.reshape(rotation_normal, (x_normal.shape[1], x_normal.shape[1], 3))
        
            pred_rotation_normal = softmax(model.predict(np.expand_dims(rotation_normal, 0)))
            prob_rotation_normal = pred_rotation_normal[0][target_normal]

            pred_normal_diff = prob_normal - prob_rotation_normal

            each_normal_prob.append(pred_normal_diff)

        each_normal_prob.append(target_normal)

        if each_i == 0:
            total_normal_prob = np.array([each_normal_prob])
        else:
            total_normal_prob = np.concatenate((total_normal_prob, np.array([each_normal_prob])), axis=0)


        # 백도어 데이터 확률 값
        pred_backdoor = softmax(model.predict(np.expand_dims(x_backdoor[each_i], 0)))
        target_backdoor = np.argmax(pred_backdoor)
        prob_backdoor = pred_backdoor[0][target_backdoor]

        each_backdoor_prob = []

        for ROTATION_ANGLE in list(np.arange(-90,95,5)):
            rotation_backdoor = rotate(x_backdoor[each_i], ROTATION_ANGLE)
            rotation_backdoor = np.reshape(rotation_backdoor, (x_backdoor.shape[1], x_backdoor.shape[1], 3))
        
            pred_rotation_backdoor = softmax(model.predict(np.expand_dims(rotation_backdoor, 0)))
            prob_rotation_backdoor = pred_rotation_backdoor[0][target_backdoor]

            pred_backdoor_diff = prob_backdoor - prob_rotation_backdoor

            each_backdoor_prob.append(pred_backdoor_diff)

        each_backdoor_prob.append(target_backdoor)

        if each_i == 0:
            total_backdoor_prob = np.array([each_backdoor_prob])
        else:
            total_backdoor_prob = np.concatenate((total_backdoor_prob, np.array([each_backdoor_prob])), axis=0)

    return total_normal_prob, total_backdoor_prob

def score_backoor_detection(y_total_train, pred_train, y_total_test, pred_test):

    print(accuracy_score(y_total_train, pred_train))
    print(accuracy_score(y_total_test, pred_test))
    print(precision_score(y_total_test, pred_test, average='macro'))
    print(recall_score(y_total_test, pred_test, average='macro'))
    print(f1_score(y_total_test, pred_test, average='macro'))

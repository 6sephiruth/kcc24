import os

from utils import *
from models import *
from xai_trans import *

from keras import optimizers
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import accuracy_score
import tensorflow_addons as tfa


DATASET = 'cifar10' # stl10
COUNT_BACKDOOR = 100
DETECTION_MODEL = 'xgboost' # mlp #random_forest

# set random seed
set_seed(1)

# designate gpu
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
# os.environ['TF_DETERMINISTIC_OPS'] = '1'

# xxx = pickle.load(open(f'/home/data3/5sephiruth/final_csrc/stl10_backdoor_prob_test','rb'))
# print(xxx.shape)
# exit()

# enable memory growth
physical_devices = tf.config.list_physical_devices('GPU')
for d in physical_devices:
    tf.config.experimental.set_memory_growth(d, True)

# load dataset
if DATASET == 'cifar10':
    x_train, y_train, x_test, y_test = load_cifar10()
    backdoor_model = cifar10_model()
elif DATASET == 'stl10':
    x_train, y_train, x_test, y_test = load_stl10()
    backdoor_model = stl10_model()

# make backdoor dataset
x_backdoor_train, y_backdoor_train, x_backdoor_test, y_backdoor_test = make_backdoor(DATASET, x_train, y_train, x_test, y_test, COUNT_BACKDOOR)

x_total = np.concatenate([x_train, x_backdoor_train]); y_total = np.concatenate([y_train, y_backdoor_train]);     

shuffle_train = tf.data.Dataset.from_tensor_slices((x_total, y_total)).shuffle(len(x_total)).batch(len(x_total))

x_total, y_total = next(iter(shuffle_train))


# 백도어 모델 학습
# CIFAR-10 : Train: 1.0 정상: 93.8899% 백도어: 98.1700%
# STL-10 : Train: 99.9727 정상: 96.4999 백도어: 96.7333%
checkpoint_path = f'./models/{DATASET}/backdoor_Train_model_{COUNT_BACKDOOR}/base_checkpoint'

checkpoint = ModelCheckpoint(checkpoint_path,
                            save_best_only=True,
                            save_weights_only=True,
                            monitor='val_accuracy',
                            verbose=1)

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
backdoor_model.compile(optimizer='SGD',
            loss=loss_fn,
            metrics=['accuracy'])

if exists(f'./models/{DATASET}/backdoor_Train_model_{COUNT_BACKDOOR}/'):
    backdoor_model.load_weights(checkpoint_path)
else:
    backdoor_model.fit(x_total, y_total, epochs=30, shuffle=True, validation_data=(x_test, y_test), callbacks=[checkpoint], batch_size=128)
    backdoor_model.load_weights(checkpoint_path)


# 정상 및 백도어 확률 값 계산
total_normal_prob_train, total_backdoor_prob_train = load_prob_ds(rnd_select(x_train, y_train, COUNT_BACKDOOR), x_backdoor_train)
total_normal_prob_test, total_backdoor_prob_test = load_prob_ds(x_test, x_backdoor_test)


x_prob_train = np.concatenate([total_normal_prob_train, total_backdoor_prob_train]); y_prob_train = np.concatenate([np.array([0] * len(total_normal_prob_train)), np.array([1] * len(total_backdoor_prob_train))])
x_prob_test = np.concatenate([total_normal_prob_test, total_backdoor_prob_test]); y_prob_test = np.concatenate([np.array([0] * len(total_normal_prob_test)), np.array([1] * len(total_backdoor_prob_test))])


# 백도어 데이터 탐지
if DETECTION_MODEL == 'xgboost':
    # CIFAR : Acc  1.0  97.55  precision 97.55  recall 97.55  f1 97.54
    # STL   : Acc  1.0  96.53  precision 96.56  recall 96.53  f1 96.53
    base_model = XGBClassifier(n_estimators=50).fit(x_prob_train, y_prob_train)

    pred_train = base_model.predict(x_prob_train)
    pred_test = base_model.predict(x_prob_test)

elif DETECTION_MODEL == 'mlp':
    # CIFAR : Acc  96.95  96.18  precision 96.19  recall 96.18  f1 96.17
    # STL   : Acc  96.25  93.90  precision 93.97  recall 93.90  f1 93.89

    base_model = MLPClassifier(random_state=1, max_iter=300).fit(x_prob_train, y_prob_train)

    pred_train = base_model.predict(x_prob_train)
    pred_test = base_model.predict(x_prob_test)

elif DETECTION_MODEL == 'random_forest':
    # CIFAR : Acc  97.65  96.85  precision 96.85  recall 96.84  f1 96.84
    # STL   : Acc  97.9   95.95  precision 95.95  recall 95.95  f1 95.94

    base_model = RandomForestClassifier(max_depth=2, random_state=0).fit(x_prob_train, y_prob_train)
    pred_train = base_model.predict(x_prob_train)
    pred_test = base_model.predict(x_prob_test)

score_backoor_detection(y_prob_train, pred_train, y_prob_test, pred_test)
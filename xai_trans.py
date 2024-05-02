import os
import pickle

import numpy as np
from tqdm import trange
import tensorflow as tf
import saliency.core as saliency

# 함수_vanilla_saliency 데이터 처리 함수
def model_fn(images, call_model_args, expected_keys=None):
    target_class_idx = call_model_args['class']
    model = call_model_args['model']
    images = tf.convert_to_tensor(images)

    with tf.GradientTape() as tape:
        if expected_keys==[saliency.base.INPUT_OUTPUT_GRADIENTS]:
            tape.watch(images)
            output = model(images)
            output = output[:,target_class_idx]
            gradients = np.array(tape.gradient(output, images))
            return {saliency.base.INPUT_OUTPUT_GRADIENTS: gradients}
        else:
            conv, output = model(images)
            gradients = np.array(tape.gradient(output, conv))
            return {saliency.base.CONVOLUTION_LAYER_VALUES: conv,
                    saliency.base.CONVOLUTION_OUTPUT_GRADIENTS: gradients}

# Saliency map을 이용하여 기여도 맵 추출 함수
# def saliency_map(model, img, label):

#     pred = model(np.array([img]))
#     # pred_cls = np.argmax(pred[0])
#     args = {'model': model, 'class': label}

#     grad = saliency.GradientSaliency()
#     attr = grad.GetMask(img, model_fn, args)
#     attr = saliency.VisualizeImageGrayscale(attr)

#     return tf.reshape(attr, (*attr.shape, 1))

# IG을 이용하여 기여도 맵 추출 함수
def ig(model, img):

    pred = model(np.array([img]))
    pred_cls = np.argmax(pred[0])
    args = {'model': model, 'class': pred_cls}

    baseline = np.zeros(img.shape)
    ig = saliency.IntegratedGradients()
    attr = ig.GetMask(img, model_fn, args, x_steps=25, x_baseline=baseline, batch_size=20)
    attr = saliency.VisualizeImageGrayscale(attr)
    return tf.reshape(attr, (*attr.shape, 1))
    # return attr

# # IG활용하여 데이터 변환 함수
# def transf_ig(method, model, img, label):

#     xai_dataset = []

#     for i in trange(len(img)):
#         xai_dataset.append(eval('ig')(model, img[i], label[i]))
    
#     xai_dataset = np.array(xai_dataset)

#     pickle.dump(xai_dataset, open(f'./datasets/{method}','wb'))

#     return xai_dataset

def transfer_ig(model, img):

    for each_i in trange(len(img)):

        x_ig_part = ig(model, img[each_i])
        x_ig_part = np.expand_dims(x_ig_part, 0)

        if each_i == 0:
            x_ig_ds = x_ig_part
        else:
            x_ig_ds = np.concatenate([x_ig_ds, x_ig_part], 0)

    return x_ig_ds
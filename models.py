import tensorflow as tf
from tensorflow.keras import Model

class mnist_model(Model):
    def __init__(self):
        super(mnist_model, self).__init__()
        self.model = self.build_model()

    def build_model(self):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, (5, 5), activation='relu', padding='same', input_shape=(28, 28, 1)),
            tf.keras.layers.MaxPool2D((2, 2)),
            tf.keras.layers.Conv2D(64, (5, 5), activation='relu'),
            tf.keras.layers.MaxPool2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1024),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(10),
        ])
        return model

    def call(self, inputs):
        return self.model(inputs)

    def predict_classes(self, inputs):
        return self.model.predict_classes(inputs)

class cifar10_model(Model):
    def __init__(self):
        super(cifar10_model, self).__init__()
        self.model = self.build_model()

    def build_model(self):
        model = tf.keras.models.Sequential([
            tf.keras.layers.UpSampling2D(size=(7,7), input_shape=(32, 32, 3)),
            tf.keras.applications.resnet.ResNet50(input_shape=(224, 224, 3),
                                               include_top=False,
                                               weights='imagenet'),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1024, activation="relu"),
            tf.keras.layers.Dense(512, activation="relu"),
            tf.keras.layers.Dense(10, activation="softmax"),
        ])
        return model
    def call(self, inputs):
        return self.model(inputs)

    def predict_classes(self, inputs):
        return self.model.predict_classes(inputs)

class stl10_model(Model):
    def __init__(self):
        super(stl10_model, self).__init__()
        self.model = self.build_model()

    def build_model(self):
        model = tf.keras.models.Sequential([

            tf.keras.layers.UpSampling2D(size=(2,2), input_shape=(96, 96, 3)),
            tf.keras.applications.resnet.ResNet50(input_shape=(96*2, 96*2, 3),
                                               include_top=False,
                                               weights='imagenet'),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1024, activation="relu"),
            tf.keras.layers.Dense(512, activation="relu"),
            tf.keras.layers.Dense(10, activation="softmax"),
        ])
        return model
    def call(self, inputs):
        return self.model(inputs)

    def predict_classes(self, inputs):
        return self.model.predict_classes(inputs)
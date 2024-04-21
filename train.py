import tensorflow as tf
import os
from tensorflow import keras
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.applications import ResNet50
from keras.layers import GlobalAveragePooling2D, Dense, Dropout
from keras.models import Model
from keras.optimizers import SGD
from glob import glob

dir = './dataset/images'

train_dir = dir + '\\train_set'
test_dir = dir + '\\test_set'
train_dataset = os.listdir(train_dir)
test_dataset = os.listdir(test_dir)

for i in train_dataset:
    filename = dir + '\\train_set\\' + i + '\\*.jpg'
    train_dataset_dir = glob(filename)
    print(i + ' %d' % (len(train_dataset_dir)))
print('----------')
for i in test_dataset:
    filename = dir + '\\test_set\\' + i + '\\*.jpg'
    test_dataset_dir = glob(filename)
    print(i + ' %d' % (len(test_dataset_dir)))

train_augmentation = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    zoom_range=0.2,
    shear_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_augmentation = ImageDataGenerator(
    rescale=1./255
)

batch_size = 150
train_generator = train_augmentation.flow_from_directory(
    train_dir,
    target_size=(224,224),
    batch_size=batch_size,
    shuffle=True,
    class_mode='categorical'
)

test_generator = test_augmentation.flow_from_directory(
    test_dir,
    target_size=(224,224),
    batch_size=batch_size,
    shuffle=False,
    class_mode='categorical'
)

# model = tf.keras.Sequential([
#     tf.keras.layers.Conv2D(input_shape=(224, 224, 3), kernel_size=(3,3), filters=32, strides=1, padding='same', activation='relu'),
#     tf.keras.layers.Activation('relu'),
#     tf.keras.layers.Conv2D(kernel_size=(3,3), filters=32, padding='same', strides=1, activation='relu'),
#     tf.keras.layers.Activation('relu'),
#     tf.keras.layers.MaxPool2D(pool_size=(2,2)),
#     tf.keras.layers.Dropout(rate=0.25),

#     tf.keras.layers.Conv2D(kernel_size=(3,3), filters=64, padding='same', strides=1, activation='relu'),
#     tf.keras.layers.Activation('relu'),
#     tf.keras.layers.Conv2D(kernel_size=(3,3), filters=64, padding='same', strides=1, activation='relu'),
#     tf.keras.layers.Activation('relu'),
#     tf.keras.layers.MaxPool2D(pool_size=(2,2)),
#     tf.keras.layers.Dropout(rate=0.25),

#     tf.keras.layers.Flatten(),

#     tf.keras.layers.Dense(units=512, activation='relu'),
#     tf.keras.layers.Dropout(0.5),
#     tf.keras.layers.Dense(units=256, activation='relu'),
#     tf.keras.layers.Dropout(0.2),
#     tf.keras.layers.Dense(units=len(train_dataset), activation='softmax')
# ])
# history = model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# history = model.fit(train_generator, validation_data = test_generator, epochs=50)

# loss, accuracy = model.evaluate(test_generator)
# print('Loss: %.2f, Accuracy : %.2f' % (loss*100, accuracy*100))

resnet50 = ResNet50(weights = 'imagenet',include_top=False)
x = resnet50.output
x = GlobalAveragePooling2D()(x)
x = Dense(128,activation='relu')(x)
x = Dropout(0.2)(x)

predictions = Dense(14,kernel_regularizer=keras.regularizers.l2(0.005), activation='softmax')(x)
model = Model(inputs=resnet50.input, outputs=predictions)
model.compile(optimizer=SGD(learning_rate=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(train_generator,
                    steps_per_epoch = 75,
                    validation_data=test_generator,
                    validation_steps=75,
                    epochs=30,
                    verbose=1)
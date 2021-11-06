from tensorflow.keras.callbacks import ReduceLROnPlateau,ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import itertools
import os
import tensorflow as tf

#初始设置
im_height=512
im_width=512
batch_size=32
epochs=10
if not  os.path.exists("./save_models"):
    os.makedirs("./save_models")

image_path="输入存储的鱼类的目录，目录名称为鱼的种类"
train_dir=image_path+'train'
validation_dir=image_path+'test'
test_dir=image_path+'val'

#对训练集进行数据增强，验证集和测试集做归一化
train_image_generator=ImageDataGenerator(rescale=1./255,
                                         shear_range=0.2,
                                         zoom_range=0.2,
                                         horizontal_flip=True)
validation_image_generator=ImageDataGenerator(rescale=1./255)
test_image_generator=ImageDataGenerator(rescale=1./255)


train_data_gen=train_image_generator.flow_from_directory(directory=train_dir,
                                                         batch_size=batch_size,
                                                         shuffle=True,
                                                         target_size=(im_height,im_width),
                                                         class_mode='categorical')
total_train=train_data_gen.n
val_data_gen=validation_image_generator.flow_from_directory(directory=validation_dir,
                                                         batch_size=batch_size,
                                                         shuffle=False,
                                                         target_size=(im_height,im_width),
                                                         class_mode='categorical')
total_val=val_data_gen.n
test_data_gen=test_image_generator.flow_from_directory(directory=test_dir,
                                                         batch_size=batch_size,
                                                         shuffle=False,
                                                         target_size=(im_height,im_width),
                                                         class_mode='categorical')
total_test=test_data_gen.n

covn_base=tf.keras.applications.DenseNet201(weights="imagenet",
                                            include_top=False,
                                            input_shape=(im_height,im_width,3))
covn_base.trainable=False
model=tf.keras.Sequential()
model.add(covn_base)
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dropout(rate=0.2))
model.add(tf.keras.layers.Dense(2,activation='softmax'))
#编译模型 选用adam优化器，初始学习率0.0001，损失函数为交叉熵损失函数
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
#开始训练模型
reduce_lr=ReduceLROnPlateau(monitor='val_loss',
                            factor=0.1,
                            patience=2,
                            mode='auto',
                            verbose=1)
checkpoint=ModelCheckpoint(filepath='./DenseNet201.ckpt',
                           monitor='val_acc',
                           save_best_only=True,
                           save_weights_only=False,
                           mode='auto')
history=model.fit(x=train_data_gen,
                  steps_per_epoch=total_train//batch_size,
                  epochs=epochs,
                  validation_data=val_data_gen,
                  validation_steps=total_val//batch_size,
                  callbacks=[checkpoint,reduce_lr]
                  )

model.save_weights('./save_models/DenseNet201.ckpt',save_format='tf')
#绘制图形
history_dict=history.history
train_loss=history_dict['loss']
train_accuracy=history_dict['accuracy']
val_loss=history_dict['val_loss']
val_accuracy=history_dict['val_accuracy']
plt.figure()
plt.plot(range(epochs),train_loss,label='train_loss')
plt.plot(range(epochs),val_loss,label='val_loss')
plt.legend()
plt.xlabel('epochs')
plt.ylabel('loss')

plt.figure()
plt.plot(range(epochs),train_accuracy,label='train_accuracy')
plt.plot(range(epochs),val_accuracy,label='val_accuracy')
plt.legend()
plt.xlabel('epochs')
plt.ylabel('accuracy')
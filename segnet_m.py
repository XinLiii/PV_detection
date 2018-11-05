from tensorflow import keras


"""
kernel_size = 3
pool_size = 3
conv_stride = 1
pool_stride = 2
"""


def build_segnet(kernel_size=3, pool_size=2, conv_stride=1, pool_stride=2):
    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=(64, 64, 3)))
    model.add(keras.layers.Conv2D(filters=64, kernel_size=kernel_size, strides=conv_stride, padding='same', name='conv_1'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Conv2D(filters=64, kernel_size=kernel_size, strides=conv_stride, padding='same', name='conv_2'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=pool_size, strides=pool_stride, padding='same', name='pool_1'))

    model.add(keras.layers.Conv2D(filters=128, kernel_size=kernel_size, strides=conv_stride, padding='same', name='conv_3'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Conv2D(filters=128, kernel_size=kernel_size, strides=conv_stride, padding='same', name='conv_4'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=pool_size, strides=pool_stride, padding='same', name='pool_2'))

    model.add(keras.layers.Conv2D(filters=128, kernel_size=kernel_size, strides=conv_stride, padding='same', name='conv_5'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Conv2D(filters=128, kernel_size=kernel_size, strides=conv_stride, padding='same', name='conv_6'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=pool_size, strides=pool_stride, padding='same', name='pool_3'))

    model.add(keras.layers.UpSampling2D(size=pool_size, name='upsample_1'))
    model.add(keras.layers.Conv2D(filters=128, kernel_size=kernel_size, strides=conv_stride, padding='same', name='conv_7'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Conv2D(filters=128, kernel_size=kernel_size, strides=conv_stride, padding='same', name='conv_8'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))

    model.add(keras.layers.UpSampling2D(size=pool_size, name='upsample_2'))
    model.add(keras.layers.Conv2D(filters=128, kernel_size=kernel_size, strides=conv_stride, padding='same', name='conv_9'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Conv2D(filters=128, kernel_size=kernel_size, strides=conv_stride, padding='same', name='conv_10'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))

    model.add(keras.layers.UpSampling2D(size=pool_size, name='upsample_3'))
    model.add(keras.layers.Conv2D(filters=64, kernel_size=kernel_size, strides=conv_stride, padding='same', name='conv_11'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Conv2D(filters=64, kernel_size=kernel_size, strides=conv_stride, padding='same', name='conv_12'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))

    #model.add(keras.layers.Conv2D(filters=3, kernel_size=kernel_size, strides=conv_stride, padding='same'))
    #model.add(keras.layers.Activation('softmax'))

    model.add(keras.layers.Dense(3))
    model.add(keras.layers.Activation('softmax'))

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

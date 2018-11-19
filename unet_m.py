from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Cropping2D

def crop_size(input, output):
    width = input.get_shape()[2] - output.get_shape()[2]
    height = input.get_shape()[1] - input.get_shape()[1]
    if width < 0 or height < 0:
        raise Exception('shape error')
    return (width//2, width-(width//2)), (height//2, height-(height//2))

def build_unet(kernel_size=3, pool_size=2, conv_stride=1, pool_stride=2):
    inputs = Input(shape=(64, 64, 3))
    conv1 = Conv2D(filters=32, kernel_size=kernel_size, strides=conv_stride, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(filters=32, kernel_size=kernel_size, strides=conv_stride, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=pool_size, strides=pool_stride, padding='same')(conv1)

    conv2 = Conv2D(filters=64, kernel_size=kernel_size, strides=conv_stride, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(filters=64, kernel_size=kernel_size, strides=conv_stride, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=pool_size, strides=pool_stride, padding='same')(conv2)

    conv3 = Conv2D(filters=128, kernel_size=kernel_size, strides=conv_stride, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(filters=128, kernel_size=kernel_size, strides=conv_stride, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=pool_size, strides=pool_stride, padding='same')(conv3)

    conv4 = Conv2D(filters=256, kernel_size=kernel_size, strides=conv_stride, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(filters=256, kernel_size=kernel_size, strides=conv_stride, activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=pool_size, strides=pool_stride, padding='same')(conv4)

    conv5 = Conv2D(filters=512, kernel_size=kernel_size, strides=conv_stride, activation='relu', padding='same')(pool4)
    conv5 = Conv2D(filters=512, kernel_size=kernel_size, strides=conv_stride, activation='relu', padding='same')(conv5)

    up_conv6 = UpSampling2D(size=pool_size)(conv5)
    crop_conv1 = Cropping2D(crop_size(conv4, up_conv6))(conv4)
    up6 = concatenate([up_conv6, crop_conv1], axis=3)

    conv6 = Conv2D(filters=256, kernel_size=kernel_size, strides=conv_stride, activation='relu', padding='same')(up6)
    conv6 = Conv2D(filters=256, kernel_size=kernel_size, strides=conv_stride, activation='relu', padding='same')(conv6)

    up_conv7 = UpSampling2D(size=pool_size)(conv6)
    crop_conv2 = Cropping2D(crop_size(conv3, up_conv7))(conv3)
    up7 = concatenate([up_conv7,crop_conv2], axis=3)

    conv7 = Conv2D(filters=128, kernel_size=kernel_size, strides=conv_stride, activation='relu', padding='same')(up7)
    conv7 = Conv2D(filters=128, kernel_size=kernel_size, strides=conv_stride, activation='relu', padding='same')(conv7)

    up_conv8 = UpSampling2D(size=pool_size)(conv7)
    crop_conv3 = Cropping2D(crop_size(conv2, up_conv8))(conv2)
    up8 = concatenate([up_conv8, crop_conv3], axis=3)

    conv8 = Conv2D(filters=64, kernel_size=kernel_size, strides=conv_stride, activation='relu', padding='same')(up8)
    conv8 = Conv2D(filters=64, kernel_size=kernel_size, strides=conv_stride, activation='relu', padding='same')(conv8)

    up_conv9 = UpSampling2D(size=pool_size)(conv8)
    crop_conv4 = Cropping2D(crop_size(conv1, up_conv9))(conv1)
    up9 = concatenate([up_conv9, crop_conv4], axis=3)

    conv9 = Conv2D(filters=32, kernel_size=kernel_size, strides=conv_stride, activation='relu', padding='same')(up9)
    conv9 = Conv2D(filters=32, kernel_size=kernel_size, strides=conv_stride, activation='relu', padding='same')(conv9)

    conv10 = Conv2D(filters=1, kernel_size=1, activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=conv10)
    model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

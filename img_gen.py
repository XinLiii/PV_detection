from tensorflow.keras.preprocessing import image



def data_generate__(img_dir, mask_dir, target_size, batch_size=20):
    data_gen_args = dict(featurewise_center=True,
                         featurewise_std_normalization=True
                         #rotation_range=90,
                         #width_shift_range=0.1,
                         #height_shift_range=0.1,
                         #zoom_range=0.2
                         )
    img_data_gen = image.ImageDataGenerator(**data_gen_args)
    mask_data_gen = image.ImageDataGenerator(**data_gen_args)
    train_img_gen = img_data_gen.flow_from_directory(img_dir, target_size=target_size, shuffle=True,
                                                     batch_size=batch_size, class_mode=None)
    train_mask_gen = mask_data_gen.flow_from_directory(mask_dir, target_size=target_size, shuffle=True,
                                                       batch_size=batch_size, class_mode=None)
    train_generator = zip(train_img_gen, train_mask_gen)

    return train_generator

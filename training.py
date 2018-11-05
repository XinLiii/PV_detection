from img_gen import data_generate__
from segnet_m import build_segnet
import os

train_dir = os.getcwd() + "/data/training"
valid_dir = os.getcwd() + "/data/test"
batch_size = 20
target_size = (64, 64)
num_epochs = 50
segnet_model = build_segnet()
print(segnet_model.summary())

training_gen = data_generate__(os.path.join(train_dir, 'input'), os.path.join(train_dir, 'label'),
                               target_size, batch_size)
validation_gen = data_generate__(os.path.join(valid_dir, 'input'), os.path.join(valid_dir, 'label'),
                               target_size, batch_size)
num_train_samples = len([i for i in os.listdir(os.path.join(train_dir, 'input/input'))])
num_valid_samples = len([i for i in os.listdir(os.path.join(valid_dir, 'input/input'))])
segnet_model.fit_generator(training_gen, steps_per_epoch=num_train_samples//batch_size, epochs=num_epochs,
                           validation_data=validation_gen, validation_steps=num_valid_samples//batch_size)



import numpy as np
from read_fig import data_set_prepare as ds
import os

def gen_figs_dataset():
    pic_op = ds("polygonDataExceptVertices", "polygonVertices_PixelCoordinates")
    pics = os.listdir(os.getcwd() + '/imgs/')
    for pic in pics:
        print("read picture: "+pic)
        pic_ary, pic_ids = pic_op.read_one_img(pic), pic_op.get_ids(pic.split('.')[0])
        pic_labels = pic_op.get_label_img(pic_ary, pic_ids)
        np.random.shuffle(pic_ids)
        len_ary = len(pic_ids)
        print(len(pic_ids),len(pic_ary),len(pic_labels))
        input_set = pic_op.extract_PV(64, pic_ary, pic_ids[0:int(0.8*len_ary)], 0, 0)
        input_label = pic_op.extract_PV(64, pic_labels, pic_ids[0:int(0.8*len_ary)], 0, 1)
        test_set = pic_op.extract_PV(64, pic_ary, pic_ids[int(0.8*len_ary):], 1, 0)
        test_label = pic_op.extract_PV(64, pic_labels, pic_ids[int(0.8*len_ary):], 1, 1)


gen_figs_dataset()

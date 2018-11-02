import numpy as np
from PIL import Image
import pandas as pd
import cv2
import os

class data_set_prepare(object):
    def __init__(self, csv_file_exceptVertices, csv_file_Vertices):
        self.csv_file_exceptVertices = csv_file_exceptVertices
        self.csv_file_Vertices = csv_file_Vertices
        self.data_frame_ExceptPolygon = pd.read_csv(self._get_dir()+"/csv_files/"+self.csv_file_exceptVertices+'.csv')
        self.data_frame_polygon = pd.read_csv(self._get_dir()+"/csv_files/"+self.csv_file_Vertices+'.csv')

    def _get_dir(self):
        return os.getcwd()

    # get the pixels array of one 5000*5000 img
    def read_one_img(self, PV_img):
        PV_img = self._get_dir()+"/imgs/" + PV_img
        im = Image.open(PV_img)
        im_ary = np.array(im)
        return im_ary

    # get ids
    def get_ids(self, PV_img):
        df = self.data_frame_ExceptPolygon
        df = df[df["image_name"] == PV_img]
        return np.array(df['polygon_id'])

    # get the pixels label array of one 5000*5000 img
    def get_label_img(self, im_ary, ids, img_name=None):
        aft_im = np.zeros(np.shape(im_ary), np.uint8)
        df = self.data_frame_polygon
        for id in ids:
            one_row = df.loc[df['polygon_id'] == id]
            one_row = np.array([i for i in np.array(one_row).reshape([-1, 1]) if (not np.isnan(i))])[2:]
            annote_range = np.array([[int(one_row[j + 1][0]), int(one_row[j][0])]
                                     for j in range(len(one_row)) if (j % 2 == 0)],np.int32)
            aft_im = cv2.fillConvexPoly(aft_im, annote_range, [255, 255, 255])
        img = Image.fromarray(aft_im, 'RGB')
        #img.save(img_name)
        return aft_im

    # get the label data set
    # set_stage:  0 training
    #            1 test
    def extract_PV(self, size_side, im_ary, ids, set_stage, set_type, city_name=None, count_number=0):
        data_stage, data_type = 'training/', 'input/'
        df = self.data_frame_ExceptPolygon
        sub_img_ary = []
        for id in ids:
            one_row = df.loc[df['polygon_id'] == id]
            pair = one_row['centroid_latitude_pixels'].values, one_row['centroid_longitude_pixels'].values
            (x1, x2), (y1, y2) = self.bound(pair, size_side)
            sub_img = im_ary[x1:x2, y1:y2, ]
            sub_img_ary.append(sub_img)
            name = str(id) + '.png'
            if set_stage == 1:
                data_stage = 'test/'
            if set_type == 1:
                data_type = 'label/'
            store_dir = self._get_dir() + '/data/' + data_stage + data_type + '/'
            if not os.path.exists(store_dir):
                os.makedirs(store_dir)
            Image.fromarray(sub_img, 'RGB').save(store_dir + name)

        return np.array(sub_img_ary)


    def bound(self, center, size):
        x, y = np.round(center[0], 0), np.round(center[1], 0)
        x1, x2 = (int(x-size/2.), int(x+size/2.))
        y1, y2 = (int(y-size/2.), int(y+size/2.))
        if x1 < 0:
            x1 = 0
            x2 = x1 + 63
        if y1 < 0:
            y1 = 0
            y2 = y1 + 63
        if x2 >= 5000:
            x2 = 4999
            x1 = 4999 - 63
        if y2 >= 5000:
            y2 = 4999
            y1 = 4999 - 63
        return (x1, x2), (y1, y2)


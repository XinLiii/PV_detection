import numpy as np
from PIL import Image
import pandas as pd
import cv2
import os

class data_set_prepare(object):
    def __init__(self, csv_file_exceptVertices, csv_file_Vertices):
        self.csv_file_exceptVertices = csv_file_exceptVertices
        self.csv_file_Vertices = csv_file_Vertices
        self.data_frame_ExceptPolygon = pd.read_csv(self._get_dir()+"/data/csv_files/"+self.csv_file_exceptVertices+'.csv')
        self.data_frame_polygon = pd.read_csv(self._get_dir()+"/data/csv_files/"+self.csv_file_Vertices+'.csv')

    def _get_dir(self):
        return os.getcwd()

    # get the pixels array of one 5000*5000 img
    def read_one_img(self, PV_img):
        PV_img = self._get_dir()+"/data/imgs/" + PV_img + '.tif'
        im = Image.open(PV_img)
        im_ary = np.array(im)
        return im_ary

    # get ids
    def get_ids(self, PV_img):
        df = self.data_frame_ExceptPolygon
        df = df[df["image_name"] == PV_img]
        return np.array(df['polygon_id'])

    # get the pixels label array of one 5000*5000 img
    def get_label_img(self, im_ary, ids):
        aft_im = np.zeros(np.shape(im_ary), np.uint8)
        df = self.data_frame_polygon
        for id in ids:
            one_row = df.loc[df['polygon_id'] == id]
            one_row = np.array([i for i in np.array(one_row).reshape([-1, 1]) if (not np.isnan(i))])[2:]
            annote_range = np.array([[int(one_row[j + 1][0]), int(one_row[j][0])]
                                     for j in range(len(one_row)) if (j % 2 == 0)],np.int32)
            aft_im = cv2.fillConvexPoly(aft_im, annote_range, [255, 255, 255])
        img = Image.fromarray(aft_im, 'RGB')
        img.save('test.png')
        return aft_im

    # get the label data set
    def extract_PV(self,size_side, im_ary, ids, count_number=0):
        df = self.data_frame_ExceptPolygon
        sub_img_ary = []
        for id in ids:
            one_row = df.loc[df['polygon_id'] == id]
            pair = one_row['centroid_latitude_pixels'].values, one_row[ 'centroid_longitude_pixels'].values
            (x1, x2), (y1, y2) = self.bound(pair, size_side)
            sub_img = im_ary[x1:x2 + 1, y1:y2 + 1, ]
            sub_img_ary.append(sub_img)
            name = str(pair) + '.png'
            Image.fromarray(sub_img, 'RGB').save(name)

        return np.array(sub_img_ary)


    def bound(self,center,size):
        x, y = round(center[0]),round(center[1])
        return (int(x-size/2.), int(x+size/2.)), (int(y-size/2.),int(y+size/2.))


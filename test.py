import numpy as np
from PIL import Image
import scipy.misc
from read_fig import data_set_prepare as ds
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

png_file = mpimg.imread('one_chanel_test.png')
print(np.array(png_file).shape)
imgplot = plt.imshow(png_file)
plt.show()
"""
im = Image.open("E:/python3/cnn_PV/data/11ska460755.tif")

im_ary = np.array(im)
sub_im = im_ary[24:64,252:292,:]
print(im_ary.shape)

img = Image.fromarray(sub_im,'RGB')
img.save('test.png')
img.show()

df = pd.read_csv("E:/python3/cnn_PV/data/cite2/polygonVertices_PixelCoordinates.csv")
a=df.loc[df['polygon_id']==1]
b=[i for i in np.array(a).reshape([-1,1]) if(not np.isnan(i))]
b=b[2:]
c=[[b[j][0],b[j+1][0]] for j in range(len(b)) if(j%2==0)]
print(c)

"""
"""
test = ds("polygonDataExceptVertices", "polygonVertices_PixelCoordinates")
raw_img = test.read_one_img("11ska460890")
print(raw_img)
#ids = test.get_ids("11ska460890")
#label_img = test.get_label_img(raw_img, ids)
#train_set_one = test.extract_PV(41,raw_img,ids)
#train_label = test.extract_PV(41,label_img,ids)


input = tf.Variable(tf.random_normal([1,41,41,3]))
filter = tf.Variable(tf.random_normal([1,3,3,1]))
op = tf.nn.max_pool_with_argmax(input,[1,3,3,1],[1,2,2,1],padding='SAME')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(op)
    print(tf.shape(op))
"""



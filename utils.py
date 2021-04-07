import cv2 as cv
import h5py
import os
import math
import random
import numpy as np
import tensorflow as tf


# read YCrCb picture
def read_pic(path, name):
    img_BGR = cv.imread(r"C:\Users\dell\Desktop\SR_data/" + path + "/" + name)
    img_YCrCb = cv.cvtColor(img_BGR, cv.COLOR_BGR2YCrCb)
    return img_YCrCb


# read Y channel
def read_Y(image):
    Y, Cr, Cb = cv.split(image)
    return Y


# read Cr， Cb channel。
def read_Cr_Cb(image):
    Y, Cr, Cb = cv.split(image)
    return Cr, Cb


# Peak signal-to-noise ratio
def psn_picture(tf_img1, tf_img2):
    return tf.image.psnr(tf_img1, tf_img2, max_val=255)


# Peak signal-to-noise ratio
def psnr1(img1, img2):
    mse = np.mean((img1 / 1.0 - img2 / 1.0) ** 2)
    if mse < 1e-10:
        return 100
    psnr = 20 * math.log10(255 / math.sqrt(mse))
    return psnr


# Peak signal-to-noise ratio
def psnr2(img1, img2):
    mse = np.mean((img1 / 255.0 - img2 / 255.0) ** 2)
    if mse < 1e-10:
        return 100
    psnr = 20 * math.log10(1 / math.sqrt(mse))
    return psnr


# Bicubic
def bic_picture(path1, path2, s):
    for filename in os.listdir(r"C:\Users\dell\Desktop\SR_data/" + path1):
        # print(filename) #just for test
        # img is used to store the image data
        print(filename)

        img = cv.imread(r"C:\Users\dell\Desktop\SR_data/" + path1 + "/" + filename)
        # print(img.shape)
        img_shape = img.shape

        height = img_shape[0]
        width = img_shape[1]

        if s > 1:
            imgX2 = cv.resize(img, (width*s, height*s), interpolation=cv.INTER_CUBIC)
        else:
            p = int(1//s)
            imgX2 = cv.resize(img, (width // p, height // p), interpolation=cv.INTER_CUBIC)
        cv.imwrite(r"C:\Users\dell\Desktop\SR_data/" + path2 + "/" + filename, imgX2)


# Cropped image data enhancement
def cut_picture(path1, path2, size):
    name_count = 0
    for filename in os.listdir(r"C:\Users\dell\Desktop\SR_data/" + path1):
        # print(filename) #just for test
        # img is used to store the image data
        print(filename)

        img = cv.imread(r"C:\Users\dell\Desktop\SR_data/" + path1 + "/" + filename)
        # print(img.shape)
        img_shape = img.shape
        height = img_shape[0]
        width = img_shape[1]
        i = 0
        j = 0
        while (i+1)*size <= height or (j+1)*size <= width:
            if (i+1)*size <= height and (j+1)*size <= width:
                img1 = img[i*size:(i+1)*size, j*size:(j+1)*size, :]
                cv.imwrite(r"C:\Users\dell\Desktop\SR_data/" + path2 + "/" + str(name_count) + ".png", img1)
                name_count += 1
                j += 1
            elif (j+1)*size > width:
                img1 = img[i * size:(i + 1) * size, -size:, :]
                cv.imwrite(r"C:\Users\dell\Desktop\SR_data/" + path2 + "/" + str(name_count) + ".png", img1)
                name_count += 1
                i += 1
                j = 0
            elif (i+1)*size > height:
                img1 = img[-size:, j*size:(j+1)*size, :]
                cv.imwrite(r"C:\Users\dell\Desktop\SR_data/" + path2 + "/" + str(name_count) + ".png", img1)
                name_count += 1
                j += 1
            print(name_count)


# creat h5 file
def creat_h5(path, path1, path2, file_name, data_name, data_name1, data_name2):
    x1 = []
    x2 = []
    x4 = []
    for filename in os.listdir(r"C:\Users\dell\Desktop\SR_data/" + path):
        img = read_pic(path, filename)
        Y = read_Y(img)
        x1.append(Y)
    for filename in os.listdir(r"C:\Users\dell\Desktop\SR_data/" + path1):
        img = read_pic(path1, filename)
        Y = read_Y(img)
        x2.append(Y)
    for filename in os.listdir(r"C:\Users\dell\Desktop\SR_data/" + path2):
        img = read_pic(path2, filename)
        Y = read_Y(img)
        x4.append(Y)
    h5f = h5py.File(r'./checkpoint/' + file_name, 'w')
    h5f.create_dataset(data_name, data=x1)
    h5f.create_dataset(data_name1, data=x2)
    h5f.create_dataset(data_name2, data=x4)
    h5f.close()


# load h5 file
def load_h5(name, key):
    h5f = h5py.File(r'./dataset/' + name, 'r')
    print(h5f.keys())
    x1data = h5f[key]
    s = x1data.shape
    x1data = np.reshape(x1data, [s[0], s[1], s[2], 1])
    return x1data


# Picture subpixel
def PS1(x, r, n_out_channel):
    # assert int(x.get_shape()[-1]) == (r ** 2) * n_out_channel
    bsize = tf.shape(x)[0]  # Handling Dimension(None) type for undefined batch dim
    a = tf.to_int32(tf.shape(x)[1])
    b = tf.to_int32(tf.shape(x)[2])
    Xr = tf.concat(x, 2)  # b*h*(r*w)*r
    x = tf.reshape(Xr, (bsize, r * a, r * b, n_out_channel))  # b*(r*h)*(r*w)*c
    return x


# The training data were randomized
def rands(x1_data, x2_data, x1_bic):
    rand_num = random.randint(0, 100)
    random.seed(rand_num)
    random.shuffle(x1_data)
    random.seed(rand_num)
    random.shuffle(x2_data)
    random.seed(rand_num)
    random.shuffle(x1_bic)


# Data enhancement image flip
def flip_picture(path, path1):
    count = 0
    for filename in os.listdir(r"C:\Users\dell\Desktop\SR_data/" + path):
        print(filename)
        img = cv.imread(r"C:\Users\dell\Desktop\SR_data/" + path + "/" + filename)
        # Flip horizontal
        flip_horizontal = cv.flip(img, 1)
        # Flip vertical
        flip_vertical = cv.flip(img, 0)
        # Horizontal plus vertical flip
        flip_hv = cv.flip(img, -1)
        # Save the horizontal flip image
        cv.imwrite(r"C:\Users\dell\Desktop\SR_data/" + path1 + "/" + str(count) + "1" + ".png", flip_horizontal)
        # Save the vertical flip image
        cv.imwrite(r"C:\Users\dell\Desktop\SR_data/" + path1 + "/" + str(count) + "2" + ".png", flip_vertical)
        # Save the horizontal and vertical flip image
        cv.imwrite(r"C:\Users\dell\Desktop\SR_data/" + path1 + "/" + str(count) + "3" + ".png", flip_hv)
        count += 1


# Pixel values may be greater than 255.
# This function changes pixels greater than 255 to 255
def np_to_pic(array, s):
    b, h, w, c = s[0], s[1], s[2], s[3]
    for i in range(b):
        for j in range(h):
            for k in range(w):
                for l in range(c):
                    if array[i][j][k][l] > 255:
                        array[i][j][k][l] = 255
    return array


if __name__ == '__main__':
    # creat_h5('train_cut_l', 'train_cut_h', 'train_cut_b', 'Y_dataset.h5', 'train_cut_l', 'train_cut_h', 'train_cut_b')
    x1d = load_h5('Y_dataset.h5', 'train_cut_l')
    x2d = load_h5('Y_dataset.h5', 'train_cut_h')
    x1_b = load_h5('Y_dataset.h5', 'train_cut_b')
    print(x1d.shape, x2d.shape, x1_b.shape)
    # flip_picture('train_cut_h', 'train_cut_he')
    # flip_picture('train_cut_b', 'train_cut_be')
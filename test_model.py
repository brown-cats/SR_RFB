from utils import (
    read_pic,
    load_h5,
    PS1,
    rands,
    read_Y,
    read_Cr_Cb,
    np_to_pic,
    psnr1,
)
import os
import numpy as np
import cv2 as cv
import tensorflow as tf
from skimage.metrics import structural_similarity

path1 = "Set5/X2/"


def load_pic(test_path):
    test_Y = []
    test_add_Y = []
    test_Cr = []
    test_Cb = []
    test_true = []
    for filename in os.listdir(r'./test_set/' + test_path + "test_lowsr"):

        print(filename)
        img_BGR_l = cv.imread(r'./test_set/' + test_path + "test_lowsr" + "/" + filename)
        img_BGR_l = cv.cvtColor(img_BGR_l, cv.COLOR_BGR2YCrCb)

        input_Y = np.reshape(read_Y(img_BGR_l), [1, img_BGR_l.shape[0], img_BGR_l.shape[1], 1]) / 255
        test_Y.append(input_Y)

        img_BGR_b = cv.imread(r'./test_set/' + test_path + "test_highsr" + "/" + filename)
        img_BGR_b = cv.cvtColor(img_BGR_b, cv.COLOR_BGR2YCrCb)
        Y = np.reshape(read_Y(img_BGR_b), [1, img_BGR_b.shape[0], img_BGR_b.shape[1], 1]) / 255
        test_add_Y.append(Y)
        Cr, Cb = read_Cr_Cb(img_BGR_b)
        test_Cr.append(Cr)
        test_Cb.append(Cb)

        img_BGR_T = cv.imread(r'./test_set/' + test_path + "test_GT" + "/" + filename)
        img_BGR_T = cv.cvtColor(img_BGR_T, cv.COLOR_BGR2YCrCb)
        test_true.append(img_BGR_T)
    return test_Y, test_add_Y, test_Cr, test_Cb, test_true


def mean_psnr(pre, cr, cb, gt, i):
    ps_da = []
    ssim_da = []
    arr = np.array(pre, np.float32)

    arr = arr * 255

    arr = np.around(arr)

    arr = np_to_pic(arr, arr.shape)

    arr = np.array(arr, dtype='uint8')

    arr = arr.reshape([cr.shape[0], cr.shape[1]])
    arr = cv.merge([arr, cr, cb])
    arr = cv.resize(arr, (gt.shape[1], gt.shape[0]), interpolation=cv.INTER_CUBIC)
    ps = psnr1(arr, gt)
    ssim = structural_similarity(arr, gt, multichannel=True)
    ps_da.append(ps)
    ssim_da.append(ssim)
    arr = cv.cvtColor(arr, cv.COLOR_YCrCb2BGR)

    cv.imwrite(r'./test_set/' + path1 + "result/" + str(i) + '.bmp', arr)
    # show_picture(arr, 'test')
    return ps, ssim


def test_model():
    test_Y, test_add_Y, test_Cr, test_Cb, test_true = load_pic(path1)
    path = r'./checkpoint/'
    with tf.Session() as sess:
        # 加载元图和权重
        saver = tf.train.import_meta_graph(path + 'model_conv/my-model-80.meta')
        saver.restore(sess, tf.train.latest_checkpoint(path + "model_conv/"))
        graph = tf.get_default_graph()  # 获取当前默认计算图
        l = len(test_Y)
        ps_data = []
        xsd = []
        print('加载的模型来预测新输入的值了！')
        for i in range(l):

            feed_dict = {"x_input:0": test_Y[i], "x1_bic_add:0": test_add_Y[i]}

            pred_y = tf.get_collection("predict")

            pred = sess.run(pred_y, feed_dict)[0]
            ps, xs = mean_psnr(pred, test_Cr[i], test_Cb[i], test_true[i], i)
            ps_data.append(ps)
            xsd.append(xs)
        print(sum(ps_data)/len(ps_data))
        print(sum(xsd)/len(xsd))

if __name__ == '__main__':
    test_model()
from utils import (
    load_h5,
    rands,
)
from model import define_model
import tensorflow as tf


def train_model():
    learning_rate = 0.0001
    batch_size = 128
    c_dim = 1
    n_batch = 10000 // batch_size

    x_input = tf.placeholder(tf.float32, [None, None, None, c_dim], name='x_input')
    y_label2 = tf.placeholder(tf.float32, [None, None, None, c_dim], name='y_label2')
    x1_bic_add = tf.placeholder(tf.float32, [None, None, None, c_dim], name='x1_bic_add')

    output = define_model(x_input, x1_bic_add)

    mse_loss = tf.reduce_mean(tf.square(y_label2 - output))
    train = tf.train.GradientDescentOptimizer(learning_rate).minimize(mse_loss)

    saver = tf.train.Saver(max_to_keep=4)
    tf.add_to_collection("predict", output)

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        # load train set
        x1d = load_h5('Y_dataset.h5', 'train_cut_l')
        x2d = load_h5('Y_dataset.h5', 'train_cut_h')
        x1_b = load_h5('Y_dataset.h5', 'train_cut_b')
        x1_data = x1d[:10000, :, :, :] / 255
        x2_data = x2d[:10000, :, :, :] / 255
        x1_bic = x1_b[:10000, :, :, :] / 255
        print(x1_data.shape, x2_data.shape, x1_bic.shape)

        counter = 0

        for i in range(2001):
            for idx in range(0, n_batch):
                batch_images = x1_data[idx * batch_size: (idx + 1) * batch_size]
                batch_bic = x1_bic[idx * batch_size: (idx + 1) * batch_size]
                batch_labels = x2_data[idx * batch_size: (idx + 1) * batch_size]

                sess.run(train, feed_dict={x_input: batch_images, y_label2: batch_labels, x1_bic_add: batch_bic})
                counter += 1
                if counter % 50 == 0:
                    print('Epoh', i, 'n_batch', idx,
                          'Train loss:', sess.run(mse_loss, feed_dict={x_input: batch_images, y_label2: batch_labels,
                                                                       x1_bic_add: batch_bic}))
            rands(x1_data, x2_data, x1_bic)

            if i % 10 == 0:
                saver.save(sess, r'./checkpoint/' + "model_conv/my-model", global_step=i)
                print("save the model")

if __name__ == '__main__':
    train_model()
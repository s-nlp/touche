import tensorflow as tf
import sys
import argparse

def restore_graph(cpt_path):
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(cpt_path + 'bert_model.ckpt.meta')
        saver.restore(sess, cpt_path + "bert_model.ckpt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Argument parser')
    parser.add_argument('--bert_dir', default='/notebook/uncased_L-12_H-768_A-12/')
    args = parser.parse_args()
    restore_graph(args.bert_dir)
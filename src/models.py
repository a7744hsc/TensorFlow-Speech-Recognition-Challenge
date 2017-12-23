import tensorflow as tf


def generate_fc_model(fingerprint_input, model_settings, is_training):
    fingerprint_size = model_settings['fingerprint_size']
    label_count = model_settings['label_count']
    weights = tf.Variable(
        tf.truncated_normal([fingerprint_size, label_count], stddev=0.001))
    bias = tf.Variable(tf.zeros([label_count]))
    logits = tf.matmul(fingerprint_input, weights) + bias
    return logits, tf.placeholder(tf.float32, name='dropout_prob')


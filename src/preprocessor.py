import os, re, random, argparse, hashlib, math
from tensorflow.python.util import compat
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
from tensorflow.python.ops import io_ops
import numpy as np

NOISE_FOLDER = '_background_noise_'
DATA_DIR = '/tmp/speech_dataset/'

test_list = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go", "silence", "unknown"];
is_training = True
batch_size= 100

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description="my audio recognizer tainer")
    ap.add_argument('--test', '-t', action="store_true")
    args = ap.parse_args()

    tf.logging.set_verbosity(tf.logging.INFO)
    sess = tf.InteractiveSession()

    model_settings = {
        'desired_samples': 16000,  # 每个数据多少点
        'window_size_samples': 480,  # 每个 spectrogram timeslice多少点， 30 ms
        'window_stride_samples': 160,  # 偏移长度
        'spectrogram_length': 1 + int((16000 - 480) / 160),  # 每个样本能采集多少声谱图
        'dct_coefficient_count': 40,  # How many bins to use for the MFCC fingerprint
        'fingerprint_size': 40 + int((16000 - 480) / 160) * 40,  # 处理后每个样本多少点
        'label_count': 12,
        'sample_rate': 16000,
    }

    #数据处理
    #silence testing unknown validation 百分比都是10
    random.seed(123)
    wanted_words = ['yes','no','up','down','left','right','on','off','stop','go']
    wanted_words_index = {'yes': 2, 'no': 3, 'up': 4, 'down': 5, 'left': 6, 'right': 7, 'on': 8, 'off': 9, 'stop': 10, 'go': 11}
    all_words={}
    data_index = {'validation': [], 'testing': [], 'training': []}
    unknown_index = {'validation': [], 'testing': [], 'training': []}

    search_path = os.path.join(DATA_DIR, '*', '*.wav')
    for wav_path in tf.gfile.Glob(search_path):
        _, word = os.path.split(os.path.dirname(wav_path))
        word = word.lower()
        if word == NOISE_FOLDER:
            continue
        all_words[word] = True
        #用名字做HASH，把数据分到 train，validate和test三组中。
        base_name = os.path.basename(wav_path)
        hash_name = re.sub(r'_nohash_.*$', '', base_name)
        hash_name_hashed = hashlib.sha1(compat.as_bytes(hash_name)).hexdigest()
        percentage_hash = ((int(hash_name_hashed, 16) %(2**20 + 1)) * (100.0 / 2**20))
        if percentage_hash < 10:
            set_index = 'validation'
        elif percentage_hash < 20:
            set_index = 'testing'
        else:
            set_index = 'training'

        if word in wanted_words_index:
            data_index[set_index].append({'label': word, 'file': wav_path})
        else:
            unknown_index[set_index].append({'label': word, 'file': wav_path})

    # We need an arbitrary file to load as the input for the silence samples.
    # It's multiplied by zero later, so the content doesn't matter.
    silence_wav_path = data_index['training'][0]['file']
    for set_index in ['validation', 'testing', 'training']:
        set_size = len(data_index[set_index])
        silence_size = int(math.ceil(set_size * 10 / 100))
        for _ in range(silence_size):
            data_index[set_index].append({
                'label': '_silence_',
                'file': silence_wav_path
            })

        random.shuffle(unknown_index[set_index])
        unknown_size = int(math.ceil(set_size * 10 / 100))
        data_index[set_index].extend(unknown_index[set_index][:unknown_size])
    # Make sure the ordering is random.
    for set_index in ['validation', 'testing', 'training']:
        random.shuffle(data_index[set_index])
    # Prepare the rest of the result data structure.
    words_list = ['_silence_','_unknown_'] +wanted_words
    word_to_index = {}
    for word in all_words:
        if word in wanted_words_index:
            word_to_index[word] = wanted_words_index[word]
        else:
            word_to_index[word] = 1
    word_to_index['_silence_'] = 0
    ################################## 生成索引结束，得到 word_to_index 和 data_index

    # 生成背景噪音数据样本
    background_data = []
    background_dir = os.path.join(DATA_DIR, NOISE_FOLDER)
    with tf.Session(graph=tf.Graph()) as sess1:
        wav_filename_placeholder = tf.placeholder(tf.string, [])
        wav_loader = io_ops.read_file(wav_filename_placeholder)
        wav_decoder = contrib_audio.decode_wav(wav_loader, desired_channels=1)
        search_path = os.path.join(DATA_DIR, NOISE_FOLDER, '*.wav')
        for wav_path in tf.gfile.Glob(search_path):
            wav_data = sess1.run(
                wav_decoder,
                feed_dict={wav_filename_placeholder: wav_path}).audio.flatten()
            background_data.append(wav_data)
            # print("====================")
            # print(wav_data)
        if not background_data:
            raise Exception('No background wav files were found in ' + search_path)

    #Build data generator pipeline
    desired_samples = model_settings['desired_samples']
    wav_filename_placeholder_ = tf.placeholder(tf.string, [],name="wav_filename_placeholder_")
    wav_loader = io_ops.read_file(wav_filename_placeholder_)
    wav_decoder = contrib_audio.decode_wav(
        wav_loader, desired_channels=1, desired_samples=desired_samples)
    # Allow the audio sample's volume to be adjusted.
    foreground_volume_placeholder_ = tf.placeholder(tf.float32, [],name="foreground_volume_placeholder_")
    scaled_foreground = tf.multiply(wav_decoder.audio,
                                    foreground_volume_placeholder_)
    # Shift the sample's start position, and pad any gaps with zeros.
    time_shift_padding_placeholder_ = tf.placeholder(tf.int32, [2, 2],name="time_shift_padding_placeholder_")
    time_shift_offset_placeholder_ = tf.placeholder(tf.int32, [2])
    padded_foreground = tf.pad(
        scaled_foreground,
        time_shift_padding_placeholder_,
        mode='CONSTANT')
    sliced_foreground = tf.slice(padded_foreground,
                                 time_shift_offset_placeholder_,
                                 [desired_samples, -1])
    # Mix in background noise.
    background_data_placeholder_ = tf.placeholder(tf.float32,
                                                       [desired_samples, 1])
    background_volume_placeholder_ = tf.placeholder(tf.float32, [])
    background_mul = tf.multiply(background_data_placeholder_,
                                 background_volume_placeholder_)
    background_add = tf.add(background_mul, sliced_foreground)
    background_clamp = tf.clip_by_value(background_add, -1.0, 1.0)
    # Run the spectrogram and MFCC ops to get a 2D 'fingerprint' of the audio.
    spectrogram = contrib_audio.audio_spectrogram(
        background_clamp,
        window_size=model_settings['window_size_samples'],
        stride=model_settings['window_stride_samples'],
        magnitude_squared=True)
    mfcc_ = contrib_audio.mfcc(
        spectrogram,
        wav_decoder.sample_rate,
        dct_coefficient_count=model_settings['dct_coefficient_count'])




    fingerprint_size = model_settings['fingerprint_size']
    label_count = model_settings['label_count']
    time_shift_samples = int((100 * 16000) / 1000)
    training_steps_list =[15000,3000]
    learning_rates_list = [0.001,0.0001]

    fingerprint_input = tf.placeholder(
        tf.float32, [None, fingerprint_size], name='fingerprint_input')

###########################Define simple model################

    dropout_prob=None
    if is_training:
        dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
    fingerprint_size = model_settings['fingerprint_size']
    label_count = model_settings['label_count']
    weights = tf.Variable(
        tf.truncated_normal([fingerprint_size, label_count], stddev=0.001))
    bias = tf.Variable(tf.zeros([label_count]))
    logits = tf.matmul(fingerprint_input, weights) + bias

    ground_truth_input = tf.placeholder(tf.int64, [None], name='groundtruth_input')
    control_dependencies = []
    # stop_when_arg_is_nan = True
    # if stop_when_arg_is_nan:
    #     checks = tf.add_check_numerics_ops()
    #     control_dependencies = [checks]
    with tf.name_scope('cross_entropy'):
        cross_entropy_mean = tf.losses.sparse_softmax_cross_entropy(
            labels=ground_truth_input, logits=logits)
    tf.summary.scalar('cross_entropy', cross_entropy_mean)
    with tf.name_scope('train'), tf.control_dependencies(control_dependencies):
        learning_rate_input = tf.placeholder(
            tf.float32, [], name='learning_rate_input')
        train_step = tf.train.GradientDescentOptimizer(
            learning_rate_input).minimize(cross_entropy_mean)
    predicted_indices = tf.argmax(logits, 1)
    correct_prediction = tf.equal(predicted_indices, ground_truth_input)
    evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', evaluation_step)
    merged_summaries = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('/tmp/train', sess.graph)
    validation_writer = tf.summary.FileWriter('/tmp/validation')

    tf.global_variables_initializer().run()

    training_steps_max = np.sum(training_steps_list)
    for training_step in range(1, training_steps_max + 1):
        # Figure out what the current learning rate is.
        training_steps_sum = 0
        for i in range(len(training_steps_list)):
            training_steps_sum += training_steps_list[i]
            if training_step <= training_steps_sum:
                learning_rate_value = learning_rates_list[i]
                break


        ####################### Pull the audio samples we'll use for training.

        background_volume_range = 0.1
        background_frequency = 0.8
        # Pick one of the partitions to choose samples from.
        candidates = data_index["training"]
        sample_count = max(0, min(batch_size, len(candidates)))
        # Data and labels will be populated and returned.
        data = np.zeros((sample_count, model_settings['fingerprint_size']))
        labels = np.zeros(sample_count)
        desired_samples = model_settings['desired_samples']
        use_background = True #self.background_data and (mode == 'training')
        pick_deterministically = False #(mode != 'training')
        # Use the processing graph we created earlier to repeatedly to generate the
        # final output sample data we'll use in training.
        for i in range(sample_count):
            # Pick which audio sample to use.
            # if how_many == -1 or pick_deterministically:
            #     sample_index = i
            # else:
            sample_index = np.random.randint(len(candidates))
            sample = candidates[sample_index]
            # If we're time shifting, set up the offset for this sample.
            time_shift_amount = np.random.randint(-time_shift_samples, time_shift_samples)

            if time_shift_amount > 0:
                time_shift_padding = [[time_shift_amount, 0], [0, 0]]
                time_shift_offset = [0, 0]
            else:
                time_shift_padding = [[0, -time_shift_amount], [0, 0]]
                time_shift_offset = [-time_shift_amount, 0]
            input_dict = {
                wav_filename_placeholder_: sample['file'],
                time_shift_padding_placeholder_: time_shift_padding,
                time_shift_offset_placeholder_: time_shift_offset,
            }
            # Choose a section of background noise to mix in.
            if use_background:
                background_index = np.random.randint(len(background_data))
                background_samples = background_data[background_index]
                background_offset = np.random.randint(
                    0, len(background_samples) - model_settings['desired_samples'])
                background_clipped = background_samples[background_offset:(
                        background_offset + desired_samples)]
                background_reshaped = background_clipped.reshape([desired_samples, 1])
                if np.random.uniform(0, 1) < background_frequency:
                    background_volume = np.random.uniform(0, background_volume_range)
                else:
                    background_volume = 0
            else:
                background_reshaped = np.zeros([desired_samples, 1])
                background_volume = 0
            input_dict[background_data_placeholder_] = background_reshaped
            input_dict[background_volume_placeholder_] = background_volume
            # If we want silence, mute out the main sample but leave the background.
            if sample['label'] == '_silence_':
                input_dict[foreground_volume_placeholder_] = 0
            else:
                input_dict[foreground_volume_placeholder_] = 1
            # Run the graph to produce the output audio.
            data[i, :] = sess.run(mfcc_, feed_dict=input_dict).flatten()
            label_index = word_to_index[sample['label']]
            labels[i] = label_index
        train_fingerprints, train_ground_truth = data, labels

        # Run the graph with this batch of training data.
        train_summary,train_accuracy, cross_entropy_value, _ = sess.run(
            [
                merged_summaries,evaluation_step, cross_entropy_mean, train_step
            ],
            feed_dict={
                fingerprint_input: train_fingerprints,
                ground_truth_input: train_ground_truth,
                learning_rate_input: learning_rate_value,
                dropout_prob: 0.5
            })
        # train_writer.add_summary(train_summary, training_step)
        tf.logging.info('Step #%d: rate %f, accuracy %.1f%%, cross entropy %f' %
                        (training_step, learning_rate_value, train_accuracy * 100,
                         cross_entropy_value))
        # is_last_step = (training_step == training_steps_max)
        # if (training_step % 100) == 0 or is_last_step:
        #     set_size = audio_processor.set_size('validation')
        #     total_accuracy = 0
        #     total_conf_matrix = None
        #     for i in xrange(0, set_size, FLAGS.batch_size):
        #         validation_fingerprints, validation_ground_truth = (
        #             audio_processor.get_data(FLAGS.batch_size, i, model_settings, 0.0,
        #                                      0.0, 0, 'validation', sess))
        #         # Run a validation step and capture training summaries for TensorBoard
        #         # with the `merged` op.
        #         validation_summary, validation_accuracy, conf_matrix = sess.run(
        #             [merged_summaries, evaluation_step, confusion_matrix],
        #             feed_dict={
        #                 fingerprint_input: validation_fingerprints,
        #                 ground_truth_input: validation_ground_truth,
        #                 dropout_prob: 1.0
        #             })
        #         validation_writer.add_summary(validation_summary, training_step)
        #         batch_size = min(FLAGS.batch_size, set_size - i)
        #         total_accuracy += (validation_accuracy * batch_size) / set_size
        #         if total_conf_matrix is None:
        #             total_conf_matrix = conf_matrix
        #         else:
        #             total_conf_matrix += conf_matrix
        #     tf.logging.info('Confusion Matrix:\n %s' % (total_conf_matrix))
        #     tf.logging.info('Step %d: Validation accuracy = %.1f%% (N=%d)' %
        #                     (training_step, total_accuracy * 100, set_size))
        #
        # # Save the model checkpoint periodically.
        # if (training_step % FLAGS.save_step_interval == 0 or
        #         training_step == training_steps_max):
        #     checkpoint_path = os.path.join(FLAGS.train_dir,
        #                                    FLAGS.model_architecture + '.ckpt')
        #     tf.logging.info('Saving to "%s-%d"', checkpoint_path, training_step)
        #     saver.save(sess, checkpoint_path, global_step=training_step)

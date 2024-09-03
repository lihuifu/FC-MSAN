import tensorflow as tf
import keras
from keras.models import Model
from keras.layers import Input, Conv1D, Dense, Dropout, MaxPool1D, Activation
from keras.layers import Flatten, Reshape, TimeDistributed, BatchNormalization


'''
A Feature Extractor Network
'''

def smal_feature_3(input_signal):
    activation = tf.nn.relu
    padding = 'same'
    cnn0 = Conv1D(kernel_size=100,
                    filters=64,
                    strides=30,
                    kernel_regularizer=keras.regularizers.l2(0.001))
    s = cnn0(input_signal)
    s = BatchNormalization()(s)
    s = Activation(activation=activation)(s)
    cnn1 = MaxPool1D(pool_size=16, strides=16)
    s = cnn1(s)
    cnn2 = Dropout(0.5)
    s = cnn2(s)
    cnn3 = Conv1D(kernel_size=4, filters=64, strides=1, padding=padding)
    s = cnn3(s)
    s = BatchNormalization()(s)
    s = Activation(activation=activation)(s)
    cnn4 = Conv1D(kernel_size=4, filters=64, strides=1, padding=padding)
    s = cnn4(s)
    s = BatchNormalization()(s)
    s = Activation(activation=activation)(s)
    cnn5 = Conv1D(kernel_size=4, filters=64, strides=1, padding=padding)
    s = cnn5(s)
    s = BatchNormalization()(s)
    s = Activation(activation=activation)(s)
    cnn6 = MaxPool1D(pool_size=4, strides=4)
    s = cnn6(s)
    cnn7 = Reshape((int(s.shape[1]) * int(s.shape[2]), ))  # Flatten
    s = cnn7(s)
    return s

def large_feature_3(input_signal):
    activation = tf.nn.relu
    padding = 'same'
    cnn8 = Conv1D(kernel_size=100,
                  filters=64,
                  strides=40,
                  kernel_regularizer=keras.regularizers.l2(0.001))
    l = cnn8(input_signal)
    l = BatchNormalization()(l)
    l = Activation(activation=activation)(l)
    cnn9 = MaxPool1D(pool_size=8, strides=8)
    l = cnn9(l)
    cnn10 = Dropout(0.5)
    l = cnn10(l)
    cnn11 = Conv1D(kernel_size=6, filters=64, strides=1, padding=padding)
    l = cnn11(l)
    l = BatchNormalization()(l)
    l = Activation(activation=activation)(l)
    cnn12 = Conv1D(kernel_size=6, filters=64, strides=1, padding=padding)
    l = cnn12(l)
    l = BatchNormalization()(l)
    l = Activation(activation=activation)(l)
    cnn13 = Conv1D(kernel_size=6, filters=64, strides=1, padding=padding)
    l = cnn13(l)
    l = BatchNormalization()(l)
    l = Activation(activation=activation)(l)
    cnn14 = MaxPool1D(pool_size=4, strides=4)
    l = cnn14(l)
    cnn15 = Reshape((int(l.shape[1]) * int(l.shape[2]), ))
    l = cnn15(l)
    return l


def smal_feature_2(input_signal):
    activation = tf.nn.relu
    padding = 'same'
    cnn0 = Conv1D(kernel_size=50,
                    filters=32,
                    strides=18,
                    kernel_regularizer=keras.regularizers.l2(0.001))
    s = cnn0(input_signal)
    s = BatchNormalization()(s)
    s = Activation(activation=activation)(s)
    cnn1 = MaxPool1D(pool_size=16, strides=16)
    s = cnn1(s)
    cnn2 = Dropout(0.5)
    s = cnn2(s)
    cnn3 = Conv1D(kernel_size=4, filters=64, strides=1, padding=padding)
    s = cnn3(s)
    s = BatchNormalization()(s)
    s = Activation(activation=activation)(s)
    cnn4 = Conv1D(kernel_size=4, filters=64, strides=1, padding=padding)
    s = cnn4(s)
    s = BatchNormalization()(s)
    s = Activation(activation=activation)(s)
    cnn5 = Conv1D(kernel_size=4, filters=64, strides=1, padding=padding)
    s = cnn5(s)
    s = BatchNormalization()(s)
    s = Activation(activation=activation)(s)
    cnn6 = MaxPool1D(pool_size=4, strides=4)
    s = cnn6(s)
    cnn7 = Reshape((int(s.shape[1]) * int(s.shape[2]), ))  # Flatten
    s = cnn7(s)
    return s

def large_feature_2(input_signal):
    activation = tf.nn.relu
    padding = 'same'
    cnn8 = Conv1D(kernel_size=200,
                  filters=64,
                  strides=30,
                  kernel_regularizer=keras.regularizers.l2(0.001))
    l = cnn8(input_signal)
    l = BatchNormalization()(l)
    l = Activation(activation=activation)(l)
    cnn9 = MaxPool1D(pool_size=8, strides=8)
    l = cnn9(l)
    cnn10 = Dropout(0.5)
    l = cnn10(l)
    cnn11 = Conv1D(kernel_size=6, filters=64, strides=1, padding=padding)
    l = cnn11(l)
    l = BatchNormalization()(l)
    l = Activation(activation=activation)(l)
    cnn12 = Conv1D(kernel_size=6, filters=64, strides=1, padding=padding)
    l = cnn12(l)
    l = BatchNormalization()(l)
    l = Activation(activation=activation)(l)
    cnn13 = Conv1D(kernel_size=6, filters=64, strides=1, padding=padding)
    l = cnn13(l)
    l = BatchNormalization()(l)
    l = Activation(activation=activation)(l)
    cnn14 = MaxPool1D(pool_size=4, strides=4)
    l = cnn14(l)
    cnn15 = Reshape((int(l.shape[1]) * int(l.shape[2]), ))
    l = cnn15(l)
    return l


def smal_feature(input_signal):
    activation = tf.nn.relu
    padding = 'same'
    cnn0 = Conv1D(kernel_size=50,
                    filters=32,
                    strides=6,
                    kernel_regularizer=keras.regularizers.l2(0.001))
    s = cnn0(input_signal)
    s = BatchNormalization()(s)
    s = Activation(activation=activation)(s)
    cnn1 = MaxPool1D(pool_size=16, strides=16)
    s = cnn1(s)
    cnn2 = Dropout(0.5)
    s = cnn2(s)
    cnn3 = Conv1D(kernel_size=8, filters=64, strides=1, padding=padding)
    s = cnn3(s)
    s = BatchNormalization()(s)
    s = Activation(activation=activation)(s)
    cnn4 = Conv1D(kernel_size=8, filters=64, strides=1, padding=padding)
    s = cnn4(s)
    s = BatchNormalization()(s)
    s = Activation(activation=activation)(s)
    cnn5 = Conv1D(kernel_size=8, filters=64, strides=1, padding=padding)
    s = cnn5(s)
    s = BatchNormalization()(s)
    s = Activation(activation=activation)(s)
    cnn6 = MaxPool1D(pool_size=8, strides=8)
    s = cnn6(s)
    cnn7 = Reshape((int(s.shape[1]) * int(s.shape[2]), ))  # Flatten
    s = cnn7(s)
    return s

def large_feature(input_signal):
    activation = tf.nn.relu
    padding = 'same'
    cnn8 = Conv1D(kernel_size=100,
                  filters=64,
                  strides=10,
                  kernel_regularizer=keras.regularizers.l2(0.001))
    l = cnn8(input_signal)
    l = BatchNormalization()(l)
    l = Activation(activation=activation)(l)
    cnn9 = MaxPool1D(pool_size=8, strides=8)
    l = cnn9(l)
    cnn10 = Dropout(0.5)
    l = cnn10(l)
    cnn11 = Conv1D(kernel_size=6, filters=64, strides=1, padding=padding)
    l = cnn11(l)
    l = BatchNormalization()(l)
    l = Activation(activation=activation)(l)
    cnn12 = Conv1D(kernel_size=6, filters=64, strides=1, padding=padding)
    l = cnn12(l)
    l = BatchNormalization()(l)
    l = Activation(activation=activation)(l)
    cnn13 = Conv1D(kernel_size=6, filters=64, strides=1, padding=padding)
    l = cnn13(l)
    l = BatchNormalization()(l)
    l = Activation(activation=activation)(l)
    cnn14 = MaxPool1D(pool_size=4, strides=4)
    l = cnn14(l)
    cnn15 = Reshape((int(l.shape[1]) * int(l.shape[2]), ))
    l = cnn15(l)
    return l

def find_topkey(input_data):
    # Print input data shape
    print("input_data shape:", input_data.shape)

    # Parameters
    batch_size = 16
    channels = 10
    time_second = 30
    freq = 100
    window_size = 1
    step_size = 1
    num_windows = step_size + 1

    # Convert input data to tensor
    data_tensor = tf.convert_to_tensor(input_data, dtype=tf.float32)

    # Define Pearson correlation function
    def pearson_correlation(x, y):
        x_mean = tf.reduce_mean(x)
        y_mean = tf.reduce_mean(y)
        x_centered = x - x_mean
        y_centered = y - y_mean
        numerator = tf.reduce_sum(x_centered * y_centered)
        denominator = tf.sqrt(tf.reduce_sum(x_centered**2) * tf.reduce_sum(y_centered**2))
        return numerator / denominator

    # Initialize weighted data tensor
    weighted_data = tf.TensorArray(tf.float32, size=input_data.shape[0], dynamic_size=False, infer_shape=False)

    # Process each batch
    for b in range(batch_size):
        batch_data = data_tensor[b]
        weighted_data_batch = tf.TensorArray(tf.float32, size=channels, dynamic_size=False, infer_shape=False)

        for i in range(num_windows, time_second):
            start_idx = (i - num_windows) * freq
            end_idx = i * freq
            window_data = batch_data[:, start_idx:end_idx]
            window_data_i = batch_data[:, end_idx:(i+1)*freq]

            # Calculate Pearson correlation matrix
            pearson_matrix = tf.TensorArray(tf.float32, size=channels, dynamic_size=False, infer_shape=False)
            for ch1 in range(channels):
                pearson_matrix_ch1 = tf.TensorArray(tf.float32, size=channels, dynamic_size=False, infer_shape=False)
                for ch2 in range(channels):
                    if ch1 != ch2:
                        pearson_corr = pearson_correlation(window_data[ch1], window_data[ch2])
                        pearson_matrix_ch1 = pearson_matrix_ch1.write(ch2, pearson_corr)
                pearson_matrix = pearson_matrix.write(ch1, pearson_matrix_ch1.stack())

            pearson_matrix = pearson_matrix.stack()

            # Update weighted data
            weighted_data_i = tf.TensorArray(tf.float32, size=freq, dynamic_size=False, infer_shape=False)
            for ch1 in range(channels):
                for ch2 in range(channels):
                    if ch1 != ch2:
                        weights = pearson_matrix[ch1, ch2]
                        weighted_data_ch1 = tf.tensordot(weights, window_data_i[ch1, :], axes=0)
                        weighted_data_i = weighted_data_i.write(ch1, weighted_data_ch1)

            weighted_data_batch = weighted_data_batch.write(i, weighted_data_i.stack())

        weighted_data = weighted_data.write(b, weighted_data_batch.stack())

    # Convert weighted data tensor array to tensor
    weighted_data = weighted_data.stack()
    return weighted_data

def find_topkey_5(input_data):
    # Print input data shape
    print("input_data shape:", input_data.shape)
    
    # Parameters
    batch_size = 16
    channels = 10
    time_second = 30
    freq = 100
    window_size = 5
    step_size = 5
    num_windows = step_size + 1
    
    # Convert input data to tensor
    data_tensor = tf.convert_to_tensor(input_data, dtype=tf.float32)
    
    # Define Pearson correlation function
    def pearson_correlation(x, y):
        x_mean = tf.reduce_mean(x)
        y_mean = tf.reduce_mean(y)
        x_centered = x - x_mean
        y_centered = y - y_mean
        numerator = tf.reduce_sum(x_centered * y_centered)
        denominator = tf.sqrt(tf.reduce_sum(x_centered**2) * tf.reduce_sum(y_centered**2))
        return numerator / denominator
    
    # Initialize weighted data tensor
    weighted_data = tf.TensorArray(tf.float32, size=input_data.shape[0], dynamic_size=False, infer_shape=False)

    # Process each batch
    for b in range(batch_size):
        batch_data = data_tensor[b]
        weighted_data_batch = tf.TensorArray(tf.float32, size=channels, dynamic_size=False, infer_shape=False)
        
        for i in range(num_windows, time_second):
            start_idx = (i - num_windows) * freq
            end_idx = i * freq
            window_data = batch_data[:, start_idx:end_idx]
            window_data_i = batch_data[:, end_idx:(i+1)*freq]

            # Calculate Pearson correlation matrix
            pearson_matrix = tf.TensorArray(tf.float32, size=channels, dynamic_size=False, infer_shape=False)
            for ch1 in range(channels):
                pearson_matrix_ch1 = tf.TensorArray(tf.float32, size=channels, dynamic_size=False, infer_shape=False)
                for ch2 in range(channels):
                    if ch1 != ch2:
                        pearson_corr = pearson_correlation(window_data[ch1], window_data[ch2])
                        pearson_matrix_ch1 = pearson_matrix_ch1.write(ch2, pearson_corr)
                pearson_matrix = pearson_matrix.write(ch1, pearson_matrix_ch1.stack())

            pearson_matrix = pearson_matrix.stack()

            # Update weighted data
            weighted_data_i = tf.TensorArray(tf.float32, size=freq, dynamic_size=False, infer_shape=False)
            for ch1 in range(channels):
                for ch2 in range(channels):
                    if ch1 != ch2:
                        weights = pearson_matrix[ch1, ch2]
                        weighted_data_ch1 = tf.tensordot(weights, window_data_i[ch1, :], axes=0)
                        weighted_data_i = weighted_data_i.write(ch1, weighted_data_ch1)
            
            weighted_data_batch = weighted_data_batch.write(i, weighted_data_i.stack())

        weighted_data = weighted_data.write(b, weighted_data_batch.stack())
    
    # Convert weighted data tensor array to tensor
    weighted_data = weighted_data.stack()
    return weighted_data

def MSAN(s, l, s2, l2, s3, l3):
    s = Dense(64)(s)
    l = Dense(64)(l)
    s2 = Dense(64)(s2)
    l2 = Dense(64)(l2)
    s3 = Dense(64)(s3)
    l3 = Dense(64)(l3)
    attention_scores = tf.matmul(s, l, transpose_b=True)
    attention_weights = tf.nn.softmax(attention_scores, axis=-1)
    attended_output_s_l = tf.matmul(attention_weights, l)

    attention_scores = tf.matmul(s, l2, transpose_b=True)
    attention_weights = tf.nn.softmax(attention_scores, axis=-1)
    attended_output_s_l2 = tf.matmul(attention_weights, l2)

    attention_scores = tf.matmul(s, s2, transpose_b=True)
    attention_weights = tf.nn.softmax(attention_scores, axis=-1)
    attended_output_s_s2 = tf.matmul(attention_weights, s2)

    attention_scores = tf.matmul(s, l3, transpose_b=True)
    attention_weights = tf.nn.softmax(attention_scores, axis=-1)
    attended_output_s_l3 = tf.matmul(attention_weights, l3)

    attention_scores = tf.matmul(s, s3, transpose_b=True)
    attention_weights = tf.nn.softmax(attention_scores, axis=-1)
    attended_output_s_s3 = tf.matmul(attention_weights, s3)

    attended_output_s_l = attended_output_s_l + attended_output_s_l2 + attended_output_s_s2 + attended_output_s_l3 + attended_output_s_s3
    return attended_output

def build_FC_MSAN(opt, channels=10, time_second=30, freq=100):
    activation = tf.nn.relu
    padding = 'same'

    ######### Input ########
    input_signal = Input(shape=(time_second * freq, 1), name='input_signal')

    ######### CNNs with small filter size at the first layer #########
    s = smal_feature(input_signal)
    s2 = smal_feature_2(input_signal)
    s3 = smal_feature_3(input_signal)

    ######### CNNs with large filter size at the first layer #########
    l = large_feature(input_signal)
    l2 = large_feature_2(input_signal)
    l3 = large_feature_3(input_signal)

    #feature = keras.layers.concatenate([s, l, s2, l2, s3, l3])
    feature = MSAN(s, l, s2, l2, s3, l3)

    fea_part = Model(input_signal, feature)

    ##################################################

    input = Input(shape=(channels, time_second * freq), name='input_signal')
    reshape = Reshape((channels, time_second * freq, 1))  # Flatten
    input_re = reshape(input)
    fea_all = TimeDistributed(fea_part)(input_re)

    topk_1 = find_topkey(input)
    topk_5 = find_topkey(input)
    
    topk_1 = topk_1 + topk_5
    fea_all_2 = TimeDistributed(fea_part)(topk_1)

    fea_all =  keras.layers.concatenate([fea_all, fea_all_2])

    merged = Flatten()(fea_all)
    merged = Dropout(0.5)(merged)
    merged = Dense(64)(merged)
    merged = Dense(5)(merged)

    fea_softmax = Activation(activation='softmax')(merged)

    # FC_MSAN with softmax
    fea_model = Model(input, fea_softmax)
    fea_model.compile(optimizer=opt,
                      loss='categorical_crossentropy',
                      metrics=['acc'])

    # FC_MSAN without softmax
    pre_model = Model(input, fea_all)
    pre_model.compile(optimizer=opt,
                      loss='categorical_crossentropy',
                      metrics=['acc'])

    return fea_model, pre_model

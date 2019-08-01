# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from music21 import converter, instrument, note, chord, stream
from keras.layers import Input, Dense, TimeDistributed, Dropout, LSTM, RepeatVector, concatenate, Reshape, Flatten
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam, SGD
from keras.utils import np_utils, plot_model, to_categorical

#To work with sessions in the case of running on GPU
import tensorflow as tf

#For F1 score
from keras import backend as K

#To easily split into training and test set. The training set will then be split into
#training and validation within the "fit" function of the model.
from sklearn.model_selection import train_test_split

from midi_randomizer import *


def prepare_sequences(notes, n_vocab):
    """ Prepare the sequences used by the Neural Network """
    sequence_length = 100

    # Get all pitch names
    pitchnames = sorted(set(notes))

    # Create a dictionary to map pitches to integers
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    network_input = []
    network_output = []

    # create input sequences and the corresponding outputs
    for i in range(len(notes) - sequence_length):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)

    # Reshape the input into a format compatible with LSTM layers
    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))

    # Normalize input between -1 and 1
    network_input = (network_input - float(n_vocab)/2) / (float(n_vocab)/2)
    network_output = np_utils.to_categorical(network_output)

    return (network_input, network_output)

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.
        Only computes a batch-wise average of recall.
        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.
        Only computes a batch-wise average of precision.
        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

class RAGNetwork():
    def __init__(self, num_notes = 1000, learning_rate = 0.0002, test_scale = 0.1):
        self.inner_dim = num_notes
        self.test_scale = test_scale
        self.number_note_cat = len(get_note_dic(False))
        self.number_of_levels = 10
        test_size = int(num_notes * test_scale)
        train_size = num_notes - test_size
        print(test_size)
        print(train_size)
        self.batch_size = self.computeHCF(test_size, train_size)

        batch_shape = (self.batch_size, self.inner_dim, 1)

        #The 2 sets of possible inputs
        self.music_input = Input(batch_shape=batch_shape, name='music_input')
        #self.music_input_layer = BatchNormalization()(self.music_input)

        self.level_input = Input(batch_shape=(self.batch_size, self.number_of_levels), name='level_input')
        #self.level_input_layers = BatchNormalization()(self.level_input)
        #self.level_input_layers = Dense(200)(self.level_input)
        #self.level_input_layers = Dense(500)(self.level_input_layers)
        self.level_input_layers = Dense(self.inner_dim)(self.level_input)
        self.level_input_layers = Reshape((self.inner_dim, 1))(self.level_input_layers)

        self.before_both_inputs = concatenate([self.level_input_layers, self.music_input])

        #The encoding layer
        #Input shape to LSTM is (time series steps, number of inputs per sequence)
        #Actual input and outputwhen model being fit (batch size, time series steps, number of inputs per sequence)
        self.encoder = LSTM(self.inner_dim, activation='relu', batch_input_shape = batch_shape, stateful=True)(self.before_both_inputs)

        #music_parsed = self.encoder(self.music_input_layer)
        #music_parsed = self.encoder(self.music_input)
        #level_parsed = self.encoder(self.level_input_layers)
        #self.both_inputs = concatenate([music_parsed, level_parsed])

        #The music recreation, note making section
        self.music_output = RepeatVector(self.inner_dim)(self.encoder)
        self.music_output = LSTM(self.inner_dim, activation='relu', return_sequences = True, batch_input_shape = batch_shape, stateful=True, name='music_output')(self.music_output)
        #self.music_output = TimeDistributed(Dense(1), name='music_output')(self.music_output)

        #The level prediction section
        self.level_output = RepeatVector(self.inner_dim)(self.encoder)
        self.level_output = LSTM(self.inner_dim, activation='relu', batch_input_shape = batch_shape, stateful=True)(self.level_output)
        #self.level_output = Dense(self.inner_dim)(self.level_output)
        #self.level_output = Flatten()(self.level_output)
        #self.level_output = Dense(500)(self.level_output)
        #self.level_output = Dense(200)(self.level_output)
        self.level_output = Dense(self.number_of_levels, activation='softmax', name="level_output")(self.level_output)

        self.part_1_model = Model(inputs = [self.music_input, self.level_input], outputs = [self.music_output, self.level_output])
        self.part_2_model = Model(inputs = [self.music_input, self.level_input], outputs = [self.music_output, self.level_output])
        self.final_model = Model(inputs = [self.music_input, self.level_input], outputs = [self.music_output])

        self.part_1_model.compile(optimizer = Adam(learning_rate, 0.5),
                                  loss= {'music_output': 'sparse_categorical_crossentropy', 'level_output': 'categorical_crossentropy'},
                                  metrics=["accuracy"])

        self.part_2_model.compile(optimizer = Adam(learning_rate, 0.5),
                                  loss= {'music_output': 'sparse_categorical_crossentropy', 'level_output': 'categorical_crossentropy'},
                                  metrics=["accuracy"])

        self.final_model.compile(optimizer = Adam(learning_rate, 0.5),
                                 loss= {'music_output': 'sparse_categorical_crossentropy'},
                                 metrics=["accuracy"])

    def plot_models(self):
        plot_model(self.part_1_model, show_shapes=True, to_file='part_1_model.png')
        plot_model(self.part_2_model, show_shapes=True, to_file='part_2_model.png')
        plot_model(self.final_model, show_shapes=True, to_file='final_model.png')

    def get_noises(self, sequences):
        noises = []
        for _ in range(len(sequences)):
            noises.append(np.random.normal(0, 1, (self.inner_dim,)))

        noises = np.array(noises)
        sequences = np.array(sequences)
        print(noises.shape)
        print(sequences.shape)

        noises = np.reshape(noises, (len(sequences), self.inner_dim, 1))
        sequences = np.reshape(sequences, (len(sequences), self.inner_dim, 1))

        return noises, sequences

    def train(self, epochs, levels, sequences):

        #Get the note sequences and random noise for music section of models
        X1, Y1 = self.get_noises(sequences)
        print(X1.shape)
        print(Y1.shape)
        x1_train, x1_test, y1_train, y1_test = train_test_split(X1, Y1, test_size=self.test_scale, shuffle= True)

        #Convert the levels to categorical values
        X2 = to_categorical(levels, num_classes=self.number_of_levels)
        Y2 = to_categorical(levels, num_classes=self.number_of_levels)
        x2_train, x2_test, y2_train, y2_test = train_test_split(X2, Y2, test_size=self.test_scale, shuffle= True)

        try:
            #Need to call initialize global variables in case running on GPU.
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                print(x1_train.shape)
                print(x2_train.shape)
                print(y1_train.shape)
                print(y2_train.shape)
                for e in range(epochs):
                    print("Epoch num: {}".format(e))
                    self.part_1_model.fit({'music_input': x1_train, 'level_input': x2_train},
                                         {'music_output': y1_train, 'level_output': y2_train},
                                         epochs=1, validation_split=0.2, batch_size = self.batch_size)

                    self.part_1_model.reset_states()

                #Print all metrics from run
                #print(hist.history)

                results = self.part_1_model.evaluate({'music_input': np.array(x1_test), 'level_input': np.array(x2_test)},
                                         {'music_output': np.array(y1_test), 'level_output': np.array(y2_test)},
                                         batch_size=batch_size)

                print("Test (loss, accuracy, f1 score): ", results)
        except Exception as e:
            print("Session failed to be initialized.")
            raise e

    def generate(self, input_notes):
        # Get pitch names and store in a dictionary
        notes = input_notes
        pitchnames = sorted(set(item for item in notes))
        int_to_note = dict((number, note) for number, note in enumerate(pitchnames))

        # Use random noise to generate sequences
        noise = np.random.normal(0, 1, (1, self.latent_dim))
        predictions = self.generator.predict(noise)

        pred_notes = [x*242+242 for x in predictions[0]]
        pred_notes = [int_to_note[int(x)] for x in pred_notes]

        create_midi(pred_notes, 'gan_final')

    def plot_loss(self):
        plt.plot(self.disc_loss, c='red')
        plt.plot(self.gen_loss, c='blue')
        plt.title("GAN Loss per Epoch")
        plt.legend(['Discriminator', 'Generator'])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig('GAN_Loss_per_Epoch_final.png', transparent=True)
        plt.close()

    def computeHCF(self, x, y):
        if x > y:
            smaller = y
        else:
            smaller = x
        for i in range(1, smaller+1):
            if((x % i == 0) and (y % i == 0)):
                hcf = i

        return hcf

if __name__ == '__main__':
    #Get the sequences from Midi files and their corresponding levels.
    levels, sequences = get_data_from_midi("./new_files")
    min_notes = min([len(x) for x in sequences])
    sequences = [x[:min_notes] for x in sequences]
    rag = RAGNetwork(num_notes = min_notes)
    rag.plot_models()
    rag.train(50, levels, sequences)

  #gan.train(epochs=5000, batch_size=32, sample_interval=1)

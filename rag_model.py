# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import glob
from music21 import converter, instrument, note, chord, stream
from keras.layers import Input, Dense, TimeDistributed, Dropout, LSTM, RepeatVector, concatenate, Reshape
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.utils import np_utils, plot_model

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

class RAGNetwork():
    def __init__(self, num_notes = 1000):
        self.inner_dim = num_notes

        #The 2 sets of possible inputs
        self.music_input = Input(shape=(self.inner_dim, 1))

        self.level_input = Input(shape=(1,))
        self.level_input_layers = Dense(200)(self.level_input)
        self.level_input_layers = Dense(500)(self.level_input_layers)
        self.level_input_layers = Dense(self.inner_dim)(self.level_input_layers)
        self.level_input_layers = Reshape((self.inner_dim, 1))(self.level_input_layers)

        #The encoding layer
        #Input shape to LSTM is (time series steps, number of inputs per sequence)
        #Actual input and outputwhen model being fit (batch size, time series steps, number of inputs per sequence)
        self.encoder = LSTM(self.inner_dim, activation='relu')

        music_parsed = self.encoder(self.music_input)
        level_parsed = self.encoder(self.level_input_layers)
        self.both_inputs = concatenate([music_parsed, level_parsed])

        #The music recreation, note making section
        self.music_output = RepeatVector(self.inner_dim)(self.both_inputs)
        self.music_output = LSTM(self.inner_dim, activation='relu', return_sequences=True)(self.music_output)
        self.music_output = TimeDistributed(Dense(1))(self.music_output)

        #The level prediction section
        self.level_output = RepeatVector(self.inner_dim)(self.both_inputs)
        self.level_output = LSTM(self.inner_dim, activation='relu', return_sequences=True)(self.level_output)
        self.level_output = TimeDistributed(Dense(self.inner_dim))(self.level_output)
        self.level_output = Dense(500)(self.level_output)
        self.level_output = Dense(200)(self.level_output)
        self.level_output = Dense(1, activation='softmax')(self.level_output)

        self.part_1_model = Model(inputs = [self.music_input, self.level_input], outputs = [self.music_output, self.level_output])
        self.part_2_model = Model(inputs = [self.music_input, self.level_input], outputs = [self.music_output, self.level_output])
        self.final_model = Model(inputs = [self.music_input, self.level_input], outputs = [self.music_output])

        optimizer = Adam(0.0002, 0.5)

    def plot_models(self):
        plot_model(self.part_1_model, show_shapes=True, to_file='part_1_model.png')
        plot_model(self.part_2_model, show_shapes=True, to_file='part_2_model.png')
        plot_model(self.final_model, show_shapes=True, to_file='final_model.png')

    def train(self, epochs, batch_size=128, sample_interval=50):

        # Load and convert the data
        notes = get_notes()
        n_vocab = len(set(notes))
        X_train, Y_train = prepare_sequences(notes, n_vocab)

        # Adversarial ground truths
        #real = np.ones((batch_size, 1))
        #fake = np.zeros((batch_size, 1))

        # Training the model
        for epoch in range(epochs):

            # Training the discriminator
            # Select a random batch of note sequences
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            real_seqs = X_train[idx]
            real = np.reshape(Y_train[idx], (batch_size, 1))

            level = np.random.randint(1, 10, size=(batch_size, 1))

            # Generate a batch of new note sequences
            gen_seqs = self.generator.predict(level)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(real_seqs, real)
            d_loss_fake = self.discriminator.train_on_batch(gen_seqs, level)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)


            #  Training the Generator
            #noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # Train the generator (to have the discriminator label samples as real)
            g_loss = self.combined.train_on_batch(real, real)

            # Print the progress and save into loss lists
            if epoch % sample_interval == 0:
              print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))
              self.disc_loss.append(d_loss[0])
              self.gen_loss.append(g_loss)

        self.generate(notes)
        self.plot_loss()

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

if __name__ == '__main__':
  rag = RAGNetwork()
  rag.plot_models()

  #gan.train(epochs=5000, batch_size=32, sample_interval=1)

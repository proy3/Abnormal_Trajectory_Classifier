import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.utils.testing import assert_almost_equal, assert_array_almost_equal
from sklearn import svm
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.externals import joblib
from keras.layers import Lambda, Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers import MaxPooling2D, concatenate, merge, Conv2D, UpSampling1D, UpSampling2D, AveragePooling2D
from keras.layers import LeakyReLU, Reshape
from keras.models import Sequential, Model
from keras.models import model_from_yaml
from keras.losses import mse, binary_crossentropy
from keras.optimizers import Adam, RMSprop
from keras.models import save_model
from keras import losses
from keras.utils import plot_model, to_categorical
import keras.backend as K
import tensorflow as tf
import matplotlib.pyplot as plt


repeat_number = 20 # 50
losses_to_check_size = 30
losses_tolerance = 0.6

best_nn_setting = (128,64,32,16,8)

# If at least 5% of the complete trajectory is abnormal, than the latter is abnormal
complete_trajectory_threshold = 0.05

class BuildSimpleOutliersDetectionMethod():
    """
    An simple LDA using sklearn.
    Inspired by the following:
    - http://scikit-learn.org/stable/auto_examples/svm/plot_oneclass.html
    """

    def __init__(self, clf_name):
        """
        Build Traditional Outliers classifier.
        """
        self.clf_name = clf_name
        if self.clf_name == 'One-Class SVM':
            # nu = 0.1 means that the data used in the fitting are normal
            self.clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
        elif self.clf_name == 'Robust covariance':
            # No contamination in the training set
            self.clf = EllipticEnvelope()
        elif self.clf_name == 'Isolation Forest':
            self.clf = IsolationForest()
        elif self.clf_name == 'Local Outlier Factor':
            self.clf = LocalOutlierFactor()
        else:
            assert 'Incorrect model name.'

    def train(self, train_data,
              save_model = True,
              test_saved_model = False,
              model_dir_path = '',
              iteration_number = 0):
        """
        Fit the SVM model.
        :param train_data:
        :param save_model:
        :param test_saved_model:
        :param model_dir_path:
        :param iteration_number:
        :return:
        """
        print('Training the SVC model:')
        # Scale the data
        scaler = MinMaxScaler()
        train_data_scaled = scaler.fit_transform(train_data)
        # Fit the model
        if self.clf_name != 'Local Outlier Factor':
            self.history = self.clf.fit(train_data_scaled)

            # list all data in history
            print(self.history)

        # evaluate the model
        if self.clf_name == 'Local Outlier Factor':
            self.predicted_class = self.clf.fit_predict(train_data_scaled)
        else:
            self.predicted_class = self.clf.predict(train_data_scaled)
        assert self.predicted_class.shape[0] == train_data_scaled.shape[0], ("The predicted data shape is not right!")

        self.accuracy = self.predicted_class[self.predicted_class == 1].size / float(self.predicted_class.size)
        print("acc: %.2f%%" % (self.accuracy*100))

        model_name = 'model_' + str(iteration_number)

        if save_model:
            # Save the scaler
            scaler_filename = model_dir_path + 'scaler_' + model_name + '.pkl'
            joblib.dump(scaler, scaler_filename)

            # Save model to pickle format
            model_filename = model_dir_path + model_name + '.pkl'
            joblib.dump(self.clf, model_filename)

            print('Saved model to disk.')

            if test_saved_model and self.clf_name != 'Local Outlier Factor':
                # Load the saved model and compare the score
                clf = joblib.load(model_filename)

                predicted_class = clf.predict(train_data_scaled)
                assert predicted_class.shape[0] == train_data_scaled.shape[0], \
                    ("The predicted data shape is not right!")

                accuracy = predicted_class[predicted_class == 1].size / float(predicted_class.size)
                print("acc: %.2f%%" % (accuracy*100))

                # Assert if the score is different than the original one
                assert_array_almost_equal(self.predicted_class, predicted_class, decimal=5,
                                          err_msg='Loaded model gives different scores than the original one.')

def test_trained_traditional_model(test_data,
                                   clf_name,
                                   model_dir_path = '',
                                   iteration_number = 0,
                                   is_abnormal = False):
    """
    Uses the saved pre-trained SVM model for scoring the test data.
    :param test_data:
    :param model_dir_path:
    :param iteration_number:
    :return:
    """
    model_name = 'model_' + str(iteration_number)
    # Get the scaler
    scaler_filename = model_dir_path + 'scaler_' + model_name + '.pkl'
    scaler = joblib.load(scaler_filename)

    # Scale the test data
    test_data_scaled = scaler.transform(test_data)

    # Load the saved model and compare the score
    model_filename = model_dir_path + model_name + '.pkl'

    # Load the saved model and compare the score
    clf = joblib.load(model_filename)

    if clf_name == 'Local Outlier Factor':
        predicted_class = clf.fit_predict(test_data_scaled)
    else:
        predicted_class = clf.predict(test_data_scaled)
    assert predicted_class.shape[0] == test_data_scaled.shape[0], ("The predicted data shape is not right!")

    expected_class = -1 if is_abnormal else 1

    accuracy = predicted_class[predicted_class == expected_class].size / float(predicted_class.size)
    print("acc: %.2f%%" % (accuracy*100))

    return accuracy, predicted_class

class BuildSimpleAutoencoder():
    """
    An simple autoencoder using Keras.
    Inspired by the following:
    - https://blog.keras.io/building-autoencoders-in-keras.html;
    - https://github.com/civisanalytics/muffnn/blob/master/muffnn/autoencoder/autoencoder.py
    """

    def __init__(self,
                 input_size=784,
                 hidden_units=(16,),
                 batch_size=128,
                 n_epochs=100,
                 hidden_activation='relu',
                 output_activation='sigmoid',
                 optimiser='rmsprop',
                 loss='mean_squared_error',
                 metrics='accuracy',
                 shuffle_data=True,
                 validation_split=0.20):
        """
        Builds and compiles an autoencoder.
        :param input_size:
        :param hidden_units:
        :param batch_size:
        :param n_epochs:
        :param learning_rate:
        :param hidden_activation:
        :param output_activation:
        :param optimiser:
        :param loss:
        :param shuffle_data:
        """
        self.input_size = input_size
        self.hidden_units = hidden_units
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.optimiser = optimiser
        self.loss = loss
        self.metrics = metrics
        self.shuffle_data = shuffle_data
        self.validation_split = validation_split

        # this is our input placeholder
        _input = Input(shape=tuple([self.input_size]))
        self.encoded = _input

        for units in hidden_units:
            self.encoded = Dense(units, activation=self.hidden_activation)(self.encoded)

        self.decoded = self.encoded

        for units in reversed(hidden_units[:-1]):
            self.decoded = Dense(units, activation=self.hidden_activation)(self.decoded)

        self.decoded = Dense(self.input_size, activation=self.output_activation)(self.decoded)

        # Create and Compile model
        self.autoencoder = Model(_input, self.decoded)
        self.autoencoder.compile(optimizer=self.optimiser, loss=self.loss, metrics=[self.metrics])

    def train(self, train_data,
              save_model = True,
              test_saved_model = False,
              model_dir_path = '',
              iteration_number = 0,
              print_and_plot_history = False, show_plots = False):
        """
        Fit the autoencoder model.
        :param train_data:
        :return:
        """
        print('Training the autoencoder model:')
        # Scale the data
        scaler = MinMaxScaler()
        train_data_scaled = scaler.fit_transform(train_data)
        # Fit the model
        self.history = self.autoencoder.fit(train_data_scaled, train_data_scaled,
                                            validation_split=self.validation_split,
                                            epochs=self.n_epochs,
                                            batch_size=self.batch_size,
                                            shuffle=self.shuffle_data)

        # evaluate the model
        predicted_train_data = self.autoencoder.predict(train_data_scaled)
        assert predicted_train_data.shape == train_data_scaled.shape, ("The predicted data shape is not right!")
        self.mse_per_sample = [mean_squared_error(train_data_scaled[i,:], predicted_train_data[i,:])
                               for i in range(train_data_scaled.shape[0])]

        scores = self.autoencoder.evaluate(train_data_scaled, train_data_scaled)
        self.global_mse = scores[0]
        self.accuracy = scores[1]
        print("%s: %.2f%%" % (self.autoencoder.metrics_names[1], self.accuracy*100))

        model_name = 'model_' + str(iteration_number)

        if save_model:
            # Save the scaler
            scaler_filename = model_dir_path + 'scaler_' + model_name + '.pkl'
            joblib.dump(scaler, scaler_filename)

            # Save model to YAML and weights to HDF5
            model_filename = model_dir_path + model_name + '.yaml'
            weights_filename = model_dir_path + model_name + '.h5'
            compile_filename = model_dir_path + model_name + '.log'

            # Save model
            model_yaml = self.autoencoder.to_yaml()
            with open(model_filename, 'w') as yaml_file:
                yaml_file.write(model_yaml)

            # Save weights
            self.autoencoder.save_weights(weights_filename)

            # Save compile info
            with open(compile_filename, 'wb') as compile_file:
                np.savetxt(compile_file, np.array([self.optimiser, self.loss, self.metrics]), delimiter=',', fmt='%s')

            print('Saved model to disk.')

            if test_saved_model:
                # Load the saved model and compare the score
                # load YAML and create model
                yaml_file = open(model_filename, 'r')
                loaded_model_yaml = yaml_file.read()
                yaml_file.close()
                loaded_model = model_from_yaml(loaded_model_yaml)
                # load weights into new model
                loaded_model.load_weights(weights_filename)
                # load compile info
                compile_info = np.genfromtxt(compile_filename, dtype='str')
                print("Loaded model from disk")

                # evaluate loaded model on test data
                loaded_model.compile(optimizer=compile_info[0], loss=compile_info[1], metrics=[compile_info[2]])
                loaded_scores = loaded_model.evaluate(train_data_scaled, train_data_scaled)
                print("%s: %.2f%%" % (loaded_model.metrics_names[1], loaded_scores[1]*100))

                # Assert if the score is different than the original one
                assert_array_almost_equal(scores, loaded_scores, decimal=5,
                                          err_msg='Loaded model gives different scores than the original one.')

        if print_and_plot_history:
            # list all data in history
            print(self.history.history.keys())
            # summarize history for accuracy
            fig1 = plt.figure()
            plt.plot(self.history.history['acc'])
            plt.plot(self.history.history['val_acc'])
            plt.title('model accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
            figure_name = model_dir_path + model_name + '_accuracy.pdf'
            fig1.savefig(figure_name)
            if show_plots:
                plt.show()
            # summarize history for loss
            fig2 = plt.figure()
            plt.plot(self.history.history['loss'])
            plt.plot(self.history.history['val_loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
            figure_name = model_dir_path + model_name + '_loss.pdf'
            fig2.savefig(figure_name)
            if show_plots:
                plt.show()


def test_trained_ae_model(test_data,
                          model_dir_path = '',
                          iteration_number = 0):
    """
    Uses the saved pre-trained autoencoder for scoring the test data.
    :param X_tests:
    """
    model_name = 'model_' + str(iteration_number)
    # Get the scaler
    scaler_filename = model_dir_path + 'scaler_' + model_name + '.pkl'
    scaler = joblib.load(scaler_filename)

    # Scale the test data
    test_data_scaled = scaler.transform(test_data)

    # Load the saved model and compare the score
    model_filename = model_dir_path + model_name + '.yaml'
    weights_filename = model_dir_path + model_name + '.h5'
    compile_filename = model_dir_path + model_name + '.log'

    # load YAML and create model
    yaml_file = open(model_filename, 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()
    loaded_model = model_from_yaml(loaded_model_yaml)
    # load weights into new model
    loaded_model.load_weights(weights_filename)
    # load compile info
    compile_info = np.genfromtxt(compile_filename, dtype='str')
    print("Loaded model from disk")

    # evaluate loaded model on test data
    loaded_model.compile(optimizer=compile_info[0], loss=compile_info[1], metrics=[compile_info[2]])
    scores = loaded_model.evaluate(test_data_scaled, test_data_scaled)
    print("%s: %.2f%%" % (loaded_model.metrics_names[1], scores[1]*100))

    # extract results
    global_mse = scores[0]
    predicted_test_data = loaded_model.predict(test_data_scaled)
    assert predicted_test_data.shape == test_data_scaled.shape, ("The predicted data shape is not right!")
    mse_per_sample = [mean_squared_error(test_data_scaled[i,:], predicted_test_data[i,:])
                      for i in range(test_data_scaled.shape[0])]

    return global_mse, mse_per_sample

class BuildSimpleVAE():
    """
    An simple autoencoder using Keras.
    Inspired by the following:
    - https://github.com/keras-team/keras/blob/master/examples/variational_autoencoder.py;
    - https://blog.keras.io/building-autoencoders-in-keras.html;
    """

    def __init__(self,
                 original_dim=125,
                 hidden_units=(128,64,32,16,8),
                 batch_size=128,
                 n_epochs=100,
                 hidden_activation='relu',
                 output_activation='sigmoid',
                 optimiser='rmsprop',
                 metrics='accuracy',
                 shuffle_data=True,
                 validation_split=0.20,
                 model_dir_path = ''):
        """
        Builds and compiles an autoencoder.
        :param input_size:
        :param hidden_units:
        :param batch_size:
        :param n_epochs:
        :param learning_rate:
        :param hidden_activation:
        :param output_activation:
        :param optimiser:
        :param loss:
        :param shuffle_data:
        """
        self.original_dim = original_dim
        self.hidden_units = hidden_units
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.optimiser = optimiser
        self.metrics = metrics
        self.shuffle_data = shuffle_data
        self.validation_split = validation_split
        self.latent_dim = hidden_units[-1]
        self.model_dir_path = model_dir_path

        # VAE model = encoder + decoder
        # build encoder model
        # this is our input placeholder
        _input = Input(shape=tuple([self.original_dim]), name='encoder_input')
        _x = _input

        for units in hidden_units[:-1]:
            _x = Dense(units, activation=self.hidden_activation)(_x)

        z_mean = Dense(self.latent_dim, name='z_mean')(_x)
        z_log_var = Dense(self.latent_dim, name='z_log_var')(_x)

        # use reparameterization trick to push the sampling out as input
        # note that "output_shape" isn't necessary with the TensorFlow backend
        z = Lambda(self._sampling, output_shape=(self.latent_dim,), name='z')([z_mean, z_log_var])

        # instantiate encoder model
        self.encoder = Model(_input, [z_mean, z_log_var, z], name='encoder')
        self.encoder.summary()
        plot_encoder_filename = self.model_dir_path + 'vae_mlp_encoder.pdf'
        plot_model(self.encoder, to_file=plot_encoder_filename, show_shapes=True)

        # build decoder model
        _latent_input = Input(shape=(self.latent_dim,), name='z_sampling')
        _x = _latent_input

        for units in reversed(hidden_units[:-1]):
            _x = Dense(units, activation=self.hidden_activation)(_x)

        _output = Dense(self.original_dim, activation=self.output_activation)(_x)

        # instantiate decoder model
        self.decoder = Model(_latent_input, _output, name='decoder')
        self.decoder.summary()
        plot_decoder_filename = self.model_dir_path + 'vae_mlp_decoder.pdf'
        plot_model(self.decoder, to_file=plot_decoder_filename, show_shapes=True)

        # instantiate VAE model
        _output = self.decoder(self.encoder(_input)[-1])
        self.vae = Model(_input, _output, name='vae_mlp')

        # Calculate losses
        self.reconstruction_loss = mse(_input, _output)
        self.reconstruction_loss *= self.original_dim
        self.kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        self.kl_loss = K.sum(self.kl_loss, axis=-1)
        self.kl_loss *= -0.5
        self.vae_loss = K.mean(self.reconstruction_loss + self.kl_loss)

        self.vae.add_loss(self.vae_loss)
        self.vae.compile(optimizer=self.optimiser, metrics=[self.metrics])
        self.vae.summary()
        plot_vae_filename = self.model_dir_path + 'vae_mlp.pdf'
        plot_model(self.vae, to_file=plot_vae_filename, show_shapes=True)

    def _sampling(self, args):
        """Reparameterization trick by sampling fr an isotropic unit Gaussian.
        # Arguments:
            args (tensor): mean and log of variance of Q(z|X)
        # Returns:
            z (tensor): sampled latent vector
        """

        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        # by default, random_normal has mean=0 and std=1.0
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    def train(self, train_data,
              save_model = True,
              iteration_number = 0,
              print_and_plot_history = False, show_plots = False):
        """
        Fit the autoencoder model.
        :param train_data:
        :return:
        """
        print('Training the autoencoder model:')
        # Scale the data
        scaler = MinMaxScaler()
        train_data_scaled = scaler.fit_transform(train_data)
        # Fit the model
        self.history = self.vae.fit(train_data_scaled,
                                    validation_split=self.validation_split,
                                    epochs=self.n_epochs,
                                    batch_size=self.batch_size,
                                    shuffle=self.shuffle_data)

        # evaluate the model
        predicted_train_data = self.vae.predict(train_data_scaled)
        assert predicted_train_data.shape == train_data_scaled.shape, ("The predicted data shape is not right!")
        self.mse_per_sample = [mean_squared_error(train_data_scaled[i,:], predicted_train_data[i,:])
                               for i in range(train_data_scaled.shape[0])]

        scores = self.vae.evaluate(train_data_scaled)
        self.global_mse = np.mean(self.mse_per_sample)
        self.accuracy = scores
        print("%s: %.4f" % (self.vae.metrics_names[0], self.accuracy))

        model_name = 'model_' + str(iteration_number)

        if save_model:
            # Save the scaler
            scaler_filename = self.model_dir_path + 'scaler_' + model_name + '.pkl'
            joblib.dump(scaler, scaler_filename)

            # Save weights to HDF5
            weights_filename = self.model_dir_path + model_name + '.h5'

            # Save weights
            self.vae.save_weights(weights_filename)

            print('Saved model to disk.')

        if print_and_plot_history:
            # list all data in history
            print(self.history.history.keys())
            # summarize history for loss
            fig1 = plt.figure()
            plt.plot(self.history.history['loss'])
            plt.plot(self.history.history['val_loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
            figure_name = self.model_dir_path + model_name + '_loss.pdf'
            fig1.savefig(figure_name)
            if show_plots:
                plt.show()

    def load_weights(self,
                     iteration_number = 0):
        """
        Load the weights from file to the model.
        :param model_dir_path:
        :param iteration_number:
        :return:
        """
        model_name = 'model_' + str(iteration_number)
        weights_filename = self.model_dir_path + model_name + '.h5'

        # load weights into new model
        self.vae.load_weights(weights_filename)
        print("Loaded weights from disk")

    def test_model(self,
                   test_data,
                   iteration_number = 0):
        """
        Apply the learned model to the test samples and get scores.
        :param test_data:
        :param model_dir_path:
        :param iteration_number:
        :return:
        """
        model_name = 'model_' + str(iteration_number)
        # Get the scaler
        scaler_filename = self.model_dir_path + 'scaler_' + model_name + '.pkl'
        scaler = joblib.load(scaler_filename)

        # Scale the test data
        test_data_scaled = scaler.transform(test_data)

        # evaluate the learned model on test data
        scores = self.vae.evaluate(test_data_scaled)
        print("%s: %.4f" % (self.vae.metrics_names[0], scores))

        # extract results
        predicted_test_data = self.vae.predict(test_data_scaled)
        assert predicted_test_data.shape == test_data_scaled.shape, ("The predicted data shape is not right!")
        mse_per_sample = [mean_squared_error(test_data_scaled[i,:], predicted_test_data[i,:])
                          for i in range(test_data_scaled.shape[0])]
        global_mse = np.mean(mse_per_sample)

        return global_mse, mse_per_sample

class BuildSimpleAAE():
    """
    An simple autoencoder using Keras.
    Inspired by the following:
    - https://blog.keras.io/building-autoencoders-in-keras.html;
    - https://github.com/civisanalytics/muffnn/blob/master/muffnn/autoencoder/autoencoder.py
    """

    def __init__(self,
                 original_dim=125,
                 hidden_units=(128,64,32,16,8),
                 batch_size=128,
                 n_epochs=20000,
                 hidden_activation='relu',
                 output_activation='sigmoid',
                 metrics='accuracy',
                 model_dir_path = ''):
        """
        Builds and compiles an autoencoder.
        :param input_size:
        :param hidden_units:
        :param batch_size:
        :param n_epochs:
        :param learning_rate:
        :param hidden_activation:
        :param output_activation:
        :param optimiser:
        :param loss:
        :param shuffle_data:
        """
        self.original_dim = original_dim
        self.hidden_units = hidden_units
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.metrics = metrics
        self.latent_dim = hidden_units[-1]
        self.model_dir_path = model_dir_path

        self.optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=self.optimizer,
                                   metrics=[self.metrics])

        # Build the encoder / decoder
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()

        # Save network's diagrams
        self.discriminator.summary()
        plot_discriminator_filename = self.model_dir_path + 'aae_mlp_discriminator.pdf'
        plot_model(self.discriminator, to_file=plot_discriminator_filename, show_shapes=True)

        self.encoder.summary()
        plot_encoder_filename = self.model_dir_path + 'aae_mlp_encoder.pdf'
        plot_model(self.encoder, to_file=plot_encoder_filename, show_shapes=True)

        self.decoder.summary()
        plot_decoder_filename = self.model_dir_path + 'aae_mlp_decoder.pdf'
        plot_model(self.decoder, to_file=plot_decoder_filename, show_shapes=True)

        # this is our input placeholder
        _input = Input(shape=tuple([self.original_dim]), name='encoder_input')

        # The generator takes the image, encodes it and reconstructs it
        # from the encoding
        encoded_repr = self.encoder(_input)
        reconstructed_img = self.decoder(encoded_repr)

        # For the adversarial_autoencoder model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator determines validity of the encoding
        validity = self.discriminator(encoded_repr)

        # The adversarial_autoencoder model  (stacked generator and discriminator)
        self.aae = Model(_input, [reconstructed_img, validity])
        self.aae.compile(loss=['mse', 'binary_crossentropy'],
                         loss_weights=[0.999, 0.001],
                         optimizer=self.optimizer,
                         metrics=[self.metrics])

        self.aae.summary()
        plot_aae_filename = self.model_dir_path + 'aae_mlp.pdf'
        plot_model(self.aae, to_file=plot_aae_filename, show_shapes=True)

    def build_encoder(self):
        # Encoder
        _input = Input(shape=tuple([self.original_dim]))
        _x = _input

        for units in self.hidden_units[:-1]:
            _x = Dense(units, activation=self.hidden_activation)(_x)

        mu = Dense(self.latent_dim)(_x)
        log_var = Dense(self.latent_dim)(_x)
        latent_repr = merge([mu, log_var],
                            mode=lambda p: p[0] + K.random_normal(K.shape(p[0])) * K.exp(p[1] / 2),
                            output_shape=lambda p: p[0])

        return Model(_input, latent_repr, name='encoder')

    def build_decoder(self):
        # Decoder
        _latent_input = Input(shape=(self.latent_dim,), name='z_sampling')
        _x = _latent_input

        for units in reversed(self.hidden_units[:-1]):
            _x = Dense(units, activation=self.hidden_activation)(_x)

        _output = Dense(self.original_dim, activation=self.output_activation)(_x)

        return Model(_latent_input, _output, name='decoder')

    def build_discriminator(self):
        # Discriminator
        model = Sequential()

        model.add(Dense(128, input_dim=self.latent_dim, activation=self.hidden_activation))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(64, activation=self.hidden_activation))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(32, activation=self.hidden_activation))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(16, activation=self.hidden_activation))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(8, activation=self.hidden_activation))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation=self.output_activation))
        model.summary()

        encoded_repr = Input(shape=(self.latent_dim, ))
        validity = model(encoded_repr)

        return Model(encoded_repr, validity, name='discriminator')

    def train(self, train_data,
              save_model = True,
              iteration_number = 0,
              print_and_plot_history = False, show_plots = False):
        """
        Fit the autoencoder model.
        :param train_data:
        :return:
        """
        print('Training the autoencoder model:')
        # Scale the data
        scaler = MinMaxScaler()
        train_data_scaled = scaler.fit_transform(train_data)

        #==================================================================
        # Train the AAE model
        # Adversarial ground truths
        valid = np.ones((self.batch_size, 1))
        fake = np.zeros((self.batch_size, 1))

        all_d_loss = []
        all_d_acc = []
        all_g_loss = []
        all_g_mse = []

        for epoch in range(self.n_epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of images
            idx = np.random.randint(0, train_data_scaled.shape[0], self.batch_size)
            inputs = train_data_scaled[idx]

            latent_fake = self.encoder.predict(inputs)
            latent_real = np.random.normal(size=(self.batch_size, self.latent_dim))

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(latent_real, valid)
            d_loss_fake = self.discriminator.train_on_batch(latent_fake, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator
            g_loss = self.aae.train_on_batch(inputs, [inputs, valid])

            # Store the loss values for plot
            all_d_loss.append(d_loss[0])
            all_d_acc.append(100*d_loss[1])
            all_g_loss.append(g_loss[0])
            all_g_mse.append(g_loss[1])

            # Plot the progress
            print ("%d [D loss: %f, acc: %.2f%%] [G loss: %f, mse: %f]" %
                   (epoch, d_loss[0], 100*d_loss[1], g_loss[0], g_loss[1]))
        #==================================================================

        # evaluate the model
        predicted_train_data = self.decoder.predict(self.encoder.predict(train_data_scaled))
        assert predicted_train_data.shape == train_data_scaled.shape, ("The predicted data shape is not right!")
        self.mse_per_sample = [mean_squared_error(train_data_scaled[i,:], predicted_train_data[i,:])
                               for i in range(train_data_scaled.shape[0])]

        self.global_mse = np.mean(self.mse_per_sample)

        model_name = 'model_' + str(iteration_number)
        discriminator_name = 'model_d_' + str(iteration_number)

        if save_model:
            # Save the scaler
            scaler_filename = self.model_dir_path + 'scaler_' + model_name + '.pkl'
            joblib.dump(scaler, scaler_filename)

            # Save weights to HDF5
            weights_filename = self.model_dir_path + model_name + '.h5'

            # Save weights
            self.aae.save_weights(weights_filename)

            print('Saved generator to disk.')

            # Save discriminator
            weights_filename_d = self.model_dir_path + discriminator_name + '.h5'

            self.discriminator.save_weights(weights_filename_d)

            print('Saved discriminator to disk.')

        if print_and_plot_history:
            _x = range(self.n_epochs)
            # summarize loss
            fig1 = plt.figure()
            plt.plot(_x, all_d_loss, 'b-', _x, all_g_loss, 'g--')
            plt.title('AAE learning loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['discriminator', 'generator'], loc='upper left')
            figure_name = self.model_dir_path + model_name + '_loss.pdf'
            fig1.savefig(figure_name)
            if show_plots:
                plt.show()
            # summarize discriminator accuracy
            fig2 = plt.figure()
            plt.plot(_x, all_d_acc, 'k-')
            plt.title('AAE learning discriminator accuracy')
            plt.ylabel('acc')
            plt.xlabel('epoch')
            figure_name = self.model_dir_path + model_name + '_d_acc.pdf'
            fig2.savefig(figure_name)
            if show_plots:
                plt.show()
            # summarize generator error
            fig3 = plt.figure()
            plt.plot(_x, all_g_mse, 'k-')
            plt.title('AAE learning generator error')
            plt.ylabel('mse')
            plt.xlabel('epoch')
            figure_name = self.model_dir_path + model_name + '_g_mse.pdf'
            fig3.savefig(figure_name)
            if show_plots:
                plt.show()

    def load_weights(self,
                     iteration_number = 0):
        """
        Load the weights from file to the model.
        :param model_dir_path:
        :param iteration_number:
        :return:
        """
        model_name = 'model_' + str(iteration_number)
        discriminator_name = 'model_d_' + str(iteration_number)
        weights_filename = self.model_dir_path + model_name + '.h5'
        weights_filename_d = self.model_dir_path + discriminator_name + '.h5'

        # load weights into new model
        self.aae.load_weights(weights_filename)
        self.discriminator.load_weights(weights_filename_d)
        print("Loaded weights from disk")

    def test_model(self,
                   test_data,
                   iteration_number = 0):
        """
        Apply the learned model to the test samples and get scores.
        :param test_data:
        :param model_dir_path:
        :param iteration_number:
        :return:
        """
        model_name = 'model_' + str(iteration_number)
        # Get the scaler
        scaler_filename = self.model_dir_path + 'scaler_' + model_name + '.pkl'
        scaler = joblib.load(scaler_filename)

        # Scale the test data
        test_data_scaled = scaler.transform(test_data)

        # extract results
        predicted_test_data = self.decoder.predict(self.encoder.predict(test_data_scaled))
        assert predicted_test_data.shape == test_data_scaled.shape, ("The predicted data shape is not right!")
        mse_per_sample = [mean_squared_error(test_data_scaled[i,:], predicted_test_data[i,:])
                          for i in range(test_data_scaled.shape[0])]
        global_mse = np.mean(mse_per_sample)

        return global_mse, mse_per_sample

class BuildSimpleBiGAN():
    """
    An simple bidirectional GAN using Keras.
    Inspired by the following:
    - https://blog.keras.io/building-autoencoders-in-keras.html;
    - https://github.com/civisanalytics/muffnn/blob/master/muffnn/autoencoder/autoencoder.py
    """

    def __init__(self,
                 original_dim=125,
                 hidden_units=(128,64,32,16,8),
                 batch_size=128,
                 n_epochs=10000,
                 hidden_activation='relu',
                 output_activation='sigmoid',
                 optimizer='rmsprop',
                 loss='mean_squared_error',
                 metrics='accuracy',
                 model_dir_path = '',
                 iteration_number = 0,
                 version = 0):
        """
        Builds and compiles an autoencoder.
        :param input_size:
        :param hidden_units:
        :param batch_size:
        :param n_epochs:
        :param learning_rate:
        :param hidden_activation:
        :param output_activation:
        :param optimiser:
        :param loss:
        :param shuffle_data:
        """
        self.original_dim = original_dim
        self.hidden_units = hidden_units
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.latent_dim = hidden_units[-1]
        self.model_dir_path = model_dir_path
        self.iteration_number = iteration_number
        self.version = version

        self.model_name = 'model_' + str(self.iteration_number)
        self.discriminator_name = 'model_d_' + str(self.iteration_number)

        #======================================================
        # Test
        self.loss = 'binary_crossentropy'
        self.optimizer = Adam(0.0002, 0.5)
        #======================================================

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=[self.loss],
                                   optimizer=self.optimizer,
                                   metrics=[self.metrics])

        # Build the generator
        self.generator = self.build_generator()

        # Build the encoder
        self.encoder = self.build_encoder()

        if self.iteration_number == 0:
            # Save network's diagrams
            self.discriminator.summary()
            plot_discriminator_filename = self.model_dir_path + 'bigan_discriminator.pdf'
            plot_model(self.discriminator, to_file=plot_discriminator_filename, show_shapes=True)

            self.encoder.summary()
            plot_encoder_filename = self.model_dir_path + 'bigan_encoder.pdf'
            plot_model(self.encoder, to_file=plot_encoder_filename, show_shapes=True)

            self.generator.summary()
            plot_generator_filename = self.model_dir_path + 'bigan_generator.pdf'
            plot_model(self.generator, to_file=plot_generator_filename, show_shapes=True)

        # The part of the bigan that trains the discriminator and encoder
        self.discriminator.trainable = False

        # Generate image from sampled noise
        _z = Input(shape=(self.latent_dim, ))
        _g_z = self.generator(_z)

        # Encode input
        _x = Input(shape=tuple([self.original_dim]), name='encoder_input')
        _e_x = self.encoder(_x)

        # Latent -> img is fake, and img -> latent is valid
        _fake = self.discriminator([_z, _g_z])
        _valid = self.discriminator([_e_x, _x])

        # Set up and compile the combined model
        # Trains generator to fool the discriminator
        self.bigan_generator = Model([_z, _x], [_fake, _valid])
        self.bigan_generator.compile(loss=[self.loss, self.loss],
                                     optimizer=self.optimizer)

        if self.iteration_number == 0:
            self.bigan_generator.summary()
            plot_bigan_generator_filename = self.model_dir_path + 'bigan_generator_mlp.pdf'
            plot_model(self.bigan_generator, to_file=plot_bigan_generator_filename, show_shapes=True)

    def build_encoder(self):
        # Encoder
        _x = Input(shape=tuple([self.original_dim]), name='encoder_input')
        _e_x = _x

        for units in self.hidden_units:
            _e_x = Dense(units, activation=self.hidden_activation)(_e_x)

        return Model(_x, _e_x, name='encoder')

    def build_generator(self):
        # Decoder
        _z = Input(shape=(self.latent_dim,), name='z_sampling')
        _g = _z

        for units in reversed(self.hidden_units[:-1]):
            _g = Dense(units, activation=self.hidden_activation)(_g)

        _g_z = Dense(self.original_dim, activation=self.output_activation, name='generated_data')(_g)

        return Model(_z, _g_z, name='generator')

    def build_discriminator(self):
        # Discriminator
        _z = Input(shape=(self.latent_dim,), name='latent_data')
        _data = Input(shape=tuple([self.original_dim]), name='data')
        _d_in = concatenate([_z, _data])

        _d = _d_in

        for units in self.hidden_units:
            _d = Dense(units, activation=self.hidden_activation)(_d)

        _v = Dense(1, activation=self.output_activation, name='validity')(_d)

        return Model([_z, _data], _v, name='discriminator')

    def train(self, train_data,
              save_model = True,
              print_and_plot_history = False, show_plots = False):
        """
        Fit the autoencoder model.
        :param train_data:
        :return:
        """
        print('Training the autoencoder model:')
        # Scale the data
        scaler = MinMaxScaler()
        train_data_scaled = scaler.fit_transform(train_data)

        #==================================================================
        # Train the BiGAN model
        # Adversarial ground truths
        valid = np.ones((self.batch_size, 1))
        fake = np.zeros((self.batch_size, 1))

        all_d_loss = []
        all_d_acc = []
        all_g_loss = []

        for epoch in range(self.n_epochs):
            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Sample noise and generate img
            z = np.random.normal(loc=0.5, scale=0.25, size=(self.batch_size, self.latent_dim))
            gen_data = self.generator.predict(z)

            # Select a random batch of images and encode
            idx = np.random.randint(0, train_data_scaled.shape[0], self.batch_size)
            data = train_data_scaled[idx]
            enc_data = self.encoder.predict(data)

            # Train the discriminator (img -> z is valid, z -> img is fake)
            d_loss_real = self.discriminator.train_on_batch([enc_data, data], valid)
            d_loss_fake = self.discriminator.train_on_batch([z, gen_data], fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator (z -> img is valid and img -> z is is invalid)
            g_loss = self.bigan_generator.train_on_batch([z, data], [valid, fake])

            # Plot the progress
            print ("%d [D loss: %f, acc: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss[0]))

            # Store the loss values for plot
            all_d_loss.append(d_loss[0])
            all_d_acc.append(100*d_loss[1])
            all_g_loss.append(g_loss[0])
        #==================================================================

        # evaluate the model
        self.mse_per_sample, self.global_mse = self._evaluate_model(train_data_scaled)

        if save_model:
            # Save the scaler
            scaler_filename = self.model_dir_path + 'scaler_' + self.model_name + '.pkl'
            joblib.dump(scaler, scaler_filename)

            # Save weights to HDF5
            weights_filename = self.model_dir_path + self.model_name + '.h5'

            # Save weights
            self.bigan_generator.save_weights(weights_filename)

            print('Saved generator to disk.')

            # Save discriminator
            weights_filename_d = self.model_dir_path + self.discriminator_name + '.h5'

            self.discriminator.save_weights(weights_filename_d)

            print('Saved discriminator to disk.')

        if print_and_plot_history:
            _x = range(self.n_epochs)
            # summarize loss
            fig1 = plt.figure()
            plt.plot(_x, all_d_loss, 'b-', _x, all_g_loss, 'g--')
            plt.title('BiGAN learning loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['discriminator', 'generator'], loc='upper left')
            figure_name = self.model_dir_path + self.model_name + '_loss.pdf'
            fig1.savefig(figure_name)
            if show_plots:
                plt.show()
            # summarize discriminator accuracy
            fig2 = plt.figure()
            plt.plot(_x, all_d_acc, 'k-')
            plt.title('BiGAN learning discriminator accuracy')
            plt.ylabel('acc')
            plt.xlabel('epoch')
            figure_name = self.model_dir_path + self.model_name + '_d_acc.pdf'
            fig2.savefig(figure_name)
            if show_plots:
                plt.show()

    def _evaluate_model(self, scaled_data):
        """
        Evaluates the trained model with the scaled data and determines the score.
        :param scaled_data:
        :return:
        """
        _e_data = self.encoder.predict(scaled_data)
        _p_data = self.discriminator.predict([_e_data, scaled_data])
        if self.version == 0:
            mse_per_sample = [mean_squared_error([_p], [1]) for _p in _p_data.flatten()]
        else:
            # Sample noise and generate img
            _z = np.random.normal(size=(self.batch_size, self.latent_dim))
            _gen_data = self.generator.predict(_z)
            _p_gen = self.discriminator.predict([_z, _gen_data])
            mse_per_sample = [mean_squared_error(_p * np.ones(_p_gen.shape), _p_gen) for _p in _p_data.flatten()]

        assert len(mse_per_sample) == scaled_data.shape[0], ("The predicted data shape is not right!")

        global_mse = np.mean(mse_per_sample)

        return mse_per_sample, global_mse

    def load_weights(self):
        """
        Load the weights from file to the model.
        :param model_dir_path:
        :param iteration_number:
        :return:
        """
        weights_filename = self.model_dir_path + self.model_name + '.h5'
        weights_filename_d = self.model_dir_path + self.discriminator_name + '.h5'

        # load weights into new model
        self.bigan_generator.load_weights(weights_filename)
        self.discriminator.load_weights(weights_filename_d)
        print("Loaded weights from disk")

    def test_model(self,
                   test_data):
        """
        Apply the learned model to the test samples and get scores.
        :param test_data:
        :param model_dir_path:
        :param iteration_number:
        :return:
        """
        # Get the scaler
        scaler_filename = self.model_dir_path + 'scaler_' + self.model_name + '.pkl'
        scaler = joblib.load(scaler_filename)

        # Scale the test data
        test_data_scaled = scaler.transform(test_data)

        # evaluate the model
        mse_per_sample, global_mse = self._evaluate_model(test_data_scaled)

        return global_mse, mse_per_sample

class BuildOriginalBiGAN():
    """
    An simple bidirectional GAN using Keras.
    Inspired by the following:
    - https://blog.keras.io/building-autoencoders-in-keras.html;
    - https://github.com/civisanalytics/muffnn/blob/master/muffnn/autoencoder/autoencoder.py
    """

    def __init__(self,
                 original_dim=125,
                 batch_size=32,
                 n_epochs=40000,
                 loss='binary_crossentropy',
                 metrics='accuracy',
                 model_dir_path = '',
                 iteration_number = 0,
                 version = 0):
        """
        Builds and compiles an autoencoder.
        :param input_size:
        :param hidden_units:
        :param batch_size:
        :param n_epochs:
        :param learning_rate:
        :param hidden_activation:
        :param output_activation:
        :param optimiser:
        :param loss:
        :param shuffle_data:
        """
        self.original_dim = original_dim
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.loss = loss
        self.metrics = metrics
        self.model_dir_path = model_dir_path
        self.iteration_number = iteration_number
        self.version = version

        self.model_name = 'model_' + str(self.iteration_number)
        self.discriminator_name = 'model_d_' + str(self.iteration_number)

        self.latent_dim = 100
        self.optimizer = Adam(0.0002, 0.5)

        # This factor increases the input size and makes it possible for a square image transformation layer
        self.input_scaling_factor = 5
        self.input_2d_size = int(np.sqrt(self.original_dim * self.input_scaling_factor))
        self.img_shape = (self.input_2d_size, self.input_2d_size, 1)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=[self.loss],
                                   optimizer=self.optimizer,
                                   metrics=[self.metrics])

        # Build the generator
        self.generator = self.build_generator()

        # Build the encoder
        self.encoder = self.build_encoder()

        # The part of the bigan that trains the discriminator and encoder
        self.discriminator.trainable = False

        # Generate image from sampled noise
        _z = Input(shape=(self.latent_dim, ))
        _g_z = self.generator(_z)

        # Encode input
        _x = Input(shape=(1, self.original_dim, ), name='input_data')
        _e_x = self.encoder(_x)

        # Latent -> img is fake, and img -> latent is valid
        _fake = self.discriminator([_z, _g_z])
        _valid = self.discriminator([_x, _e_x])

        # Set up and compile the combined model
        # Trains generator to fool the discriminator
        self.bigan_generator = Model([_z, _x], [_fake, _valid])
        self.bigan_generator.compile(loss=[self.loss, self.loss],
                                     optimizer=self.optimizer)

        if self.iteration_number == 0:
            self.bigan_generator.summary()
            plot_bigan_generator_filename = self.model_dir_path + 'bigan_generator.pdf'
            plot_model(self.bigan_generator, to_file=plot_bigan_generator_filename, show_shapes=True)

    def build_encoder(self):
        # Encoder
        model = Sequential()

        model.add(UpSampling1D(self.input_scaling_factor, input_shape=(1,self.original_dim,)))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(self.latent_dim))

        model.summary()

        x = Input(shape=(1, self.original_dim, ), name='input_data')
        z = model(x)

        return Model(x, z)

    def build_generator(self):
        # Decoder
        model = Sequential()

        model.add(Dense(512, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.img_shape)))
        model.add(Dense(self.original_dim, activation='tanh'))
        model.add(Reshape((1, self.original_dim, )))

        model.summary()

        z = Input(shape=(self.latent_dim,))
        gen_img = model(z)

        return Model(z, gen_img)

    def build_discriminator(self):
        # Discriminator
        z = Input(shape=(self.latent_dim, ))
        x = Input(shape=(1, self.original_dim, ), name='input_data')
        img=UpSampling1D(self.input_scaling_factor)(x)
        d_in = concatenate([z, Flatten()(img)])

        model = Dense(1024)(d_in)
        model = LeakyReLU(alpha=0.2)(model)
        model = Dropout(0.5)(model)
        model = Dense(1024)(model)
        model = LeakyReLU(alpha=0.2)(model)
        model = Dropout(0.5)(model)
        model = Dense(1024)(model)
        model = LeakyReLU(alpha=0.2)(model)
        model = Dropout(0.5)(model)
        validity = Dense(1, activation="sigmoid")(model)

        return Model([z, x], validity)

    def train(self, train_data,
              save_model = True,
              print_and_plot_history = False, show_plots = False):
        """
        Fit the autoencoder model.
        :param train_data:
        :return:
        """
        print('Training the autoencoder model:')
        # Scale the data
        scaler = MinMaxScaler()
        train_data_scaled = scaler.fit_transform(train_data)

        #==================================================================
        # Train the BiGAN model
        # Adversarial ground truths
        valid = np.ones((self.batch_size, 1))
        fake = np.zeros((self.batch_size, 1))

        all_d_loss = []
        all_d_acc = []
        all_g_loss = []

        for epoch in range(self.n_epochs):
            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Sample noise and generate img
            z = np.random.normal(size=(self.batch_size, self.latent_dim))
            gen_data = self.generator.predict(z)

            # Select a random batch of images and encode
            idx = np.random.randint(0, train_data_scaled.shape[0], self.batch_size)
            data = train_data_scaled[idx]
            enc_data = self.encoder.predict(data)

            # Train the discriminator (img -> z is valid, z -> img is fake)
            d_loss_real = self.discriminator.train_on_batch([enc_data, data], valid)
            d_loss_fake = self.discriminator.train_on_batch([z, gen_data], fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator (z -> img is valid and img -> z is is invalid)
            g_loss = self.bigan_generator.train_on_batch([z, data], [valid, fake])

            # Plot the progress
            print ("%d [D loss: %f, acc: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss[0]))

            # Store the loss values for plot
            all_d_loss.append(d_loss[0])
            all_d_acc.append(100*d_loss[1])
            all_g_loss.append(g_loss[0])
        #==================================================================

        # evaluate the model
        self.mse_per_sample, self.global_mse = self._evaluate_model(train_data_scaled)

        if save_model:
            # Save the scaler
            scaler_filename = self.model_dir_path + 'scaler_' + self.model_name + '.pkl'
            joblib.dump(scaler, scaler_filename)

            # Save weights to HDF5
            weights_filename = self.model_dir_path + self.model_name + '.h5'

            # Save weights
            self.bigan_generator.save_weights(weights_filename)

            print('Saved generator to disk.')

            # Save discriminator
            weights_filename_d = self.model_dir_path + self.discriminator_name + '.h5'

            self.discriminator.save_weights(weights_filename_d)

            print('Saved discriminator to disk.')

        if print_and_plot_history:
            _x = range(self.n_epochs)
            # summarize loss
            fig1 = plt.figure()
            plt.plot(_x, all_d_loss, 'b-', _x, all_g_loss, 'g--')
            plt.title('BiGAN learning loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['discriminator', 'generator'], loc='upper left')
            figure_name = self.model_dir_path + self.model_name + '_loss.pdf'
            fig1.savefig(figure_name)
            if show_plots:
                plt.show()
            # summarize discriminator accuracy
            fig2 = plt.figure()
            plt.plot(_x, all_d_acc, 'k-')
            plt.title('BiGAN learning discriminator accuracy')
            plt.ylabel('acc')
            plt.xlabel('epoch')
            figure_name = self.model_dir_path + self.model_name + '_d_acc.pdf'
            fig2.savefig(figure_name)
            if show_plots:
                plt.show()

    def _evaluate_model(self, scaled_data):
        """
        Evaluates the trained model with the scaled data and determines the score.
        :param scaled_data:
        :return:
        """
        _e_data = self.encoder.predict(scaled_data)
        _p_data = self.discriminator.predict([_e_data, scaled_data])
        if self.version == 0:
            mse_per_sample = [mean_squared_error([_p], [1]) for _p in _p_data.flatten()]
        else:
            # Sample noise and generate img
            _z = np.random.normal(size=(self.batch_size, self.latent_dim))
            _gen_data = self.generator.predict(_z)
            _p_gen = self.discriminator.predict([_z, _gen_data])
            mse_per_sample = [mean_squared_error(_p * np.ones(_p_gen.shape), _p_gen) for _p in _p_data.flatten()]

        assert len(mse_per_sample) == scaled_data.shape[0], ("The predicted data shape is not right!")

        global_mse = np.mean(mse_per_sample)

        return mse_per_sample, global_mse

    def load_weights(self):
        """
        Load the weights from file to the model.
        :param model_dir_path:
        :param iteration_number:
        :return:
        """
        weights_filename = self.model_dir_path + self.model_name + '.h5'
        weights_filename_d = self.model_dir_path + self.discriminator_name + '.h5'

        # load weights into new model
        self.bigan_generator.load_weights(weights_filename)
        self.discriminator.load_weights(weights_filename_d)
        print("Loaded weights from disk")

    def test_model(self,
                   test_data):
        """
        Apply the learned model to the test samples and get scores.
        :param test_data:
        :param model_dir_path:
        :param iteration_number:
        :return:
        """
        # Get the scaler
        scaler_filename = self.model_dir_path + 'scaler_' + self.model_name + '.pkl'
        scaler = joblib.load(scaler_filename)

        # Scale the test data
        test_data_scaled = scaler.transform(test_data)

        # evaluate the model
        mse_per_sample, global_mse = self._evaluate_model(test_data_scaled)

        return global_mse, mse_per_sample

class BuildSimpleBiGANAAE():
    """
    An simple bidirectional GAN using Keras.
    Inspired by the following:
    - https://blog.keras.io/building-autoencoders-in-keras.html;
    - https://github.com/civisanalytics/muffnn/blob/master/muffnn/autoencoder/autoencoder.py
    """

    def __init__(self,
                 original_dim=125,
                 hidden_units=(128,64,32,16,8),
                 batch_size=128,
                 n_epochs=10000,
                 aae_epochs=100,
                 hidden_activation='relu',
                 output_activation='sigmoid',
                 optimizer='rmsprop',
                 loss='mean_squared_error',
                 metrics='accuracy',
                 model_dir_path = '',
                 iteration_number = 0,
                 version = 0):
        """
        Builds and compiles an autoencoder.
        :param input_size:
        :param hidden_units:
        :param batch_size:
        :param n_epochs:
        :param learning_rate:
        :param hidden_activation:
        :param output_activation:
        :param optimiser:
        :param loss:
        :param shuffle_data:
        """
        self.original_dim = original_dim
        self.hidden_units = hidden_units
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.aae_epochs = aae_epochs
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.latent_dim = hidden_units[-1]
        self.model_dir_path = model_dir_path
        self.iteration_number = iteration_number
        self.version = version

        self.model_name = 'model_' + str(self.iteration_number)
        self.discriminator_name = 'model_d_' + str(self.iteration_number)

        #======================================================
        # Test
        self.loss = 'binary_crossentropy'
        self.optimizer = Adam(0.0002, 0.5)
        #======================================================

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=[self.loss],
                                   optimizer=self.optimizer,
                                   metrics=[self.metrics])

        # Build and compile the generator
        self.build_and_compile_generator()
        self.generator = self.aae_generator

        # Build the encoder
        self.encoder = self.build_encoder()

        if self.iteration_number == 0:
            # Save network's diagrams
            self.discriminator.summary()
            plot_discriminator_filename = self.model_dir_path + 'bigan_discriminator.pdf'
            plot_model(self.discriminator, to_file=plot_discriminator_filename, show_shapes=True)

            self.encoder.summary()
            plot_encoder_filename = self.model_dir_path + 'bigan_encoder.pdf'
            plot_model(self.encoder, to_file=plot_encoder_filename, show_shapes=True)

        # The part of the bigan that trains the discriminator and encoder
        self.discriminator.trainable = False

        # Generate image from sampled noise
        _z = Input(shape=(self.latent_dim, ))
        _g_z = self.aae_generator(_z)

        # Encode input
        _x = Input(shape=tuple([self.original_dim]), name='encoder_input')
        _e_x = self.encoder(_x)

        # Latent -> img is fake, and img -> latent is valid
        _fake = self.discriminator([_z, _g_z])
        _valid = self.discriminator([_e_x, _x])

        # Set up and compile the combined model
        # Trains generator to fool the discriminator
        self.bigan_generator = Model([_z, _x], [_fake, _valid])
        self.bigan_generator.compile(loss=[self.loss, self.loss],
                                     optimizer=self.optimizer)

        if self.iteration_number == 0:
            self.bigan_generator.summary()
            plot_bigan_generator_filename = self.model_dir_path + 'bigan_generator_mlp.pdf'
            plot_model(self.bigan_generator, to_file=plot_bigan_generator_filename, show_shapes=True)

        self.random_z_samples = np.random.normal(size=(self.batch_size, self.latent_dim))

    def build_encoder(self):
        # Encoder
        _x = Input(shape=tuple([self.original_dim]), name='encoder_input')
        _e_x = _x

        for units in self.hidden_units:
            _e_x = Dense(units, activation=self.hidden_activation)(_e_x)

        return Model(_x, _e_x, name='encoder')

    def build_and_compile_generator(self):
        """
        The generator of BiGAN is the generator of AAE.
        :return:
        """
        def build_encoder():
            # Encoder
            _input = Input(shape=tuple([self.original_dim]))
            _x = _input

            for units in self.hidden_units[:-1]:
                _x = Dense(units, activation=self.hidden_activation)(_x)

            mu = Dense(self.latent_dim)(_x)
            log_var = Dense(self.latent_dim)(_x)
            latent_repr = merge([mu, log_var],
                                mode=lambda p: p[0] + K.random_normal(K.shape(p[0])) * K.exp(p[1] / 2),
                                output_shape=lambda p: p[0])

            return Model(_input, latent_repr, name='encoder')

        def build_decoder():
            # Decoder
            _latent_input = Input(shape=(self.latent_dim,), name='z_sampling')
            _x = _latent_input

            for units in reversed(self.hidden_units[:-1]):
                _x = Dense(units, activation=self.hidden_activation)(_x)

            _output = Dense(self.original_dim, activation=self.output_activation)(_x)

            return Model(_latent_input, _output, name='decoder')

        def build_discriminator():
            # Discriminator
            model = Sequential()

            model.add(Dense(128, input_dim=self.latent_dim, activation=self.hidden_activation))
            model.add(LeakyReLU(alpha=0.2))
            model.add(Dense(64, activation=self.hidden_activation))
            model.add(LeakyReLU(alpha=0.2))
            model.add(Dense(32, activation=self.hidden_activation))
            model.add(LeakyReLU(alpha=0.2))
            model.add(Dense(16, activation=self.hidden_activation))
            model.add(LeakyReLU(alpha=0.2))
            model.add(Dense(8, activation=self.hidden_activation))
            model.add(LeakyReLU(alpha=0.2))
            model.add(Dense(1, activation=self.output_activation))
            model.summary()

            encoded_repr = Input(shape=(self.latent_dim, ))
            validity = model(encoded_repr)

            return Model(encoded_repr, validity, name='discriminator')

        # Generator is AAE
        # Build and compile the discriminator
        self.aae_discriminator = build_discriminator()
        self.aae_discriminator.compile(loss='binary_crossentropy',
                                       optimizer=self.optimizer,
                                       metrics=[self.metrics])

        # Build the encoder / decoder
        self.aae_encoder = build_encoder()
        self.aae_generator = build_decoder()

        # Save network's diagrams
        self.aae_discriminator.summary()
        plot_discriminator_filename = self.model_dir_path + 'aae_discriminator.pdf'
        plot_model(self.aae_discriminator, to_file=plot_discriminator_filename, show_shapes=True)

        self.aae_encoder.summary()
        plot_encoder_filename = self.model_dir_path + 'aae_encoder.pdf'
        plot_model(self.aae_encoder, to_file=plot_encoder_filename, show_shapes=True)

        self.aae_generator.summary()
        plot_decoder_filename = self.model_dir_path + 'aae_generator.pdf'
        plot_model(self.aae_generator, to_file=plot_decoder_filename, show_shapes=True)

        # this is our input placeholder
        _input = Input(shape=tuple([self.original_dim]), name='encoder_input')

        # The generator takes the image, encodes it and reconstructs it
        # from the encoding
        encoded_repr = self.aae_encoder(_input)
        reconstructed_img = self.aae_generator(encoded_repr)

        # For the adversarial_autoencoder model we will only train the generator
        self.aae_discriminator.trainable = False

        # The discriminator determines validity of the encoding
        validity = self.aae_discriminator(encoded_repr)

        # The adversarial_autoencoder model  (stacked generator and discriminator)
        self.aae = Model(_input, [reconstructed_img, validity])
        self.aae.compile(loss=['mse', 'binary_crossentropy'],
                         loss_weights=[0.999, 0.001],
                         optimizer=self.optimizer,
                         metrics=[self.metrics])

        self.aae.summary()
        plot_aae_filename = self.model_dir_path + 'aae.pdf'
        plot_model(self.aae, to_file=plot_aae_filename, show_shapes=True)

    def build_discriminator(self):
        # Discriminator
        _z = Input(shape=(self.latent_dim,), name='latent_data')
        _data = Input(shape=tuple([self.original_dim]), name='data')
        _d_in = concatenate([_z, _data])

        _d = _d_in

        #for units in self.hidden_units:
        #    _d = Dense(units, activation=self.hidden_activation)(_d)

        _v = Dense(1, activation=self.output_activation, name='validity')(_d)

        return Model([_z, _data], _v, name='discriminator')

    def _train_on_batch_generator(self, train_data_scaled, valid, fake, epoch):
        """
        Train the AAE model for the generator.
        :return:
        """
        for aae_epoch in range(self.aae_epochs):
            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of images
            idx = np.random.randint(0, train_data_scaled.shape[0], self.batch_size)
            inputs = train_data_scaled[idx]

            latent_fake = self.aae_encoder.predict(inputs)
            latent_real = np.random.normal(size=(self.batch_size, self.latent_dim))

            # Train the discriminator
            d_loss_real = self.aae_discriminator.train_on_batch(latent_real, valid)
            d_loss_fake = self.aae_discriminator.train_on_batch(latent_fake, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator
            g_loss = self.aae.train_on_batch(inputs, [inputs, valid])

            # Plot the progress
            print ("%d aae_generator: %d [D loss: %f, acc: %.2f%%] [G loss: %f, mse: %f]" %
                    (epoch, aae_epoch, d_loss[0], 100*d_loss[1], g_loss[0], g_loss[1]))

    def train(self, train_data,
              save_model = True,
              print_and_plot_history = False, show_plots = False):
        """
        Fit the autoencoder model.
        :param train_data:
        :return:
        """
        print('Training the autoencoder model:')
        # Scale the data
        scaler = MinMaxScaler()
        train_data_scaled = scaler.fit_transform(train_data)

        #==================================================================
        # Train the BiGAN model
        # Adversarial ground truths
        valid = np.ones((self.batch_size, 1))
        fake = np.zeros((self.batch_size, 1))

        all_d_loss = []
        all_d_acc = []
        all_g_loss = []

        for epoch in range(self.n_epochs):
            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Sample noise and generate img
            z = np.random.normal(size=(self.batch_size, self.latent_dim))
            gen_data = self.generator.predict(z)

            # Select a random batch of images and encode
            idx = np.random.randint(0, train_data_scaled.shape[0], self.batch_size)
            data = train_data_scaled[idx]
            enc_data = self.encoder.predict(data)

            # Train the discriminator (img -> z is valid, z -> img is fake)
            d_loss_real = self.discriminator.train_on_batch([enc_data, data], valid)
            d_loss_fake = self.discriminator.train_on_batch([z, gen_data], fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train AAE
            self._train_on_batch_generator(train_data_scaled, valid, fake, epoch)

            # Train the generator (z -> img is valid and img -> z is is invalid)
            g_loss = self.bigan_generator.train_on_batch([z, data], [valid, fake])

            # Plot the progress
            print ("%d [D loss: %f, acc: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss[0]))

            # Store the loss values for plot
            all_d_loss.append(d_loss[0])
            all_d_acc.append(100*d_loss[1])
            all_g_loss.append(g_loss[0])
        #==================================================================

        # evaluate the model
        self.mse_per_sample, self.global_mse = self._evaluate_model(train_data_scaled)

        if save_model:
            # Save the scaler
            scaler_filename = self.model_dir_path + 'scaler_' + self.model_name + '.pkl'
            joblib.dump(scaler, scaler_filename)

            # Save weights to HDF5
            weights_filename = self.model_dir_path + self.model_name + '.h5'

            # Save weights
            self.bigan_generator.save_weights(weights_filename)

            print('Saved generator to disk.')

            # Save discriminator
            weights_filename_d = self.model_dir_path + self.discriminator_name + '.h5'

            self.discriminator.save_weights(weights_filename_d)

            print('Saved discriminator to disk.')

        if print_and_plot_history:
            _x = range(self.n_epochs)
            # summarize loss
            fig1 = plt.figure()
            plt.plot(_x, all_d_loss, 'b-', _x, all_g_loss, 'g--')
            plt.title('BiGAN learning loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['discriminator', 'generator'], loc='upper left')
            figure_name = self.model_dir_path + self.model_name + '_loss.pdf'
            fig1.savefig(figure_name)
            if show_plots:
                plt.show()
            # summarize discriminator accuracy
            fig2 = plt.figure()
            plt.plot(_x, all_d_acc, 'k-')
            plt.title('BiGAN learning discriminator accuracy')
            plt.ylabel('acc')
            plt.xlabel('epoch')
            figure_name = self.model_dir_path + self.model_name + '_d_acc.pdf'
            fig2.savefig(figure_name)
            if show_plots:
                plt.show()

    def _evaluate_model(self, scaled_data):
        """
        Evaluates the trained model with the scaled data and determines the score.
        :param scaled_data:
        :return:
        """
        _e_data = self.encoder.predict(scaled_data)
        _p_data = self.discriminator.predict([_e_data, scaled_data])
        if self.version == 0:
            mse_per_sample = [mean_squared_error([_p], [1]) for _p in _p_data.flatten()]
        else:
            # Sample noise and generate img
            _gen_data = self.generator.predict(self.random_z_samples)
            _p_gen = self.discriminator.predict([self.random_z_samples, _gen_data])
            mse_per_sample = [mean_squared_error(_p * np.ones(_p_gen.shape), _p_gen) for _p in _p_data.flatten()]

        assert len(mse_per_sample) == scaled_data.shape[0], ("The predicted data shape is not right!")

        global_mse = np.mean(mse_per_sample)

        return mse_per_sample, global_mse

    def load_weights(self):
        """
        Load the weights from file to the model.
        :param model_dir_path:
        :param iteration_number:
        :return:
        """
        weights_filename = self.model_dir_path + self.model_name + '.h5'
        weights_filename_d = self.model_dir_path + self.discriminator_name + '.h5'

        # load weights into new model
        self.bigan_generator.load_weights(weights_filename)
        self.discriminator.load_weights(weights_filename_d)
        print("Loaded weights from disk")

    def test_model(self,
                   test_data):
        """
        Apply the learned model to the test samples and get scores.
        :param test_data:
        :param model_dir_path:
        :param iteration_number:
        :return:
        """
        # Get the scaler
        scaler_filename = self.model_dir_path + 'scaler_' + self.model_name + '.pkl'
        scaler = joblib.load(scaler_filename)

        # Scale the test data
        test_data_scaled = scaler.transform(test_data)

        # evaluate the model
        mse_per_sample, global_mse = self._evaluate_model(test_data_scaled)

        return global_mse, mse_per_sample

class BuildSimpleWAE():
    """
    An simple autoencoder using Keras.
    Inspired by the following:
    - https://blog.keras.io/building-autoencoders-in-keras.html;
    - https://github.com/skolouri/swae/blob/master/MNIST_SlicedWassersteinAutoEncoder_uniform.ipynb
    """

    def __init__(self,
                 original_dim=125,
                 batch_size=500,
                 n_epochs=20000,
                 metrics='accuracy',
                 model_dir_path = ''):
        """
        Builds and compiles an autoencoder.
        :param input_size:
        :param hidden_units:
        :param batch_size:
        :param n_epochs:
        :param learning_rate:
        :param hidden_activation:
        :param output_activation:
        :param optimiser:
        :param loss:
        :param shuffle_data:
        """
        self.original_dim = original_dim
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.metrics = metrics
        self.model_dir_path = model_dir_path

        self.optimizer = Adam(0.0002, 0.5)
        self.inter_dim = 128   # This is the dimension of intermediate latent variable
                               #(after convolution and before embedding)
        self.embedding_dim = 2 # Dimension of the embedding space
        self.depth = 16        # This is a design parameter and in fact it is not the depth!
        self.L = 50            # Number of random projections

        # This factor increases the input size and makes it possible for a square image transformation layer
        self.input_scaling_factor = 5
        self.input_2d_size = int(np.sqrt(self.original_dim * self.input_scaling_factor))

        # this is our input placeholder
        self.input = Input(shape=tuple([self.original_dim]), name='input_data')

        # Build the encoder / decoder
        self.build_encoder()
        self.build_decoder()

        theta=K.variable(self.generateTheta()) #Define a Keras Variable for \theta_ls
        z=K.variable(self.generateZ())         #Define a Keras Variable for samples of z

        # Generate the autoencoder by combining encoder and decoder
        aencoded=self.encoder(self.input)
        ae=self.decoder(aencoded)
        self.autoencoder=Model(inputs=[self.input],outputs=[ae])
        self.autoencoder.summary()

        # Let projae be the projection of the encoded samples
        projae=K.dot(aencoded,K.transpose(theta))
        # Let projz be the projection of the $q_Z$ samples
        projz=K.dot(z,K.transpose(theta))
        # Calculate the Sliced Wasserstein distance by sorting
        # the projections and calculating the L2 distance between
        W2=(tf.nn.top_k(tf.transpose(projae),k=self.batch_size).values-
            tf.nn.top_k(tf.transpose(projz),k=self.batch_size).values)**2

        self.w2weight=K.variable(10.0)
        crossEntropyLoss= (1.0)*K.mean(K.binary_crossentropy(self.input,ae))
        L1Loss= (1.0)*K.mean(K.abs(self.input-ae))
        W2Loss= self.w2weight*K.mean(W2)
        # I have a combination of L1 and Cross-Entropy loss for the first term and then
        # W2 for the second term
        vae_Loss=L1Loss+crossEntropyLoss+W2Loss
        self.autoencoder.add_loss(vae_Loss) # Add the custom loss to the model

        #Compile the model
        self.autoencoder.compile(optimizer='rmsprop',loss='')

    def build_encoder(self):
        # Encoder
        # Reshape the input to 28x28x1 MNIST image shape
        x=UpSampling1D(self.input_scaling_factor)(self.input)
        x=Reshape((self.input_2d_size,self.input_2d_size,1))(x)
        x=Conv2D(self.depth*1, (3, 3), padding='same')(x)
        x=LeakyReLU(alpha=0.2)(x)
        # x=BatchNormalization(momentum=0.8)(x)
        x=Conv2D(self.depth*1, (3, 3), padding='same')(x)
        x=LeakyReLU(alpha=0.2)(x)
        # x=BatchNormalization(momentum=0.8)(x)
        x=AveragePooling2D((2, 2), padding='same')(x)
        x=Conv2D(self.depth*2, (3, 3), padding='same')(x)
        x=LeakyReLU(alpha=0.2)(x)
        # x=BatchNormalization(momentum=0.8)(x)
        x=Conv2D(self.depth*2, (3, 3), padding='same')(x)
        x=LeakyReLU(alpha=0.2)(x)
        # x=BatchNormalization(momentum=0.8)(x)
        x=AveragePooling2D((2, 2), padding='same')(x)
        x=Conv2D(self.depth*4, (3, 3), padding='same')(x)
        x=LeakyReLU(alpha=0.2)(x)
        # x=BatchNormalization(momentum=0.8)(x)
        x=Conv2D(self.depth*4, (3, 3), padding='same')(x)
        x=LeakyReLU(alpha=0.2)(x)
        # x=BatchNormalization(momentum=0.8)(x)
        x=AveragePooling2D((2, 2), padding='same')(x)
        x=Flatten()(x)
        x=Dense(self.inter_dim,activation='relu')(x)
        encoded=Dense(self.embedding_dim)(x)

        self.encoder=Model(inputs=[self.input],outputs=[encoded])
        self.encoder.summary()

    def build_decoder(self):
        # Decoder
        x=Dense(self.inter_dim)(self.embedding_dim)
        x=Dense(self.depth*64,activation='relu')(x)
        # x=BatchNormalization(momentum=0.8)(x)
        x=Reshape((4,4,4*self.depth))(x)
        x=UpSampling2D((2, 2))(x)
        x=Conv2D(self.depth*4, (3, 3), padding='same')(x)
        x=LeakyReLU(alpha=0.2)(x)
        # x=BatchNormalization(momentum=0.8)(x)
        x=Conv2D(self.depth*4, (3, 3), padding='same')(x)
        x=LeakyReLU(alpha=0.2)(x)
        x=UpSampling2D((2, 2))(x)
        x=Conv2D(self.depth*4, (3, 3), padding='valid')(x)
        x=LeakyReLU(alpha=0.2)(x)
        # x=BatchNormalization(momentum=0.8)(x)
        x=Conv2D(self.depth*4, (3, 3), padding='same')(x)
        x=LeakyReLU(alpha=0.2)(x)
        x=UpSampling2D((2, 2))(x)
        x=Conv2D(self.depth*2, (3, 3), padding='same')(x)
        x=LeakyReLU(alpha=0.2)(x)
        # x=BatchNormalization(momentum=0.8)(x)
        x=Conv2D(self.depth*2, (3, 3), padding='same')(x)
        x=LeakyReLU(alpha=0.2)(x)
        # x=BatchNormalization(momentum=0.8)(x)
        # x=BatchNormalization(momentum=0.8)(x)
        x=Conv2D(1, (3, 3), padding='same')(x)
        # Reshape to the original input shape
        x=Flatten()(x)
        decoded=Dense(self.original_dim, activation='sigmoid')(x)

        self.decoder=Model(inputs=[self.embedding_dim],outputs=[decoded])
        self.decoder.summary()

    def generateTheta(self):
        # This function generates L random samples from the unit `ndim'-u
        theta=[w/np.sqrt((w**2).sum()) for w in np.random.normal(size=(self.L,self.embedding_dim))]
        return np.asarray(theta)

    def generateZ(self):
        # This function generates samples from a uniform distribution in
        # the `endim'-dimensional space
        z=2*(np.random.uniform(size=(self.batch_size,self.embedding_dim))-0.5)
        return z

    def train(self, train_data,
              save_model = True,
              iteration_number = 0,
              print_and_plot_history = False, show_plots = False):
        """
        Fit the autoencoder model.
        :param train_data:
        :return:
        """
        print('Training the autoencoder model:')
        # Scale the data
        scaler = MinMaxScaler()
        train_data_scaled = scaler.fit_transform(train_data)

        #==================================================================
        # Train the WAE model
        # Adversarial ground truths
        valid = np.ones((self.batch_size, 1))
        fake = np.zeros((self.batch_size, 1))

        all_d_loss = []
        all_d_acc = []
        all_g_loss = []
        all_g_mse = []

        for epoch in range(self.n_epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of images
            idx = np.random.randint(0, train_data_scaled.shape[0], self.batch_size)
            inputs = train_data_scaled[idx]
        #==================================================================

        # evaluate the model
        predicted_train_data = self.decoder.predict(self.encoder.predict(train_data_scaled))
        assert predicted_train_data.shape == train_data_scaled.shape, ("The predicted data shape is not right!")
        self.mse_per_sample = [mean_squared_error(train_data_scaled[i,:], predicted_train_data[i,:])
                               for i in range(train_data_scaled.shape[0])]

        self.global_mse = np.mean(self.mse_per_sample)

        model_name = 'model_' + str(iteration_number)
        discriminator_name = 'model_d_' + str(iteration_number)

        if save_model:
            # Save the scaler
            scaler_filename = self.model_dir_path + 'scaler_' + model_name + '.pkl'
            joblib.dump(scaler, scaler_filename)

            # Save weights to HDF5
            weights_filename = self.model_dir_path + model_name + '.h5'

            # Save weights
            self.aae.save_weights(weights_filename)

            print('Saved generator to disk.')

            # Save discriminator
            weights_filename_d = self.model_dir_path + discriminator_name + '.h5'

            print('Saved discriminator to disk.')

        if print_and_plot_history:
            _x = range(self.n_epochs)
            # summarize loss
            fig1 = plt.figure()
            plt.plot(_x, all_d_loss, 'b-', _x, all_g_loss, 'g--')
            plt.title('AAE learning loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['discriminator', 'generator'], loc='upper left')
            figure_name = self.model_dir_path + model_name + '_loss.pdf'
            fig1.savefig(figure_name)
            if show_plots:
                plt.show()
            # summarize discriminator accuracy
            fig2 = plt.figure()
            plt.plot(_x, all_d_acc, 'k-')
            plt.title('AAE learning discriminator accuracy')
            plt.ylabel('acc')
            plt.xlabel('epoch')
            figure_name = self.model_dir_path + model_name + '_d_acc.pdf'
            fig2.savefig(figure_name)
            if show_plots:
                plt.show()
            # summarize generator error
            fig3 = plt.figure()
            plt.plot(_x, all_g_mse, 'k-')
            plt.title('AAE learning generator error')
            plt.ylabel('mse')
            plt.xlabel('epoch')
            figure_name = self.model_dir_path + model_name + '_g_mse.pdf'
            fig3.savefig(figure_name)
            if show_plots:
                plt.show()

    def load_weights(self,
                     iteration_number = 0):
        """
        Load the weights from file to the model.
        :param model_dir_path:
        :param iteration_number:
        :return:
        """
        model_name = 'model_' + str(iteration_number)
        discriminator_name = 'model_d_' + str(iteration_number)
        weights_filename = self.model_dir_path + model_name + '.h5'
        weights_filename_d = self.model_dir_path + discriminator_name + '.h5'

        print("Loaded weights from disk")

    def test_model(self,
                   test_data,
                   iteration_number = 0):
        """
        Apply the learned model to the test samples and get scores.
        :param test_data:
        :param model_dir_path:
        :param iteration_number:
        :return:
        """
        model_name = 'model_' + str(iteration_number)
        # Get the scaler
        scaler_filename = self.model_dir_path + 'scaler_' + model_name + '.pkl'
        scaler = joblib.load(scaler_filename)

        # Scale the test data
        test_data_scaled = scaler.transform(test_data)

        # extract results
        predicted_test_data = self.decoder.predict(self.encoder.predict(test_data_scaled))
        assert predicted_test_data.shape == test_data_scaled.shape, ("The predicted data shape is not right!")
        mse_per_sample = [mean_squared_error(test_data_scaled[i,:], predicted_test_data[i,:])
                          for i in range(test_data_scaled.shape[0])]
        global_mse = np.mean(mse_per_sample)

        return global_mse, mse_per_sample

def abs_diff(X):
    s = X[0]
    for i in range(1, len(X)):
        s -= X[i]
    s = K.abs(s)
    return s

def diff_times_ten(X):
    s = X[0]
    for i in range(1, len(X)):
        s -= X[i]
    s *= 10
    return s

def consecutive_mse(X):
    s = X[0]
    for i in range(1, len(X)):
        s -= X[i]
    s **= 2
    return K.mean(K.reshape(s, (-1, 5, 25)), axis=2)

def decrease_size_by_averaging(X):
    return K.mean(K.reshape(X, (-1, 25, 5)), axis=2)

def check_if_slightly_increasing(X, decreasing = False):
    if decreasing:
        return (np.count_nonzero(np.diff(X) <= 0) / float(len(X))) >= losses_tolerance
    else:
        return (np.count_nonzero(np.diff(X) >= 0) / float(len(X))) >= losses_tolerance

class BuildOurMethodV1():
    """
    First version of our method: Implement of generator using pre-trained DAE.
    Inspired by the following:
    - https://blog.keras.io/building-autoencoders-in-keras.html;
    - https://github.com/civisanalytics/muffnn/blob/master/muffnn/autoencoder/autoencoder.py
    """

    def __init__(self,
                 original_dim=125,
                 hidden_units=(128,64,32,16,8),
                 batch_size=128,
                 n_epochs=20000,
                 hidden_activation='relu',
                 output_activation='sigmoid',
                 loss='mean_squared_error',
                 metrics='accuracy',
                 model_dir_path = '',
                 iteration_number = 0):
        """
        Builds and compiles an autoencoder.
        :param input_size:
        :param hidden_units:
        :param batch_size:
        :param n_epochs:
        :param learning_rate:
        :param hidden_activation:
        :param output_activation:
        :param optimiser:
        :param loss:
        :param shuffle_data:
        """
        self.original_dim = original_dim
        self.hidden_units = hidden_units
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.optimizer_a = 'rmsprop'
        self.optimizer = RMSprop(lr=0.00005)
        self.loss = loss
        self.metrics = metrics
        self.latent_dim = hidden_units[-1]
        self.model_dir_path = model_dir_path
        self.iteration_number = iteration_number

        self.model_name = 'model_' + str(self.iteration_number)
        self.model_d_name = 'model_d_' + str(self.iteration_number)
        self.model_ae_name = 'model_ae_' + str(self.iteration_number)

        # Build and compile the regular deep autoencoder separately
        self.pt_dae = self.build_dae()
        self.pt_dae.compile(optimizer=self.optimizer_a, loss=self.loss, metrics=[self.metrics])

        # Optimizer will be different for GAN
        #self.optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=self.loss,
            optimizer=self.optimizer,
            metrics=[self.metrics])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False
        self.pt_dae.trainable = False

        # The discriminator takes generated images as input and determines validity
        rep_img = self.pt_dae(img)
        diff = Lambda(abs_diff, output_shape=(self.original_dim,), name='abs_diff')([rep_img, img])
        validity = self.discriminator(diff)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, validity)
        self.combined.compile(loss=self.loss, optimizer=self.optimizer)

    def build_generator(self):
        # Decoder
        _latent_input = Input(shape=(self.latent_dim,), name='z_sampling')
        _x = _latent_input

        for units in reversed(self.hidden_units[:-1]):
            _x = Dense(units, activation=self.hidden_activation)(_x)

        _output = Dense(self.original_dim, activation=self.output_activation)(_x)

        return Model(_latent_input, _output, name='generator')

    def build_discriminator(self):
        # Discriminator
        _input = Input(shape=tuple([self.original_dim]), name='input_sampling')
        _x = _input

        for units in self.hidden_units:
            _x = Dense(units, activation=self.hidden_activation)(_x)

        _validity = Dense(1, activation=self.output_activation)(_x)

        return Model(_input, _validity, name='discriminator')

    def build_dae(self):
        # Deep Regular Autoencoder
        # this is our input placeholder
        _input = Input(shape=tuple([self.original_dim]))
        _encoded = _input

        for units in self.hidden_units:
            _encoded = Dense(units, activation=self.hidden_activation)(_encoded)

        _decoded = _encoded

        for units in reversed(self.hidden_units[:-1]):
            _decoded = Dense(units, activation=self.hidden_activation)(_decoded)

        _decoded = Dense(self.original_dim, activation=self.output_activation)(_decoded)

        return Model(_input, _decoded, name='dae')

    def _train_on_batch_discriminator(self, scaled_data, label):
        # Train discriminator using pre-trained deep autoencoder
        predicted_data = self.pt_dae.predict(scaled_data)

        diff_data = np.abs(predicted_data - scaled_data)

        return self.discriminator.train_on_batch(diff_data, label)

    def _discriminator_predict(self, scaled_data):
        predicted_data = self.pt_dae.predict(scaled_data)

        diff_data = np.abs(predicted_data - scaled_data)

        return self.discriminator.predict(diff_data)

    def _discriminator_evaluate(self, scaled_data, labels):
        predicted_data = self.pt_dae.predict(scaled_data)

        diff_data = np.abs(predicted_data - scaled_data)

        return self.discriminator.evaluate(diff_data, labels)

    def train(self, train_data,
              save_model = True,
              print_and_plot_history = False, show_plots = False):
        """
        Fit the autoencoder model.
        :param train_data:
        :return:
        """
        print('Training GAN using pre-trained deep autoencoder:')

        # Scale the data
        scaler = MinMaxScaler(feature_range=(-1,1))
        train_data_scaled = scaler.fit_transform(train_data)

        # First, train the deep regular autoencoder which will be used as a pre-trained
        # network for training discriminator
        print('--------------------------------------------------------')
        print('Training the regular deep autoencoder...')
        pt_dae_epochs = 100
        pt_dae_validation_split=0.20
        pt_dae_shuffle_data = True
        self.history = self.pt_dae.fit(train_data_scaled, train_data_scaled,
                                       validation_split=pt_dae_validation_split,
                                       epochs=pt_dae_epochs,
                                       batch_size=self.batch_size,
                                       shuffle=pt_dae_shuffle_data)
        # evaluate the model
        self.ae_score = self.pt_dae.evaluate(train_data_scaled, train_data_scaled)
        predicted_train_data = self.pt_dae.predict(train_data_scaled)
        assert predicted_train_data.shape == train_data_scaled.shape, ("The predicted data shape is not right!")

        # list all data in history
        print(self.history.history.keys())
        print('--------------------------------------------------------')

        #==================================================================
        # Train the GAN model
        # Adversarial ground truths
        valid = np.ones((self.batch_size, 1))
        fake = np.zeros((self.batch_size, 1))

        all_a_loss = []
        all_a_acc = []
        all_d_loss = []
        all_d_acc = []
        all_g_loss = []

        last_ten_d_acc = np.ones(losses_to_check_size)
        last_ten_g_losses = np.ones(losses_to_check_size)
        max_a_acc_criteria = 0.6
        min_d_acc_criteria = 0.9
        max_d_acc_criteria = 0.99
        min_g_loss_criteria = 0.5
        max_g_loss_criteria = 0.9
        g_loss_criteria_increment_step = 0.1
        n_step_min_rep= 10
        n_step_counter = 0
        last_g_loss = 1
        train_dae = True
        help_discriminator = False
        help_generator = False

        print('--------------------------------------------------------')
        print('Training GAN...')
        for epoch in range(self.n_epochs):

            # ---------------------
            #  Train DAE
            # ---------------------

            # Select a random batch of images
            idx = np.random.randint(0, train_data_scaled.shape[0], self.batch_size)
            inputs = train_data_scaled[idx]

            if train_dae:
                a_loss = self.pt_dae.train_on_batch(inputs, inputs)

                if a_loss[1] > max_a_acc_criteria:
                    train_dae = False

            if not help_generator:
                # ---------------------
                #  Train Discriminator
                # ---------------------

                noise = np.random.normal(0, 1, (self.batch_size, self.latent_dim))

                # Generate a batch of new images
                gen_data = self.generator.predict(noise)

                # Train the discriminator
                d_loss_real = self._train_on_batch_discriminator(inputs, valid)
                d_loss_fake = self._train_on_batch_discriminator(gen_data, fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # Store
                last_ten_d_acc = np.delete(np.append(last_ten_d_acc, d_loss[1]), 0)

            if help_discriminator:
                #last_ten_g_losses = np.delete(np.append(last_ten_g_losses, [1]), 0)
                if np.all(last_ten_d_acc >= max_d_acc_criteria):
                    help_discriminator = False
                # Store the loss values for plot
                all_a_loss.append(a_loss[0])
                all_a_acc.append(100*a_loss[1])
                all_d_loss.append(d_loss[0])
                all_d_acc.append(100*d_loss[1])
                all_g_loss.append(last_g_loss)
                # Plot the progress
                print ("%d [DAE loss: %f, acc.: %.2f%%] [D loss: %f, acc.: %.2f%%] [G loss: %f]" %
                       (epoch, a_loss[0], 100*a_loss[1], d_loss[0], 100*d_loss[1], last_g_loss))
                continue

            # ---------------------
            #  Train Generator
            # ---------------------

            noise = np.random.normal(0, 1, (self.batch_size, self.latent_dim))

            # Train the generator (to have the discriminator label samples as valid)
            g_loss = self.combined.train_on_batch(noise, valid)

            # Store
            last_ten_g_losses = np.delete(np.append(last_ten_g_losses, g_loss), 0)

            if help_generator:
                if check_if_slightly_increasing(last_ten_g_losses, decreasing=True) and g_loss < min_g_loss_criteria:
                    help_generator = False
                    n_step_counter += 1
                    if min_g_loss_criteria < (max_g_loss_criteria - 0.01) and n_step_counter >= n_step_min_rep:
                        min_g_loss_criteria += g_loss_criteria_increment_step
                        n_step_counter = 0
                # Store the loss values for plot
                all_a_loss.append(a_loss[0])
                all_a_acc.append(100*a_loss[1])
                all_d_loss.append(d_loss[0])
                all_d_acc.append(100*d_loss[1])
                all_g_loss.append(g_loss)
                # Plot the progress
                print ("%d [DAE loss: %f, acc.: %.2f%%] [D loss: %f, acc.: %.2f%%] [G loss: %f]" %
                       (epoch, a_loss[0], 100*a_loss[1], d_loss[0], 100*d_loss[1], g_loss))
                continue

            if not help_discriminator and epoch > losses_to_check_size:
                if np.all(last_ten_d_acc < min_d_acc_criteria) or d_loss[0] > g_loss:
                    last_g_loss = g_loss
                    help_discriminator = True
                    help_generator = False          # Ensure that these two are not true at the same time
                elif not help_generator:
                    if g_loss > min_g_loss_criteria:
                        help_generator = True
                        help_discriminator = False  # Ensure that these two are not true at the same time

            # Store the loss values for plot
            all_a_loss.append(a_loss[0])
            all_a_acc.append(100*a_loss[1])
            all_d_loss.append(d_loss[0])
            all_d_acc.append(100*d_loss[1])
            all_g_loss.append(g_loss)

            # Plot the progress
            print ("%d [DAE loss: %f, acc.: %.2f%%] [D loss: %f, acc.: %.2f%%] [G loss: %f]" %
                   (epoch, a_loss[0], 100*a_loss[1], d_loss[0], 100*d_loss[1], g_loss))
        print('--------------------------------------------------------')
        #==================================================================

        # Test the discriminator
        p_train_data = self._discriminator_predict(train_data_scaled)
        assert p_train_data.shape[0] == train_data_scaled.shape[0], ("The predicted data shape is not right!")

        # Test the generator
        noise = np.random.normal(0, 1, (self.batch_size, self.latent_dim))
        g_data = self.generator.predict(noise)
        assert g_data.shape == (self.batch_size, self.original_dim), ("The predicted data shape is not right!")
        p_g_data = self._discriminator_predict(g_data)
        assert p_g_data.shape[0] == g_data.shape[0], ("The predicted data shape is not right!")

        if save_model:
            # Save the scaler
            scaler_filename = self.model_dir_path + 'scaler_' + self.model_name + '.pkl'
            joblib.dump(scaler, scaler_filename)

            # Save weights to HDF5
            model_weights_filename = self.model_dir_path + self.model_name + '.h5'
            model_d_weights_filename = self.model_dir_path + self.model_d_name + '.h5'
            model_ae_weights_filename = self.model_dir_path + self.model_ae_name + '.h5'

            # Save weights
            self.combined.save_weights(model_weights_filename)
            self.discriminator.save_weights(model_d_weights_filename)
            self.pt_dae.save_weights(model_ae_weights_filename)

            print('Saved model to disk.')

        if print_and_plot_history:
            _x = range(self.n_epochs)
            # summarize loss
            fig1 = plt.figure()
            plt.plot(_x, all_a_loss, 'k-', _x, all_d_loss, 'b--', _x, all_g_loss, 'g.-')
            plt.title('Model learning loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['autoencoder', 'discriminator', 'generator'], loc='upper left')
            figure_name = self.model_dir_path + self.model_name + '_loss.pdf'
            fig1.savefig(figure_name)
            if show_plots:
                plt.show()
            # summarize discriminator accuracy
            fig2 = plt.figure()
            plt.plot(_x, all_a_acc, 'k-', _x, all_d_acc, 'b--')
            plt.title('Model learning accuracy')
            plt.ylabel('acc')
            plt.xlabel('epoch')
            plt.legend(['autoencoder', 'discriminator'], loc='upper left')
            figure_name = self.model_dir_path + self.model_name + '_acc.pdf'
            fig2.savefig(figure_name)
            if show_plots:
                plt.show()

    def load_weights(self):
        """
        Load the weights from file to the model.
        :param model_dir_path:
        :param iteration_number:
        :return:
        """
        model_weights_filename = self.model_dir_path + self.model_name + '.h5'
        model_d_weights_filename = self.model_dir_path + self.model_d_name + '.h5'
        model_ae_weights_filename = self.model_dir_path + self.model_ae_name + '.h5'

        # load weights into new model
        self.pt_dae.load_weights(model_ae_weights_filename)
        self.discriminator.load_weights(model_d_weights_filename)
        self.combined.load_weights(model_weights_filename)
        print("Loaded weights from disk")

    def test_model(self,
                   test_data,
                   is_abnormal = False):
        """
        Apply the learned model to the test samples and get scores.
        :param test_data:
        :param model_dir_path:
        :param iteration_number:
        :return:
        """
        # Get the scaler
        scaler_filename = self.model_dir_path + 'scaler_' + self.model_name + '.pkl'
        scaler = joblib.load(scaler_filename)

        # Scale the test data
        test_data_scaled = scaler.transform(test_data)

        # evaluate
        if is_abnormal:
            labels = np.zeros((test_data_scaled.shape[0], 1))
        else:
            labels = np.ones((test_data_scaled.shape[0], 1))

        test_score = self._discriminator_evaluate(test_data_scaled, labels)

        return test_score[0], test_score[1]

class BuildOurMethodV2():
    """
    First version of our method: Implement of generator using pre-trained DAE.
    Inspired by the following:
    - https://blog.keras.io/building-autoencoders-in-keras.html;
    - https://github.com/civisanalytics/muffnn/blob/master/muffnn/autoencoder/autoencoder.py
    """

    def __init__(self,
                 original_dim=125,
                 hidden_units=(128,64,32,16,8),
                 batch_size=128,
                 n_epochs=20000,
                 hidden_activation='relu',
                 output_activation='sigmoid',
                 loss='mean_squared_error',
                 metrics='accuracy',
                 model_dir_path = '',
                 iteration_number = 0):
        """
        Builds and compiles an autoencoder.
        :param input_size:
        :param hidden_units:
        :param batch_size:
        :param n_epochs:
        :param learning_rate:
        :param hidden_activation:
        :param output_activation:
        :param optimiser:
        :param loss:
        :param shuffle_data:
        """
        self.original_dim = original_dim
        self.hidden_units = hidden_units
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.optimizer_a = 'rmsprop'
        self.optimizer = RMSprop(lr=0.00005)
        self.loss = loss
        self.metrics = metrics
        self.latent_dim = hidden_units[-1]
        self.model_dir_path = model_dir_path
        self.iteration_number = iteration_number

        self.model_name = 'model_' + str(self.iteration_number)
        self.model_d_name = 'model_d_' + str(self.iteration_number)
        self.model_ae_name = 'model_ae_' + str(self.iteration_number)

        # Build encoder and decoder
        self.encoder = self.build_encoder()
        self.decoder = self.build_generator()

        # Build and compile the regular deep autoencoder separately
        input_data = Input(shape=tuple([self.original_dim]))
        r_input_data = self.decoder(self.encoder(input_data))
        self.autoencoder = Model(input_data, r_input_data, name='autoencoder')
        self.autoencoder.compile(optimizer=self.optimizer_a, loss=self.loss, metrics=[self.metrics])

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=self.loss,
            optimizer=self.optimizer,
            metrics=[self.metrics])

        # Build generator of GAN
        self.generator = self.build_generator()

        # The generator takes noise as input and generates data
        z = Input(shape=(self.latent_dim,))
        gen_data = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False
        self.autoencoder.trainable = False

        # The discriminator takes generated images as input and determines validity
        rep_data = self.autoencoder(gen_data)
        diff = Lambda(diff_times_ten, output_shape=(self.original_dim,), name='diff_times_ten')([rep_data, gen_data])
        validity = self.discriminator(diff)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, validity)
        self.combined.compile(loss=self.loss, optimizer=self.optimizer)

    def build_encoder(self):
        # Encoder
        _input = Input(shape=tuple([self.original_dim]))
        _encoded = _input

        for units in self.hidden_units:
            _encoded = Dense(units, activation=self.hidden_activation)(_encoded)

        return Model(_input, _encoded, name='encoder')

    def build_generator(self):
        # Decoder
        _latent_input = Input(shape=(self.latent_dim,), name='z_sampling')
        _x = _latent_input

        for units in reversed(self.hidden_units[:-1]):
            _x = Dense(units, activation=self.hidden_activation)(_x)

        _output = Dense(self.original_dim, activation=self.output_activation)(_x)

        return Model(_latent_input, _output, name='generator')

    def build_discriminator(self):
        # Discriminator
        _input = Input(shape=tuple([self.original_dim]), name='input_sampling')
        _x = _input

        for units in self.hidden_units:
            _x = Dense(units, activation=self.hidden_activation)(_x)

        _validity = Dense(1, activation=self.output_activation)(_x)

        return Model(_input, _validity, name='discriminator')

    def _train_on_batch_discriminator(self, scaled_data, label):
        # Train discriminator using pre-trained deep autoencoder
        predicted_data = self.autoencoder.predict(scaled_data)

        diff_data = 10*(predicted_data - scaled_data)

        return self.discriminator.train_on_batch(diff_data, label)

    def _discriminator_predict(self, scaled_data):
        predicted_data = self.autoencoder.predict(scaled_data)

        diff_data = 10*(predicted_data - scaled_data)

        return self.discriminator.predict(diff_data)

    def _discriminator_evaluate(self, scaled_data, labels):
        predicted_data = self.autoencoder.predict(scaled_data)

        diff_data = 10*(predicted_data - scaled_data)

        return self.discriminator.evaluate(diff_data, labels)

    def _generate_fake_and_or_normal_data(self, scaled_data):
        gen_n_data = None

        n_times_batch_size = 10

        noise = np.random.normal(0, 1, (n_times_batch_size * self.batch_size, self.latent_dim))

        # Select a random batch of images
        idx = np.random.randint(0, scaled_data.shape[0], n_times_batch_size * self.batch_size)
        inputs = scaled_data[idx]

        # Get the encoded data
        encoded_data = self.encoder.predict(inputs)

        # Generate a batch of new data
        gen_data = self.generator.predict(noise + encoded_data)

        # Reconstruct the generated data
        r_gen_data = self.autoencoder.predict(gen_data)

        # Get the mse for each sample
        mse_per_sample = [mean_squared_error(gen_data[i,:], r_gen_data[i,:])
                          for i in range(gen_data.shape[0])]

        # Get the label
        gen_label = np.array(mse_per_sample) <= self.ae_mse

        gen_n_data_size = np.sum(gen_label)
        gen_a_data_size = np.sum(gen_label == 0)

        if gen_n_data_size >= self.batch_size:
            gen_n_data = gen_data[np.where(gen_label == 1)[0],:]

            # Select a random batch of generated data
            idx = np.random.randint(0, gen_n_data.shape[0], self.batch_size)
            gen_n_data = gen_n_data[idx]

        if gen_a_data_size >= self.batch_size:
            gen_a_data = gen_data[np.where(gen_label == 0)[0],:]

            # Select a random batch of generated data
            idx = np.random.randint(0, gen_a_data.shape[0], self.batch_size)
            gen_a_data = gen_a_data[idx]
        else:
            gen_a_data = np.empty(shape=(0, self.original_dim))
            while True:
                noise = np.random.normal(0, 1, (self.batch_size, self.latent_dim))
                gen_data = self.generator.predict(noise)
                r_gen_data = self.autoencoder.predict(gen_data)

                mse_per_sample = [mean_squared_error(gen_data[i,:], r_gen_data[i,:])
                                  for i in range(gen_data.shape[0])]
                gen_label = np.array(mse_per_sample) <= self.ae_mse

                gen_a_data_size = np.sum(gen_label == 0)

                if gen_a_data_size == self.batch_size:
                    gen_a_data = gen_data
                    break
                else:
                    sub_gen_a_data = gen_data[np.where(gen_label == 0)[0],:]
                    gen_a_data = np.append(gen_a_data, sub_gen_a_data, axis=0)

                    if gen_a_data.shape[0] >= self.batch_size:
                        # Select a random batch of generated data
                        idx = np.random.randint(0, gen_a_data.shape[0], self.batch_size)
                        gen_a_data = gen_a_data[idx]
                        break

        return gen_a_data, gen_n_data

    def train(self, train_data,
              save_model = True,
              print_and_plot_history = False, show_plots = False):
        """
        Fit the autoencoder model.
        :param train_data:
        :return:
        """
        print('Training GAN using pre-trained deep autoencoder:')

        # Scale the data
        scaler = MinMaxScaler(feature_range=(-1,1))
        train_data_scaled = scaler.fit_transform(train_data)

        # First, train the deep regular autoencoder which will be used as a pre-trained
        # network for training discriminator
        print('--------------------------------------------------------')
        print('Training the regular deep autoencoder...')
        pt_dae_epochs = 100
        pt_dae_validation_split=0.20
        pt_dae_shuffle_data = True
        self.history = self.autoencoder.fit(train_data_scaled, train_data_scaled,
                                            validation_split=pt_dae_validation_split,
                                            epochs=pt_dae_epochs,
                                            batch_size=self.batch_size,
                                            shuffle=pt_dae_shuffle_data)
        # evaluate the model
        self.ae_score = self.autoencoder.evaluate(train_data_scaled, train_data_scaled)
        self.ae_mse = self.ae_score[0]
        predicted_train_data = self.autoencoder.predict(train_data_scaled)
        assert predicted_train_data.shape == train_data_scaled.shape, ("The predicted data shape is not right!")

        # Get the mse for each sample
        #self.ae_mse_per_sample = [mean_squared_error(train_data_scaled[i,:], predicted_train_data[i,:])
        #                          for i in range(train_data_scaled.shape[0])]
        #ae_mse = np.mean(self.ae_mse_per_sample)
        #print('self.ae_mse = {}; ae_mse = {}'.format(self.ae_mse, ae_mse))

        # list all data in history
        print(self.history.history.keys())
        print('--------------------------------------------------------')

        #==================================================================
        # Train the GAN model
        # Adversarial ground truths
        valid = np.ones((self.batch_size, 1))
        fake = np.zeros((self.batch_size, 1))

        all_a_loss = []
        all_a_acc = []
        all_d_loss = []
        all_d_acc = []
        all_g_loss = []

        last_ten_d_acc = np.ones(losses_to_check_size)
        last_ten_g_losses = np.ones(losses_to_check_size)
        max_a_acc_criteria = 0.6
        min_d_acc_criteria = 0.9
        max_d_acc_criteria = 0.99
        min_g_loss_criteria = 0.3
        max_g_loss_criteria = 0.7
        g_loss_criteria_increment_step = 0.1
        n_step_min_rep= 50
        n_step_counter = 0
        last_g_loss = 1
        train_dae = True
        help_discriminator = False
        help_generator = False
        #n_epochs_gen_data_as_fake = 300
        noise_width_min = 0.1
        noise_width_max = 1
        d_loss_real = 0

        print('--------------------------------------------------------')
        print('Training GAN...')
        for epoch in range(self.n_epochs):

            noise_width = (noise_width_max - noise_width_min) * np.random.random_sample() + noise_width_min

            noise = np.random.normal(0, noise_width, (self.batch_size, self.latent_dim))

            # ---------------------
            #  Train DAE
            # ---------------------

            # Select a random batch of images
            idx = np.random.randint(0, train_data_scaled.shape[0], self.batch_size)
            inputs = train_data_scaled[idx]

            if train_dae:
                a_loss = self.autoencoder.train_on_batch(inputs, inputs)

                # Get the encoded data
                encoded_data = self.encoder.predict(inputs)

                if a_loss[1] > max_a_acc_criteria:
                    train_dae = False

            if not help_generator:
                # ---------------------
                #  Train Discriminator
                # ---------------------

                if False:# epoch > n_epochs_gen_data_as_fake:
                    gen_a_data, gen_n_data = self._generate_fake_and_or_normal_data(train_data_scaled)

                    if gen_n_data is not None:
                        inputs = gen_n_data
                else:
                    #noise = np.random.normal(0, noise_width, (self.batch_size, self.latent_dim))
                    # Generate a batch of new data
                    gen_a_data = self.generator.predict(noise + encoded_data)

                # Train the discriminator
                d_loss_real = self._train_on_batch_discriminator(inputs, valid)
                d_loss_fake = self._train_on_batch_discriminator(gen_a_data, fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # Store
                last_ten_d_acc = np.delete(np.append(last_ten_d_acc, d_loss[1]), 0)

            if help_discriminator:
                #last_ten_g_losses = np.delete(np.append(last_ten_g_losses, [1]), 0)
                if np.all(last_ten_d_acc >= max_d_acc_criteria):
                    help_discriminator = False
                # Store the loss values for plot
                all_a_loss.append(a_loss[0])
                all_a_acc.append(100*a_loss[1])
                all_d_loss.append(d_loss[0])
                all_d_acc.append(100*d_loss[1])
                all_g_loss.append(last_g_loss)
                # Plot the progress
                print ("%d [DAE loss: %f, acc.: %.2f%%] [D loss: %f, acc.: %.2f%%] [G loss: %f]" %
                       (epoch, a_loss[0], 100*a_loss[1], d_loss[0], 100*d_loss[1], last_g_loss))
                continue

            # ---------------------
            #  Train Generator
            # ---------------------

            #noise = np.random.normal(0, noise_width, (self.batch_size, self.latent_dim))

            # Train the generator (to have the discriminator label samples as valid)
            g_loss = self.combined.train_on_batch(noise + encoded_data, valid)

            # Store
            last_ten_g_losses = np.delete(np.append(last_ten_g_losses, g_loss), 0)

            if help_generator:
                if check_if_slightly_increasing(last_ten_g_losses, decreasing=True) and g_loss < min_g_loss_criteria:
                    help_generator = False
                    n_step_counter += 1
                    if min_g_loss_criteria < (max_g_loss_criteria - 0.01) and n_step_counter >= n_step_min_rep:
                        min_g_loss_criteria += g_loss_criteria_increment_step
                        n_step_counter = 0
                # Store the loss values for plot
                all_a_loss.append(a_loss[0])
                all_a_acc.append(100*a_loss[1])
                all_d_loss.append(d_loss[0])
                all_d_acc.append(100*d_loss[1])
                all_g_loss.append(g_loss)
                # Plot the progress
                print ("%d [DAE loss: %f, acc.: %.2f%%] [D loss: %f, acc.: %.2f%%] [G loss: %f]" %
                       (epoch, a_loss[0], 100*a_loss[1], d_loss[0], 100*d_loss[1], g_loss))
                continue

            if not help_discriminator and epoch > losses_to_check_size:
                if (np.all(last_ten_d_acc < min_d_acc_criteria) and d_loss_real[1] < min_d_acc_criteria) \
                        or d_loss[0] > g_loss:
                    last_g_loss = g_loss
                    help_discriminator = True
                    help_generator = False          # Ensure that these two are not true at the same time
                elif not help_generator:
                    if g_loss > min_g_loss_criteria:
                        help_generator = True
                        help_discriminator = False  # Ensure that these two are not true at the same time

            # Store the loss values for plot
            all_a_loss.append(a_loss[0])
            all_a_acc.append(100*a_loss[1])
            all_d_loss.append(d_loss[0])
            all_d_acc.append(100*d_loss[1])
            all_g_loss.append(g_loss)

            # Plot the progress
            print ("%d [DAE loss: %f, acc.: %.2f%%] [D loss: %f, acc.: %.2f%%] [G loss: %f]" %
                   (epoch, a_loss[0], 100*a_loss[1], d_loss[0], 100*d_loss[1], g_loss))
        print('--------------------------------------------------------')
        #==================================================================

        # Test the discriminator
        p_train_data = self._discriminator_predict(train_data_scaled)
        assert p_train_data.shape[0] == train_data_scaled.shape[0], ("The predicted data shape is not right!")

        # Test the generator
        noise = np.random.normal(0, 1, (self.batch_size, self.latent_dim))
        g_data = self.generator.predict(noise)
        assert g_data.shape == (self.batch_size, self.original_dim), ("The predicted data shape is not right!")
        p_g_data = self._discriminator_predict(g_data)
        assert p_g_data.shape[0] == g_data.shape[0], ("The predicted data shape is not right!")

        if save_model:
            # Save the scaler
            scaler_filename = self.model_dir_path + 'scaler_' + self.model_name + '.pkl'
            joblib.dump(scaler, scaler_filename)

            # Save weights to HDF5
            model_weights_filename = self.model_dir_path + self.model_name + '.h5'
            model_d_weights_filename = self.model_dir_path + self.model_d_name + '.h5'
            model_ae_weights_filename = self.model_dir_path + self.model_ae_name + '.h5'

            # Save weights
            self.combined.save_weights(model_weights_filename)
            self.discriminator.save_weights(model_d_weights_filename)
            self.autoencoder.save_weights(model_ae_weights_filename)

            print('Saved model to disk.')

        if print_and_plot_history:
            _x = range(self.n_epochs)
            # summarize loss
            fig1 = plt.figure()
            plt.plot(_x, all_a_loss, 'k-', _x, all_d_loss, 'b--', _x, all_g_loss, 'g.-')
            plt.title('Model learning loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['autoencoder', 'discriminator', 'generator'], loc='upper left')
            figure_name = self.model_dir_path + self.model_name + '_loss.pdf'
            fig1.savefig(figure_name)
            if show_plots:
                plt.show()
            # summarize discriminator accuracy
            fig2 = plt.figure()
            plt.plot(_x, all_a_acc, 'k-', _x, all_d_acc, 'b--')
            plt.title('Model learning accuracy')
            plt.ylabel('acc')
            plt.xlabel('epoch')
            plt.legend(['autoencoder', 'discriminator'], loc='upper left')
            figure_name = self.model_dir_path + self.model_name + '_acc.pdf'
            fig2.savefig(figure_name)
            if show_plots:
                plt.show()

    def load_weights(self):
        """
        Load the weights from file to the model.
        :param model_dir_path:
        :param iteration_number:
        :return:
        """
        model_weights_filename = self.model_dir_path + self.model_name + '.h5'
        model_d_weights_filename = self.model_dir_path + self.model_d_name + '.h5'
        model_ae_weights_filename = self.model_dir_path + self.model_ae_name + '.h5'

        # load weights into new model
        self.autoencoder.load_weights(model_ae_weights_filename)
        self.discriminator.load_weights(model_d_weights_filename)
        self.combined.load_weights(model_weights_filename)
        print("Loaded weights from disk")

    def test_model(self,
                   test_data,
                   is_abnormal = False):
        """
        Apply the learned model to the test samples and get scores.
        :param test_data:
        :param model_dir_path:
        :param iteration_number:
        :return:
        """
        # Get the scaler
        scaler_filename = self.model_dir_path + 'scaler_' + self.model_name + '.pkl'
        scaler = joblib.load(scaler_filename)

        # Scale the test data
        test_data_scaled = scaler.transform(test_data)

        # evaluate
        if is_abnormal:
            labels = np.zeros((test_data_scaled.shape[0], 1))
        else:
            labels = np.ones((test_data_scaled.shape[0], 1))

        test_score = self._discriminator_evaluate(test_data_scaled, labels)

        #======================================================================
        # Test
        #p_label = self._discriminator_predict(test_data_scaled)
        #if is_abnormal:
        #    labels = np.ones((test_data_scaled.shape[0], 1))
        #    test_test = self._discriminator_evaluate(test_data_scaled, labels)
        #    mse_test = test_test[0]
        #    acc_test = test_test[1]
        #======================================================================

        return test_score[0], test_score[1]

class BuildOurMethodV3():
    """
    First version of our method: Implement of generator using pre-trained DAE.
    Inspired by the following:
    - https://blog.keras.io/building-autoencoders-in-keras.html;
    - https://github.com/civisanalytics/muffnn/blob/master/muffnn/autoencoder/autoencoder.py
    """

    def __init__(self,
                 original_dim=125,
                 hidden_units=(128,64,32,16,8),
                 batch_size=128,
                 n_epochs=10000,
                 hidden_activation='relu',
                 output_activation='sigmoid',
                 loss='mean_squared_error',
                 metrics='accuracy',
                 model_dir_path = '',
                 iteration_number = 0):
        """
        Builds and compiles an autoencoder.
        :param input_size:
        :param hidden_units:
        :param batch_size:
        :param n_epochs:
        :param learning_rate:
        :param hidden_activation:
        :param output_activation:
        :param optimiser:
        :param loss:
        :param shuffle_data:
        """
        self.original_dim = original_dim
        self.hidden_units = hidden_units
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.optimizer = RMSprop(lr=0.00005)
        self.loss = loss
        self.metrics = metrics
        self.latent_dim = hidden_units[-1]
        self.model_dir_path = model_dir_path
        self.iteration_number = iteration_number
        self.squeezed_size = 25

        self.model_name = 'model_' + str(self.iteration_number)
        self.model_d_name = 'model_d_' + str(self.iteration_number)
        self.model_ae_name = 'model_ae_' + str(self.iteration_number)

        # Build encoder and decoder
        self.encoder = self.build_encoder()
        self.decoder = self.build_generator()

        # Build and compile the regular deep autoencoder separately
        input_data = Input(shape=tuple([self.original_dim]))
        r_input_data = self.decoder(self.encoder(input_data))

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=self.loss,
            optimizer=self.optimizer,
            metrics=[self.metrics])

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        #diff = Lambda(abs_diff, output_shape=(self.original_dim,), name='abs_diff')([input_data, r_input_data])
        squeezed_r_data = Lambda(decrease_size_by_averaging, output_shape=(self.squeezed_size,),
                                 name='squeeze')(r_input_data)

        # The discriminator takes generated images as input and determines validity
        validity = self.discriminator(squeezed_r_data)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(input_data, validity)
        self.combined.compile(loss=self.loss, optimizer=self.optimizer, metrics=[self.metrics])

    def build_encoder(self):
        # Encoder
        _input = Input(shape=tuple([self.original_dim]))
        _encoded = _input

        for units in self.hidden_units:
            _encoded = Dense(units, activation=self.hidden_activation)(_encoded)

        return Model(_input, _encoded, name='encoder')

    def build_generator(self):
        # Decoder
        _latent_input = Input(shape=(self.latent_dim,), name='z_sampling')
        _x = _latent_input

        for units in reversed(self.hidden_units[:-1]):
            _x = Dense(units, activation=self.hidden_activation)(_x)

        _output = Dense(self.original_dim, activation=self.output_activation)(_x)

        return Model(_latent_input, _output, name='generator')

    def build_discriminator(self):
        # Discriminator
        _input = Input(shape=tuple([self.squeezed_size]), name='input_sampling')
        _x = _input

        hidden_units = (24, 12, 6, 3)

        for units in hidden_units:
            _x = Dense(units, activation=self.hidden_activation)(_x)

        _validity = Dense(1, activation=self.output_activation)(_x)

        return Model(_input, _validity, name='discriminator')

    def _autoencoder_predict(self, scaled_data):
        return self.decoder.predict(self.encoder.predict(scaled_data))

    def _squeeze_data(self, scaled_data):
        return scaled_data.reshape(-1, 25, 5).mean(axis=2)
        #return [x.reshape(-1, 5).mean(axis=1) for x in scaled_data]

    def _train_on_batch_discriminator(self, scaled_data, label):
        # Train discriminator using pre-trained deep autoencoder
        #predicted_data = self._autoencoder_predict(scaled_data)

        #diff_data = np.abs(predicted_data - scaled_data)

        return self.discriminator.train_on_batch(self._squeeze_data(scaled_data), label)

    def _test_on_batch_discriminator(self, scaled_data, label):
        # Train discriminator using pre-trained deep autoencoder
        #predicted_data = self._autoencoder_predict(scaled_data)

        #diff_data = np.abs(predicted_data - scaled_data)

        return self.discriminator.test_on_batch(self._squeeze_data(scaled_data), label)

    def train(self, train_data,
              save_model = True,
              print_and_plot_history = False, show_plots = False):
        """
        Fit the autoencoder model.
        :param train_data:
        :return:
        """
        print('Training GAN using pre-trained deep autoencoder:')

        # Scale the data
        #scaler = MinMaxScaler(feature_range=(-1,1))
        scaler = MinMaxScaler()
        train_data_scaled = scaler.fit_transform(train_data)

        #==================================================================
        # Train the GAN model
        # Adversarial ground truths
        valid = np.ones((self.batch_size, 1))
        fake = np.zeros((self.batch_size, 1))

        #all_a_loss = []
        #all_a_acc = []
        all_d_loss = []
        all_d_acc = []
        all_g_loss = []
        all_g_acc = []

        #last_ten_d_acc = np.ones(losses_to_check_size)
        #last_ten_g_losses = np.ones(losses_to_check_size)
        min_d_r_acc_criteria = 0.8
        max_d_r_acc_criteria = 0.9
        min_d_f_acc_criteria = 0.01
        max_d_f_acc_criteria = 0.2
        #max_d_acc_criteria = 0.5
        #min_g_loss_criteria = 0.3
        min_g_acc_criteria = 0.5
        #g_loss_criteria_increment_step = 0.1
        #n_step_min_rep= 50
        #n_step_counter = 0
        last_g_loss = 1
        last_g_acc = 0
        help_discriminator = False
        help_generator = False
        start_help_generator_epoch = 1000
        stop_help_generator_epoch = 5000
        #noise_width_min = 0.1
        #noise_width_max = 1
        noise_width = 1
        d_loss_real = 0
        d_loss_fake = 0

        print('--------------------------------------------------------')
        print('Training GAN...')
        for epoch in range(self.n_epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of images
            idx = np.random.randint(0, train_data_scaled.shape[0], self.batch_size)
            i_data = train_data_scaled[idx]

            #noise = np.random.normal(0, noise_width, (self.batch_size, self.original_dim))

            if epoch > start_help_generator_epoch and epoch < stop_help_generator_epoch:
                help_generator = True
            elif help_generator:
                help_generator = False

            # ---------------------
            #  Train Discriminator
            # ---------------------

            if not help_generator:
                # ---------------------
                #  Train Discriminator
                # ---------------------
                r_data = self._autoencoder_predict(i_data)

                # Train the discriminator
                d_loss_real = self._train_on_batch_discriminator(i_data, valid)
                d_loss_fake = self._train_on_batch_discriminator(r_data, fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # Store
                #last_ten_d_acc = np.delete(np.append(last_ten_d_acc, d_loss[1]), 0)
            else:
                # ---------------------
                #  Evaluate Discriminator
                # ---------------------
                r_data = self._autoencoder_predict(i_data)

                # Train the discriminator
                d_loss_real = self._test_on_batch_discriminator(i_data, valid)
                d_loss_fake = self._test_on_batch_discriminator(r_data, fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            #if help_discriminator:
            #    if d_loss_real[1] >= max_d_r_acc_criteria:
            #        help_discriminator = False
                # Store the loss values for plot
            #    all_d_loss.append(d_loss[0])
            #    all_d_acc.append(100*d_loss[1])
            #    all_g_loss.append(last_g_loss)
            #    all_g_acc.append(last_g_acc)
                # Plot the progress
            #    print ("%d [D loss: %f, acc.: %.2f%%] x[G loss: %f, acc.: %.2f%%]" %
            #           (epoch, d_loss[0], 100*d_loss[1], last_g_loss, 100*last_g_acc))
            #    continue

            # ---------------------
            #  Train Generator
            # ---------------------

            #noise = np.random.normal(0, noise_width, (self.batch_size, self.original_dim))

            # Train the generator (to have the discriminator label samples as valid)
            g_loss = self.combined.train_on_batch(i_data, valid)

            # Store
            #last_ten_g_losses = np.delete(np.append(last_ten_g_losses, g_loss), 0)

            #if help_generator:

                # ---------------------
                #  Evaluate Discriminator
                # ---------------------
            #    r_data = self._autoencoder_predict(i_data)

                # Train the discriminator
            #    d_loss_real = self._test_on_batch_discriminator(i_data, valid)
            #    d_loss_fake = self._test_on_batch_discriminator(r_data, fake)
            #    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            #    if d_loss_fake[1] < min_d_f_acc_criteria:
            #        help_generator = False
                # Store the loss values for plot
            #    all_d_loss.append(d_loss[0])
            #    all_d_acc.append(100*d_loss[1])
            #    all_g_loss.append(g_loss[0])
            #    all_g_acc.append(100*g_loss[1])
                # Plot the progress
            #    print ("%d x[D loss: %f, acc.: %.2f%%] [G loss: %f, acc.: %.2f%%]" %
            #           (epoch, d_loss[0], 100*d_loss[1], g_loss[0], 100*g_loss[1]))
            #    continue

            #if not help_generator:
            #    if d_loss_fake[1] > max_d_f_acc_criteria or g_loss[1] < min_g_acc_criteria:
            #        help_generator = True
            #        help_discriminator = False  # Ensure that these two are not true at the same time
            #    elif not help_discriminator and d_loss_real[1] < min_d_r_acc_criteria:
            #        last_g_loss = g_loss[0]
            #        last_g_acc = g_loss[1]
            #        help_discriminator = True
            #        help_generator = False      # Ensure that these two are not true at the same time

            # Store the loss values for plot
            all_d_loss.append(d_loss[0])
            all_d_acc.append(100*d_loss[1])
            all_g_loss.append(g_loss[0])
            all_g_acc.append(100*g_loss[1])

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f, acc.: %.2f%%]" %
                   (epoch, d_loss[0], 100*d_loss[1], g_loss[0], 100*g_loss[1]))
        print('--------------------------------------------------------')
        #==================================================================

        # Test the discriminator
        p_train_data = self.combined.predict(train_data_scaled)
        assert p_train_data.shape[0] == train_data_scaled.shape[0], ("The predicted data shape is not right!")

        # Test the generator
        noise = np.random.normal(0, 1, (self.batch_size, self.latent_dim))
        g_data = self.decoder.predict(noise)
        assert g_data.shape == (self.batch_size, self.original_dim), ("The predicted data shape is not right!")
        p_g_data = self.combined.predict(g_data)
        assert p_g_data.shape[0] == g_data.shape[0], ("The predicted data shape is not right!")

        if save_model:
            # Save the scaler
            scaler_filename = self.model_dir_path + 'scaler_' + self.model_name + '.pkl'
            joblib.dump(scaler, scaler_filename)

            # Save weights to HDF5
            model_weights_filename = self.model_dir_path + self.model_name + '.h5'
            model_d_weights_filename = self.model_dir_path + self.model_d_name + '.h5'

            # Save weights
            self.combined.save_weights(model_weights_filename)
            self.discriminator.save_weights(model_d_weights_filename)

            print('Saved model to disk.')

        if print_and_plot_history:
            _x = range(self.n_epochs)
            # summarize loss
            fig1 = plt.figure()
            plt.plot(_x, all_d_loss, 'k-', _x, all_g_loss, 'b--')
            plt.title('Model learning loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['discriminator', 'generator'], loc='upper left')
            figure_name = self.model_dir_path + self.model_name + '_loss.pdf'
            fig1.savefig(figure_name)
            if show_plots:
                plt.show()
            # summarize discriminator accuracy
            fig2 = plt.figure()
            plt.plot(_x, all_d_acc, 'k-', _x, all_g_acc, 'b--')
            plt.title('Model learning accuracy')
            plt.ylabel('acc')
            plt.xlabel('epoch')
            plt.legend(['discriminator', 'generator'], loc='upper left')
            figure_name = self.model_dir_path + self.model_name + '_acc.pdf'
            fig2.savefig(figure_name)
            if show_plots:
                plt.show()

    def load_weights(self):
        """
        Load the weights from file to the model.
        :param model_dir_path:
        :param iteration_number:
        :return:
        """
        model_weights_filename = self.model_dir_path + self.model_name + '.h5'
        model_d_weights_filename = self.model_dir_path + self.model_d_name + '.h5'

        # load weights into new model
        self.discriminator.load_weights(model_d_weights_filename)
        self.combined.load_weights(model_weights_filename)
        print("Loaded weights from disk")

    def test_model(self,
                   test_data,
                   is_abnormal = False):
        """
        Apply the learned model to the test samples and get scores.
        :param test_data:
        :param model_dir_path:
        :param iteration_number:
        :return:
        """
        # Get the scaler
        scaler_filename = self.model_dir_path + 'scaler_' + self.model_name + '.pkl'
        scaler = joblib.load(scaler_filename)

        # Scale the test data
        test_data_scaled = scaler.transform(test_data)

        # evaluate
        if is_abnormal:
            labels = np.zeros((test_data_scaled.shape[0], 1))
        else:
            labels = np.ones((test_data_scaled.shape[0], 1))

        test_score = self.combined.evaluate(test_data_scaled, labels)

        #======================================================================
        # Test
        #p_label = self._discriminator_predict(test_data_scaled)
        #if is_abnormal:
        #    labels = np.ones((test_data_scaled.shape[0], 1))
        #    test_test = self._discriminator_evaluate(test_data_scaled, labels)
        #    mse_test = test_test[0]
        #    acc_test = test_test[1]
        #======================================================================

        return test_score[0], test_score[1]

class BuildOurMethodV4():
    """
    First version of our method: Implement of generator using pre-trained DAE.
    Inspired by the following:
    - https://blog.keras.io/building-autoencoders-in-keras.html;
    - https://github.com/civisanalytics/muffnn/blob/master/muffnn/autoencoder/autoencoder.py
    """

    def __init__(self,
                 original_dim=125,
                 hidden_units=best_nn_setting,
                 batch_size=128,
                 n_epochs=500000,
                 hidden_activation='relu',
                 output_activation='sigmoid',
                 loss='mean_squared_error',
                 metrics='accuracy',
                 model_dir_path = '',
                 iteration_number = 0):
        """
        Builds and compiles an autoencoder.
        :param input_size:
        :param hidden_units:
        :param batch_size:
        :param n_epochs:
        :param learning_rate:
        :param hidden_activation:
        :param output_activation:
        :param optimiser:
        :param loss:
        :param shuffle_data:
        """
        self.original_dim = original_dim
        self.hidden_units = hidden_units
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.optimizer_a = 'rmsprop'
        self.initial_d_lr = 0.000005
        self.initial_g_lr = self.initial_d_lr / 2    #g_lr must initially be lower than d_lr
        self.stopping_g_lr = self.initial_d_lr / 4
        self.g_lr_range = np.linspace(self.initial_g_lr, self.stopping_g_lr, 11)
        self.optimizer_d = RMSprop(lr=self.initial_d_lr) #RMSprop(lr=0.00005)
        self.optimizer_g = RMSprop(lr=self.initial_g_lr)
        self.loss = loss
        self.metrics = metrics
        self.latent_dim = hidden_units[-1]
        self.model_dir_path = model_dir_path
        self.iteration_number = iteration_number
        self.consecutive_mse_size = 5

        self.model_name = 'model_' + str(self.iteration_number)
        self.model_d_name = 'model_d_' + str(self.iteration_number)
        self.model_ae_name = 'model_ae_' + str(self.iteration_number)

        # Build encoder and decoder
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()

        # Build and compile the regular deep autoencoder separately
        input_data = Input(shape=tuple([self.original_dim]))
        r_input_data = self.decoder(self.encoder(input_data))
        self.autoencoder = Model(input_data, r_input_data, name='autoencoder')
        self.autoencoder.compile(optimizer=self.optimizer_a, loss=self.loss, metrics=[self.metrics])

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=self.loss,
            optimizer=self.optimizer_d,
            metrics=[self.metrics])

        # Build generator of GAN
        self.generator = self.build_generator()

        # The generator takes noise as input and generates data
        z = Input(shape=(self.latent_dim,))
        gen_mse_data = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False
        self.autoencoder.trainable = False

        # The discriminator takes generated images as input and determines validity
        #rep_data = self.autoencoder(gen_data)
        #mse_data = Lambda(consecutive_mse, output_shape=(self.consecutive_mse_size,),
        #                  name='consecutive_mse')([rep_data, gen_data])
        validity = self.discriminator(gen_mse_data)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, validity)
        self.combined.compile(loss=self.loss, optimizer=self.optimizer_g, metrics=[self.metrics])

    def build_encoder(self):
        # Encoder
        _input = Input(shape=tuple([self.original_dim]))
        _encoded = _input

        for units in self.hidden_units:
            _encoded = Dense(units, activation=self.hidden_activation)(_encoded)

        return Model(_input, _encoded, name='encoder')

    def build_decoder(self):
        # Decoder
        _latent_input = Input(shape=(self.latent_dim,), name='z_sampling')
        _x = _latent_input

        for units in reversed(self.hidden_units[:-1]):
            _x = Dense(units, activation=self.hidden_activation)(_x)

        _output = Dense(self.original_dim, activation=self.output_activation)(_x)

        return Model(_latent_input, _output, name='generator')

    def build_generator(self):
        # Decoder
        _latent_input = Input(shape=(self.latent_dim,), name='z_sampling')
        _x = _latent_input

        for units in reversed(self.hidden_units[:-1]):
            _x = Dense(units, activation=self.hidden_activation)(_x)

        _output = Dense(self.consecutive_mse_size, activation=self.output_activation)(_x)

        return Model(_latent_input, _output, name='generator')

    def build_discriminator(self):
        # Discriminator
        _input = Input(shape=tuple([self.consecutive_mse_size]), name='input_sampling')
        _x = _input

        #hidden_units = (4,2)

        for units in self.hidden_units:
            _x = Dense(units, activation=self.hidden_activation)(_x)

        _validity = Dense(1, activation=self.output_activation)(_x)

        return Model(_input, _validity, name='discriminator')

    def _compute_consecutive_mse(self, scaled_data):
        predicted_data = self.autoencoder.predict(scaled_data)
        squared_error = (predicted_data - scaled_data)**2
        return squared_error.reshape(-1, 5, 25).mean(axis=2)

    def _train_on_batch_discriminator(self, scaled_data, label):
        return self.discriminator.train_on_batch(self._compute_consecutive_mse(scaled_data), label)

    def _discriminator_predict(self, scaled_data):
        return self.discriminator.predict(self._compute_consecutive_mse(scaled_data))

    def _discriminator_evaluate(self, scaled_data, labels):
        return self.discriminator.evaluate(self._compute_consecutive_mse(scaled_data), labels)

    def train(self, train_data,
              save_model = True,
              print_and_plot_history = False, show_plots = False):
        """
        Fit the autoencoder model.
        :param train_data:
        :return:
        """
        print('Training GAN using pre-trained deep autoencoder:')

        # Scale the data
        #scaler = MinMaxScaler(feature_range=(-1,1))
        scaler = MinMaxScaler()
        train_data_scaled = scaler.fit_transform(train_data)

        # First, train the deep regular autoencoder which will be used as a pre-trained
        # network for training discriminator
        print('--------------------------------------------------------')
        print('Training the regular deep autoencoder...')
        pt_dae_epochs = 100
        pt_dae_validation_split=0.20
        pt_dae_shuffle_data = True
        self.history = self.autoencoder.fit(train_data_scaled, train_data_scaled,
                                            validation_split=pt_dae_validation_split,
                                            epochs=pt_dae_epochs,
                                            batch_size=self.batch_size,
                                            shuffle=pt_dae_shuffle_data)
        # evaluate the model
        self.ae_score = self.autoencoder.evaluate(train_data_scaled, train_data_scaled)
        self.ae_mse = self.ae_score[0]
        predicted_train_data = self.autoencoder.predict(train_data_scaled)
        assert predicted_train_data.shape == train_data_scaled.shape, ("The predicted data shape is not right!")

        # Get the mse for each sample
        #self.ae_mse_per_sample = [mean_squared_error(train_data_scaled[i,:], predicted_train_data[i,:])
        #                          for i in range(train_data_scaled.shape[0])]
        #ae_mse = np.mean(self.ae_mse_per_sample)
        #print('self.ae_mse = {}; ae_mse = {}'.format(self.ae_mse, ae_mse))

        # list all data in history
        print(self.history.history.keys())
        print('--------------------------------------------------------')

        #==================================================================
        # Train the GAN model
        # Adversarial ground truths
        valid = np.ones((self.batch_size, 1))
        fake = np.zeros((self.batch_size, 1))

        all_a_loss = []
        all_a_acc = []
        all_d_loss = []
        all_d_acc = []
        all_d_lr = []
        all_g_loss = []
        all_g_acc = []
        all_g_lr = []

        noise_width = 1

        train_only_d_real = False
        g_acc_started_from_zero = False
        last_g_acc_s = np.zeros(100)
        max_epoch_for_helping_g = 50000

        print('--------------------------------------------------------')
        print('Training GAN...')
        for epoch in range(self.n_epochs):

            noise = np.random.normal(0, noise_width, (self.batch_size, self.latent_dim))

            # ---------------------
            #  Train DAE
            # ---------------------

            # Select a random batch of images
            idx = np.random.randint(0, train_data_scaled.shape[0], self.batch_size)
            inputs = train_data_scaled[idx]

            a_loss = self.autoencoder.test_on_batch(inputs, inputs)

            # Generate a batch of new data
            gen_a_mse_data = self.generator.predict(noise)

            # Train the discriminator
            d_loss_real = self._train_on_batch_discriminator(inputs, valid)
            if not train_only_d_real:
                d_loss_fake = self.discriminator.train_on_batch(gen_a_mse_data, fake)
            else:
                d_loss_fake = self.discriminator.test_on_batch(gen_a_mse_data, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            if d_loss_real[1] < 0.99:
                train_only_d_real = True
            else:
                train_only_d_real = False

            # ---------------------
            #  Train Generator
            # ---------------------

            if not train_only_d_real or epoch < 100:
                # Train the generator (to have the discriminator label samples as valid)
                g_loss = self.combined.train_on_batch(noise, valid)
            else:
                # Test the generator (to have the discriminator label samples as valid)
                g_loss = self.combined.test_on_batch(noise, valid)

            # Store
            last_g_acc_s = np.delete(np.append(last_g_acc_s, g_loss[1]), 0)

            if not g_acc_started_from_zero:
                if round(g_loss[1],4) == 0.0:
                    g_acc_started_from_zero = True

            # Adjust the learning rate of the generator
            g_acc_approx = round(g_loss[1],1)
            changed_g_lr = self.g_lr_range[int(g_acc_approx*10)]
            K.set_value(self.combined.optimizer.lr, changed_g_lr)

            # Store the loss values for plot
            all_a_loss.append(a_loss[0])
            all_a_acc.append(100*a_loss[1])
            all_d_loss.append(d_loss[0])
            all_d_acc.append(100*d_loss[1])
            all_g_loss.append(g_loss[0])
            all_g_acc.append(100*g_loss[1])
            # Store the discriminator lr
            d_lr = float(K.get_value(self.discriminator.optimizer.lr))
            all_d_lr.append(d_lr)
            # Store the generator lr
            g_lr = float(K.get_value(self.combined.optimizer.lr))
            all_g_lr.append(g_lr)

            # Plot the progress
            print ("%d [DAE loss: %f, acc.: %.2f%%] [D loss: %f, acc.: %.2f%%, lr: %.8f] "
                   "[G loss: %f, acc.: %.2f%%, lr: %.8f]" %
                   (epoch, a_loss[0], 100*a_loss[1], d_loss[0], 100*d_loss[1], d_lr, g_loss[0], 100*g_loss[1], g_lr))

            # Stop if the stooping criteria is met
            if g_acc_started_from_zero and all(last_g_acc_s > 0.95):
                self.n_epochs = epoch + 1
                break

            # If the g_acc is still null after many epochs, increase th g_lr by factor 2
            if g_acc_started_from_zero and (epoch % max_epoch_for_helping_g == 0) and epoch > 0:
                if round(g_loss[1],4) == 0.0:
                    self.initial_g_lr *= 2
                    self.g_lr_range = np.linspace(self.initial_g_lr, self.stopping_g_lr, 11)
                    K.set_value(self.combined.optimizer.lr, self.initial_g_lr)
        print('--------------------------------------------------------')
        #==================================================================

        # Test the discriminator
        p_train_data = self._discriminator_predict(train_data_scaled)
        assert p_train_data.shape[0] == train_data_scaled.shape[0], ("The predicted data shape is not right!")

        # Test the generator
        noise = np.random.normal(0, 1, (self.batch_size, self.latent_dim))
        g_data = self.generator.predict(noise)
        assert g_data.shape == (self.batch_size, self.consecutive_mse_size), ("The predicted data shape is not right!")
        #p_g_data = self._discriminator_predict(g_data)
        #assert p_g_data.shape[0] == g_data.shape[0], ("The predicted data shape is not right!")

        if save_model:
            # Save the scaler
            scaler_filename = self.model_dir_path + 'scaler_' + self.model_name + '.pkl'
            joblib.dump(scaler, scaler_filename)

            # Save weights to HDF5
            model_weights_filename = self.model_dir_path + self.model_name + '.h5'
            model_d_weights_filename = self.model_dir_path + self.model_d_name + '.h5'
            model_ae_weights_filename = self.model_dir_path + self.model_ae_name + '.h5'

            # Save weights
            self.combined.save_weights(model_weights_filename)
            self.discriminator.save_weights(model_d_weights_filename)
            self.autoencoder.save_weights(model_ae_weights_filename)

            print('Saved model to disk.')

        if print_and_plot_history:
            _x = range(self.n_epochs)
            # summarize loss
            fig1 = plt.figure()
            plt.plot(_x, all_a_loss, 'k-', _x, all_d_loss, 'b--', _x, all_g_loss, 'g.-')
            plt.title('Model learning loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['autoencoder', 'discriminator', 'generator'], loc='upper left')
            figure_name = self.model_dir_path + self.model_name + '_loss.pdf'
            fig1.savefig(figure_name)
            if show_plots:
                plt.show()
            # summarize discriminator accuracy
            fig2 = plt.figure()
            plt.plot(_x, all_a_acc, 'k-', _x, all_d_acc, 'b--', _x, all_g_acc, 'g.-')
            plt.title('Model learning accuracy')
            plt.ylabel('acc')
            plt.xlabel('epoch')
            plt.legend(['autoencoder', 'discriminator', 'generator'], loc='upper left')
            figure_name = self.model_dir_path + self.model_name + '_acc.pdf'
            fig2.savefig(figure_name)
            if show_plots:
                plt.show()
            # summarize discriminator accuracy
            fig3 = plt.figure()
            plt.plot(_x, all_d_lr, 'k-', _x, all_g_lr, 'b--')
            plt.title('Model learning rate')
            plt.ylabel('lr')
            plt.xlabel('epoch')
            plt.legend(['discriminator', 'generator'], loc='upper left')
            figure_name = self.model_dir_path + self.model_name + '_lr.pdf'
            fig3.savefig(figure_name)
            if show_plots:
                plt.show()

    def load_weights(self):
        """
        Load the weights from file to the model.
        :param model_dir_path:
        :param iteration_number:
        :return:
        """
        model_weights_filename = self.model_dir_path + self.model_name + '.h5'
        model_d_weights_filename = self.model_dir_path + self.model_d_name + '.h5'
        model_ae_weights_filename = self.model_dir_path + self.model_ae_name + '.h5'

        # load weights into new model
        self.autoencoder.load_weights(model_ae_weights_filename)
        self.discriminator.load_weights(model_d_weights_filename)
        self.combined.load_weights(model_weights_filename)
        print("Loaded weights from disk")

    def test_model(self,
                   test_data,
                   is_abnormal = False,
                   test_ae = False):
        """
        Apply the learned model to the test samples and get scores.
        :param test_data:
        :param model_dir_path:
        :param iteration_number:
        :return:
        """
        # Get the scaler
        scaler_filename = self.model_dir_path + 'scaler_' + self.model_name + '.pkl'
        scaler = joblib.load(scaler_filename)

        # Scale the test data
        test_data_scaled = scaler.transform(test_data)

        # evaluate
        if is_abnormal:
            labels = np.zeros((test_data_scaled.shape[0], 1))
        else:
            labels = np.ones((test_data_scaled.shape[0], 1))

        test_score = self._discriminator_evaluate(test_data_scaled, labels)

        if test_ae:
            # Test autoencoder standalone for comparison purpose
            ae_scores = self.autoencoder.evaluate(test_data_scaled,test_data_scaled)
            ae_global_mse = ae_scores[0]
            ae_predicted_test_data = self.autoencoder.predict(test_data_scaled)
            assert ae_predicted_test_data.shape == test_data_scaled.shape, ("The predicted data shape is not right!")
            ae_mse_per_sample = [mean_squared_error(test_data_scaled[i,:], ae_predicted_test_data[i,:])
                                 for i in range(test_data_scaled.shape[0])]

            return test_score[0], test_score[1], ae_global_mse, ae_mse_per_sample
        else:
            return test_score[0], test_score[1]

    def test_complete_model(self,
                            test_data,
                            is_abnormal = False,
                            test_ae = False,
                            result_dir_path = '',
                            classification_threshold = complete_trajectory_threshold):
        """
        Apply the learned model to the test samples and get scores.
        :param test_data: includes the object id
        :param model_dir_path:
        :param iteration_number:
        :return:
        """
        # Get the scaler
        scaler_filename = self.model_dir_path + 'scaler_' + self.model_name + '.pkl'
        scaler = joblib.load(scaler_filename)

        # Sort and extract the test data by object id
        object_id_vector = np.unique(test_data[:, 0])
        object_id_vector = np.sort(object_id_vector)

        acc_list = []
        ae_acc_list = []

        average_acc = False

        if test_ae:
            trained_model_summary_results_filename = result_dir_path + 'summary_results.csv'

            # Ref.: https://stackoverflow.com/questions/29451030/why-doesnt-np-genfromtxt-remove-header-while-importing-in-python
            with open(trained_model_summary_results_filename, 'r') as results:
                line = results.readline()
                header = [e for e in line.strip().split(',') if e]
                results_array = np.genfromtxt(results, names=header, dtype=None, delimiter=',')

            threshold_value = results_array['ae_threshold'][self.iteration_number]

            #average_acc = True

        for object_id in object_id_vector:
            object_test_data = test_data[test_data[:, 0] == object_id, :]

            # Remove the first column
            object_test_data = object_test_data[:, 1:]

            # Scale the test data
            object_test_data_scaled = scaler.transform(object_test_data)

            # evaluate
            if is_abnormal:
                labels = np.zeros((object_test_data_scaled.shape[0], 1))
            else:
                labels = np.ones((object_test_data_scaled.shape[0], 1))

            test_score = self._discriminator_evaluate(object_test_data_scaled, labels)

            # Store the accuracy
            acc_list.append(test_score[1])

            if test_ae:
                # Test autoencoder standalone for comparison purpose
                ae_predicted_test_data = self.autoencoder.predict(object_test_data_scaled)
                assert ae_predicted_test_data.shape == object_test_data_scaled.shape, \
                    ("The predicted data shape is not right!")
                ae_mse_per_sample = [mean_squared_error(object_test_data_scaled[i,:], ae_predicted_test_data[i,:])
                                     for i in range(object_test_data_scaled.shape[0])]

                if is_abnormal:
                    ae_acc = sum([s > threshold_value for s in ae_mse_per_sample])/float(len(ae_mse_per_sample))
                else:
                    ae_acc = sum([s <= threshold_value for s in ae_mse_per_sample])/float(len(ae_mse_per_sample))

                # Store the accuracy
                ae_acc_list.append(ae_acc)

        acc_list = np.array(acc_list)

        if average_acc:
            global_acc = np.mean(acc_list)
        else:
            if is_abnormal:
                global_acc = sum(acc_list >= classification_threshold) / float(len(acc_list))
            else:
                global_acc = sum((1 - acc_list) < classification_threshold) / float(len(acc_list))

        if test_ae:
            ae_acc_list = np.array(ae_acc_list)

            if average_acc:
                ae_global_acc = np.mean(ae_acc_list)
            else:
                if is_abnormal:
                    ae_global_acc = sum(ae_acc_list >= classification_threshold) / float(len(ae_acc_list))
                else:
                    ae_global_acc = sum((1 - ae_acc_list) < classification_threshold) / float(len(ae_acc_list))

            return global_acc, ae_global_acc
        else:
            return global_acc


class BuildOurMethodV4v2:
    """
    First version of our method: Implement of generator using pre-trained DAE.
    Inspired by the following:
    - https://blog.keras.io/building-autoencoders-in-keras.html;
    - https://github.com/civisanalytics/muffnn/blob/master/muffnn/autoencoder/autoencoder.py
    -----------------------------------------
    Based on the new layout: [label, x, y, v_x, v_y, h, w, o]
    The following equations is used in order to get the appropriate sizes:
    i_dim = 1 + 7 * t_size;
    i_dim = 8 * pmse_size;
    t_size - pmse_size = 4;
    --> i_dim = 232; t_size = 33; pmse_size = 29.
    """

    def __init__(self,
                 original_dim=232,
                 hidden_units=best_nn_setting,
                 batch_size=128,
                 n_epochs=500000,
                 hidden_activation='relu',
                 output_activation='sigmoid',
                 loss='mean_squared_error',
                 metrics='accuracy',
                 model_dir_path = '',
                 iteration_number = 0):
        """
        Builds and compiles an autoencoder.
        :param input_size:
        :param hidden_units:
        :param batch_size:
        :param n_epochs:
        :param learning_rate:
        :param hidden_activation:
        :param output_activation:
        :param optimiser:
        :param loss:
        :param shuffle_data:
        """
        self.original_dim = original_dim
        self.hidden_units = hidden_units
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.optimizer_a = 'rmsprop'
        self.initial_d_lr = 0.000005
        self.initial_g_lr = self.initial_d_lr / 2    #g_lr must initially be lower than d_lr
        self.stopping_g_lr = self.initial_d_lr / 4
        self.g_lr_range = np.linspace(self.initial_g_lr, self.stopping_g_lr, 11)
        self.optimizer_d = RMSprop(lr=self.initial_d_lr) #RMSprop(lr=0.00005)
        self.optimizer_g = RMSprop(lr=self.initial_g_lr)
        self.loss = loss
        self.metrics = metrics
        self.latent_dim = hidden_units[-1]
        self.model_dir_path = model_dir_path
        self.iteration_number = iteration_number
        self.mse_size = 8
        self.pmse_size = 29

        self.model_name = 'model_' + str(self.iteration_number)
        self.model_d_name = 'model_d_' + str(self.iteration_number)
        self.model_ae_name = 'model_ae_' + str(self.iteration_number)

        # Build encoder and decoder
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()

        # Build and compile the regular deep autoencoder separately
        input_data = Input(shape=tuple([self.original_dim]))
        r_input_data = self.decoder(self.encoder(input_data))
        self.autoencoder = Model(input_data, r_input_data, name='autoencoder')
        self.autoencoder.compile(optimizer=self.optimizer_a, loss=self.loss, metrics=[self.metrics])

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=self.loss,
            optimizer=self.optimizer_d,
            metrics=[self.metrics])

        # Build generator of GAN
        self.generator = self.build_generator()

        # The generator takes noise as input and generates data
        z = Input(shape=(self.latent_dim,))
        gen_mse_data = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False
        self.autoencoder.trainable = False

        # The discriminator takes generated images as input and determines validity
        validity = self.discriminator(gen_mse_data)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, validity)
        self.combined.compile(loss=self.loss, optimizer=self.optimizer_g, metrics=[self.metrics])

    def build_encoder(self):
        # Encoder
        _input = Input(shape=tuple([self.original_dim]))
        _encoded = _input

        for units in self.hidden_units:
            _encoded = Dense(units, activation=self.hidden_activation)(_encoded)

        return Model(_input, _encoded, name='encoder')

    def build_decoder(self):
        # Decoder
        _latent_input = Input(shape=(self.latent_dim,), name='z_sampling')
        _x = _latent_input

        for units in reversed(self.hidden_units[:-1]):
            _x = Dense(units, activation=self.hidden_activation)(_x)

        _output = Dense(self.original_dim, activation=self.output_activation)(_x)

        return Model(_latent_input, _output, name='generator')

    def build_generator(self):
        # Decoder
        _latent_input = Input(shape=(self.latent_dim,), name='z_sampling')
        _x = _latent_input

        for units in reversed(self.hidden_units[:-1]):
            _x = Dense(units, activation=self.hidden_activation)(_x)

        _output = Dense(self.mse_size, activation=self.output_activation)(_x)

        return Model(_latent_input, _output, name='generator')

    def build_discriminator(self):
        # Discriminator
        _input = Input(shape=tuple([self.mse_size]), name='input_sampling')
        _x = _input

        #hidden_units = (4,2)

        for units in self.hidden_units:
            _x = Dense(units, activation=self.hidden_activation)(_x)

        _validity = Dense(1, activation=self.output_activation)(_x)

        return Model(_input, _validity, name='discriminator')

    def _compute_consecutive_mse(self, scaled_data):
        predicted_data = self.autoencoder.predict(scaled_data)
        squared_error = (predicted_data - scaled_data)**2
        return squared_error.reshape(-1, self.mse_size, self.pmse_size).mean(axis=2)

    def _train_on_batch_discriminator(self, scaled_data, label):
        return self.discriminator.train_on_batch(self._compute_consecutive_mse(scaled_data), label)

    def _discriminator_predict(self, scaled_data):
        return self.discriminator.predict(self._compute_consecutive_mse(scaled_data))

    def _discriminator_evaluate(self, scaled_data, labels):
        return self.discriminator.evaluate(self._compute_consecutive_mse(scaled_data), labels)

    def train(self, train_data,
              save_model = True,
              print_and_plot_history = False, show_plots = False):
        """
        Fit the autoencoder model.
        :param train_data:
        :return:
        """
        print('Training GAN using pre-trained deep autoencoder:')

        # Scale the data
        #scaler = MinMaxScaler(feature_range=(-1,1))
        scaler = MinMaxScaler()
        train_data_scaled = scaler.fit_transform(train_data)

        # First, train the deep regular autoencoder which will be used as a pre-trained
        # network for training discriminator
        print('--------------------------------------------------------')
        print('Training the regular deep autoencoder...')
        pt_dae_epochs = 100
        pt_dae_validation_split=0.20
        pt_dae_shuffle_data = True
        self.history = self.autoencoder.fit(train_data_scaled, train_data_scaled,
                                            validation_split=pt_dae_validation_split,
                                            epochs=pt_dae_epochs,
                                            batch_size=self.batch_size,
                                            shuffle=pt_dae_shuffle_data)
        # evaluate the model
        self.ae_score = self.autoencoder.evaluate(train_data_scaled, train_data_scaled)
        self.ae_mse = self.ae_score[0]
        predicted_train_data = self.autoencoder.predict(train_data_scaled)
        assert predicted_train_data.shape == train_data_scaled.shape, ("The predicted data shape is not right!")

        # Get the mse for each sample
        #self.ae_mse_per_sample = [mean_squared_error(train_data_scaled[i,:], predicted_train_data[i,:])
        #                          for i in range(train_data_scaled.shape[0])]
        #ae_mse = np.mean(self.ae_mse_per_sample)
        #print('self.ae_mse = {}; ae_mse = {}'.format(self.ae_mse, ae_mse))

        # list all data in history
        print(self.history.history.keys())
        print('--------------------------------------------------------')

        #==================================================================
        # Train the GAN model
        # Adversarial ground truths
        valid = np.ones((self.batch_size, 1))
        fake = np.zeros((self.batch_size, 1))

        all_a_loss = []
        all_a_acc = []
        all_d_loss = []
        all_d_acc = []
        all_d_lr = []
        all_g_loss = []
        all_g_acc = []
        all_g_lr = []

        noise_width = 1

        train_only_d_real = False
        g_acc_started_from_zero = False
        last_g_acc_s = np.zeros(100)
        max_epoch_for_helping_g = 50000

        print('--------------------------------------------------------')
        print('Training GAN...')
        for epoch in range(self.n_epochs):

            noise = np.random.normal(0, noise_width, (self.batch_size, self.latent_dim))

            # ---------------------
            #  Train DAE
            # ---------------------

            # Select a random batch of images
            idx = np.random.randint(0, train_data_scaled.shape[0], self.batch_size)
            inputs = train_data_scaled[idx]

            a_loss = self.autoencoder.test_on_batch(inputs, inputs)

            # Generate a batch of new data
            gen_a_mse_data = self.generator.predict(noise)

            # Train the discriminator
            d_loss_real = self._train_on_batch_discriminator(inputs, valid)
            if not train_only_d_real:
                d_loss_fake = self.discriminator.train_on_batch(gen_a_mse_data, fake)
            else:
                d_loss_fake = self.discriminator.test_on_batch(gen_a_mse_data, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            if d_loss_real[1] < 0.99:
                train_only_d_real = True
            else:
                train_only_d_real = False

            # ---------------------
            #  Train Generator
            # ---------------------

            if not train_only_d_real or epoch < 100:
                # Train the generator (to have the discriminator label samples as valid)
                g_loss = self.combined.train_on_batch(noise, valid)
            else:
                # Test the generator (to have the discriminator label samples as valid)
                g_loss = self.combined.test_on_batch(noise, valid)

            # Store
            last_g_acc_s = np.delete(np.append(last_g_acc_s, g_loss[1]), 0)

            if not g_acc_started_from_zero:
                if round(g_loss[1],4) == 0.0:
                    g_acc_started_from_zero = True

            # Adjust the learning rate of the generator
            g_acc_approx = round(g_loss[1],1)
            changed_g_lr = self.g_lr_range[int(g_acc_approx*10)]
            K.set_value(self.combined.optimizer.lr, changed_g_lr)

            # Store the loss values for plot
            all_a_loss.append(a_loss[0])
            all_a_acc.append(100*a_loss[1])
            all_d_loss.append(d_loss[0])
            all_d_acc.append(100*d_loss[1])
            all_g_loss.append(g_loss[0])
            all_g_acc.append(100*g_loss[1])
            # Store the discriminator lr
            d_lr = float(K.get_value(self.discriminator.optimizer.lr))
            all_d_lr.append(d_lr)
            # Store the generator lr
            g_lr = float(K.get_value(self.combined.optimizer.lr))
            all_g_lr.append(g_lr)

            # Plot the progress
            print ("%d [DAE loss: %f, acc.: %.2f%%] [D loss: %f, acc.: %.2f%%, lr: %.8f] "
                   "[G loss: %f, acc.: %.2f%%, lr: %.8f]" %
                   (epoch, a_loss[0], 100*a_loss[1], d_loss[0], 100*d_loss[1], d_lr, g_loss[0], 100*g_loss[1], g_lr))

            # Stop if the stooping criteria is met
            if g_acc_started_from_zero and all(last_g_acc_s > 0.95):
                self.n_epochs = epoch + 1
                break

            # If the g_acc is still null after many epochs, increase th g_lr by factor 2
            if g_acc_started_from_zero and (epoch % max_epoch_for_helping_g == 0) and epoch > 0:
                if round(g_loss[1],4) == 0.0:
                    self.initial_g_lr *= 2
                    self.g_lr_range = np.linspace(self.initial_g_lr, self.stopping_g_lr, 11)
                    K.set_value(self.combined.optimizer.lr, self.initial_g_lr)
        print('--------------------------------------------------------')
        #==================================================================

        # Test the discriminator
        p_train_data = self._discriminator_predict(train_data_scaled)
        assert p_train_data.shape[0] == train_data_scaled.shape[0], ("The predicted data shape is not right!")

        # Test the generator
        noise = np.random.normal(0, 1, (self.batch_size, self.latent_dim))
        g_data = self.generator.predict(noise)
        assert g_data.shape == (self.batch_size, self.mse_size), ("The predicted data shape is not right!")
        #p_g_data = self._discriminator_predict(g_data)
        #assert p_g_data.shape[0] == g_data.shape[0], ("The predicted data shape is not right!")

        if save_model:
            # Save the scaler
            scaler_filename = self.model_dir_path + 'scaler_' + self.model_name + '.pkl'
            joblib.dump(scaler, scaler_filename)

            # Save weights to HDF5
            model_weights_filename = self.model_dir_path + self.model_name + '.h5'
            model_d_weights_filename = self.model_dir_path + self.model_d_name + '.h5'
            model_ae_weights_filename = self.model_dir_path + self.model_ae_name + '.h5'

            # Save weights
            self.combined.save_weights(model_weights_filename)
            self.discriminator.save_weights(model_d_weights_filename)
            self.autoencoder.save_weights(model_ae_weights_filename)

            print('Saved model to disk.')

        if print_and_plot_history:
            _x = range(self.n_epochs)
            # summarize loss
            fig1 = plt.figure()
            plt.plot(_x, all_a_loss, 'k-', _x, all_d_loss, 'b--', _x, all_g_loss, 'g.-')
            plt.title('Model learning loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['autoencoder', 'discriminator', 'generator'], loc='upper left')
            figure_name = self.model_dir_path + self.model_name + '_loss.pdf'
            fig1.savefig(figure_name)
            if show_plots:
                plt.show()
            # summarize discriminator accuracy
            fig2 = plt.figure()
            plt.plot(_x, all_a_acc, 'k-', _x, all_d_acc, 'b--', _x, all_g_acc, 'g.-')
            plt.title('Model learning accuracy')
            plt.ylabel('acc')
            plt.xlabel('epoch')
            plt.legend(['autoencoder', 'discriminator', 'generator'], loc='upper left')
            figure_name = self.model_dir_path + self.model_name + '_acc.pdf'
            fig2.savefig(figure_name)
            if show_plots:
                plt.show()
            # summarize discriminator accuracy
            fig3 = plt.figure()
            plt.plot(_x, all_d_lr, 'k-', _x, all_g_lr, 'b--')
            plt.title('Model learning rate')
            plt.ylabel('lr')
            plt.xlabel('epoch')
            plt.legend(['discriminator', 'generator'], loc='upper left')
            figure_name = self.model_dir_path + self.model_name + '_lr.pdf'
            fig3.savefig(figure_name)
            if show_plots:
                plt.show()

    def load_weights(self):
        """
        Load the weights from file to the model.
        :param model_dir_path:
        :param iteration_number:
        :return:
        """
        model_weights_filename = self.model_dir_path + self.model_name + '.h5'
        model_d_weights_filename = self.model_dir_path + self.model_d_name + '.h5'
        model_ae_weights_filename = self.model_dir_path + self.model_ae_name + '.h5'

        # load weights into new model
        self.autoencoder.load_weights(model_ae_weights_filename)
        self.discriminator.load_weights(model_d_weights_filename)
        self.combined.load_weights(model_weights_filename)
        print("Loaded weights from disk")

    def test_model(self,
                   test_data,
                   is_abnormal = False,
                   test_ae = False):
        """
        Apply the learned model to the test samples and get scores.
        :param test_data:
        :param model_dir_path:
        :param iteration_number:
        :return:
        """
        # Get the scaler
        scaler_filename = self.model_dir_path + 'scaler_' + self.model_name + '.pkl'
        scaler = joblib.load(scaler_filename)

        # Scale the test data
        test_data_scaled = scaler.transform(test_data)

        # evaluate
        if is_abnormal:
            labels = np.zeros((test_data_scaled.shape[0], 1))
        else:
            labels = np.ones((test_data_scaled.shape[0], 1))

        test_score = self._discriminator_evaluate(test_data_scaled, labels)

        if test_ae:
            # Test autoencoder standalone for comparison purpose
            ae_scores = self.autoencoder.evaluate(test_data_scaled,test_data_scaled)
            ae_global_mse = ae_scores[0]
            ae_predicted_test_data = self.autoencoder.predict(test_data_scaled)
            assert ae_predicted_test_data.shape == test_data_scaled.shape, ("The predicted data shape is not right!")
            ae_mse_per_sample = [mean_squared_error(test_data_scaled[i,:], ae_predicted_test_data[i,:])
                                 for i in range(test_data_scaled.shape[0])]

            return test_score[0], test_score[1], ae_global_mse, ae_mse_per_sample
        else:
            return test_score[0], test_score[1]

    def test_complete_model(self,
                            test_data,
                            is_abnormal = False,
                            test_ae = False,
                            result_dir_path = '',
                            classification_threshold = complete_trajectory_threshold):
        """
        Apply the learned model to the test samples and get scores.
        :param test_data: includes the object id
        :param model_dir_path:
        :param iteration_number:
        :return:
        """
        # Get the scaler
        scaler_filename = self.model_dir_path + 'scaler_' + self.model_name + '.pkl'
        scaler = joblib.load(scaler_filename)

        # Sort and extract the test data by object id
        object_id_vector = np.unique(test_data[:, 0])
        object_id_vector = np.sort(object_id_vector)

        acc_list = []
        ae_acc_list = []

        average_acc = False

        if test_ae:
            trained_model_summary_results_filename = result_dir_path + 'summary_results.csv'

            # Ref.: https://stackoverflow.com/questions/29451030/why-doesnt-np-genfromtxt-remove-header-while-importing-in-python
            with open(trained_model_summary_results_filename, 'r') as results:
                line = results.readline()
                header = [e for e in line.strip().split(',') if e]
                results_array = np.genfromtxt(results, names=header, dtype=None, delimiter=',')

            threshold_value = results_array['ae_threshold'][self.iteration_number]

            #average_acc = True

        for object_id in object_id_vector:
            object_test_data = test_data[test_data[:, 0] == object_id, :]

            # Remove the first column
            object_test_data = object_test_data[:, 1:]

            # Scale the test data
            object_test_data_scaled = scaler.transform(object_test_data)

            # evaluate
            if is_abnormal:
                labels = np.zeros((object_test_data_scaled.shape[0], 1))
            else:
                labels = np.ones((object_test_data_scaled.shape[0], 1))

            test_score = self._discriminator_evaluate(object_test_data_scaled, labels)

            # Store the accuracy
            acc_list.append(test_score[1])

            if test_ae:
                # Test autoencoder standalone for comparison purpose
                ae_predicted_test_data = self.autoencoder.predict(object_test_data_scaled)
                assert ae_predicted_test_data.shape == object_test_data_scaled.shape, \
                    ("The predicted data shape is not right!")
                ae_mse_per_sample = [mean_squared_error(object_test_data_scaled[i,:], ae_predicted_test_data[i,:])
                                     for i in range(object_test_data_scaled.shape[0])]

                if is_abnormal:
                    ae_acc = sum([s > threshold_value for s in ae_mse_per_sample])/float(len(ae_mse_per_sample))
                else:
                    ae_acc = sum([s <= threshold_value for s in ae_mse_per_sample])/float(len(ae_mse_per_sample))

                # Store the accuracy
                ae_acc_list.append(ae_acc)

        acc_list = np.array(acc_list)

        if average_acc:
            global_acc = np.mean(acc_list)
        else:
            if is_abnormal:
                global_acc = sum(acc_list >= classification_threshold) / float(len(acc_list))
            else:
                global_acc = sum((1 - acc_list) < classification_threshold) / float(len(acc_list))

        if test_ae:
            ae_acc_list = np.array(ae_acc_list)

            if average_acc:
                ae_global_acc = np.mean(ae_acc_list)
            else:
                if is_abnormal:
                    ae_global_acc = sum(ae_acc_list >= classification_threshold) / float(len(ae_acc_list))
                else:
                    ae_global_acc = sum((1 - ae_acc_list) < classification_threshold) / float(len(ae_acc_list))

            return global_acc, ae_global_acc
        else:
            return global_acc


def test_trained_model(test_data,
                       clf_name,
                       model_dir_path = '',
                       iteration_number = 0,
                       is_abnormal = False,
                       threshold_value = 0):
    """
    Test any model.
    :param test_data:
    :param clf_name:
    :param model_dir_path:
    :param iteration_number:
    :param is_abnormal:
    :return:
    """
    if clf_name == 'single_ae' or clf_name == 'deep_ae':
        _, mse_per_sample = test_trained_ae_model(test_data=test_data,
                                                  model_dir_path=model_dir_path,
                                                  iteration_number=iteration_number)
        if is_abnormal:
            test_ratio = sum([score > threshold_value for score in mse_per_sample])/float(len(mse_per_sample))
        else:
            test_ratio = sum([score <= threshold_value for score in mse_per_sample])/float(len(mse_per_sample))
    else:
        test_ratio, _ = test_trained_traditional_model(test_data=test_data,
                                                       clf_name=clf_name,
                                                       model_dir_path=model_dir_path,
                                                       iteration_number=iteration_number,
                                                       is_abnormal=is_abnormal)
    return test_ratio

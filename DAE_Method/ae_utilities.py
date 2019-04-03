import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.utils.testing import assert_array_almost_equal
from sklearn import svm
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.externals import joblib
from keras.layers import Input, Dense
from keras.models import Model
from keras.models import model_from_yaml
import matplotlib.pyplot as plt


repeat_number = 20

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

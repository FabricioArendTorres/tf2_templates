"""
An example for how to use the DVIB, including early stopping on validation data and slow decrease of the lambda.

"""
import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow import keras
from tensorflow.keras import layers

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import argparse
import pprint

from data_generator import generate_banana_data
from DVIB import *

tfd = tfp.distributions


def plot_train_history(train_history: dict) -> None:
    # plot train history
    fig, axs = plt.subplots(2, len(train_history) // 2 + 1 if len(train_history) % 2 else len(train_history) // 2,
                            figsize=(15, 3))
    for ax, (key, val) in zip(axs.reshape(-1), train_history.items()):
        ax.plot(val)
        ax.set_title(key)
    plt.suptitle("Metrics during training")
    plt.show()


def load_mnist(download=True):
    from tensorflow.python.keras.utils.data_utils import get_file
    origin_folder = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/'

    if download:
        path = get_file(
            "mnist.npz",
            origin=origin_folder + 'mnist.npz',
            file_hash=
            '731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1')
    else:
        path = "mnist.npz"
    with np.load(path, allow_pickle=True) as f:
        if download:
            np.savez("mnist.npz", **f)
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']

        return (x_train, y_train), (x_test, y_test)


def main():
    (X, Y), (X_test, Y_test) = load_mnist(download=True)
    # Rescale the images from [0,255] to the [0.0,1.0] range.
    X, X_test = X[..., np.newaxis] / 255.0, X_test[..., np.newaxis] / 255.0

    # pre-encoding to onehot not necessary anymore, one_hot encoding is done by our models "DVIB.target_transform(y)"
    # however we will need to pass the depth
    depth = len(np.unique(Y))
    # Y, Y_test = tf.one_hot(Y, depth=depth).numpy(), tf.one_hot(Y_test, depth=depth).numpy()

    # define the encoder and decoder for our DVIB
    units = 512
    activation = 'relu'
    input_shape = X.shape[1:]
    output_shape = (depth,)

    # encoder net will be split into two parts for mu & sd
    encoder_net = keras.Sequential([
        layers.InputLayer(input_shape),  # + np.prod(output_shape)),
        layers.Conv2D(filters=64, kernel_size=3, activation=activation),
        layers.MaxPool2D(),
        layers.Conv2D(filters=32, kernel_size=3, activation=activation),
        layers.MaxPool2D(),
        layers.Conv2D(filters=32, kernel_size=3, activation=activation),
        layers.Flatten()
    ])
    encoder_mu = keras.Sequential([
        layers.Dense(args.latent_dim)
    ])
    encoder_sd = tf.keras.layers.Dense(units=args.latent_dim,
                                       activation=tf.nn.softplus)

    decoder_mu = keras.Sequential(
        [
            layers.InputLayer((args.latent_dim,)),
            layers.Dense(units=units, activation=activation),
            layers.Dense(units=np.prod(output_shape)),
            layers.Reshape(output_shape)
        ]
    )

    ib = ClassificationDVIB(onehot_depth=depth,
                            latent_dim=args.latent_dim,
                            encoder_net=encoder_net,
                            encoder_mu=encoder_mu,
                            encoder_sd=encoder_sd,
                            decoder_mu=decoder_mu,
                            starting_lambda=20.)

    opt = keras.optimizers.Adam(learning_rate=1e-3)
    ib.compile(optimizer=opt, metrics=["ACC"])
    # callback for stopping when it gets worse on validation data
    # note that I just used the test set here out of laziness (instead of a proper separate validation data set)
    early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_expected_ll", patience=10, mode="max",
                                                      restore_best_weights=True)
    # callback for decreasing lambda each x epochs by a factor
    lambda_cb = LambdaCallback(decrease_lambda_each=1, lambda_factor=0.5)

    try:
        train_history: dict = ib.fit(x=X, y=Y, epochs=args.num_epochs,
                                     validation_data=(X_test, Y_test),
                                     validation_freq=3,
                                     callbacks=[early_stopping_cb, lambda_cb]
                                     ).history
    except KeyboardInterrupt:
        print("Manually interrupted training..")
        train_history: dict = ib.history.history

    # plot all the logged values
    plot_train_history(train_history)

    # evaluate all our metrics on a test set with our custom test_step function
    test_results = ib.evaluate(X_test, Y_test, return_dict=True)
    train_results = ib.evaluate(X, Y, return_dict=True)

    print("Test Results:")
    print(test_results)
    print("\nTrain Results:")
    print(train_results)

    # Predict on train and test set with our custom predict_step function.
    # The predict function is called batch-wise, so we can predict on large test sets without doing the loop manually
    y_hat_mu_train, y_hat_sd_train = ib.predict(X)
    y_hat_mu, y_hat_sd = ib.predict(X_test, batch_size=16)
    breakpoint()


if __name__ == "__main__":
    floatType = 'float32'

    parser = argparse.ArgumentParser(description="DVIB Demo")

    parser.add_argument("--mb-size", default=64, type=int)
    parser.add_argument("--latent-dim", default=5, type=int)
    parser.add_argument("--num-epochs", default=50, type=int)

    parser.add_argument("--seed", default=1234, type=int)

    args = parser.parse_args()
    pprint.pformat(vars(args))

    tf.keras.backend.set_floatx(floatType)
    tf.get_logger().setLevel('ERROR')
    sns.set()

    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    main()

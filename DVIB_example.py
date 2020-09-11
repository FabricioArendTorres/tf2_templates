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
from DVIB import DVIB, LambdaCallback

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


def main():
    (X, Y), (X_test, Y_test) = generate_banana_data(n_sqrt_train, n_sqrt_test, seed=args.seed)

    # create training and test data set from numpy arrays
    dataset = tf.data.Dataset.from_tensor_slices((X, Y)).shuffle(n_sqrt_train ** 2).batch(args.mb_size)
    dataset_test = tf.data.Dataset.from_tensor_slices((X_test, Y_test)).shuffle(n_sqrt_test ** 2).batch(args.mb_size)

    # define the encoder and decoder for our DVIB
    units = 50
    activation = 'relu'
    input_shape = X.shape[1:]
    output_shape = Y.shape[1:]

    encoder_net = keras.Sequential([
        layers.InputLayer(input_shape),  # + np.prod(output_shape)),
        layers.Dense(units=units, activation=activation),
        layers.Dense(units=units * 3, activation=activation)
    ])
    encoder_mu = keras.Sequential([
        layers.Dense(units),
        layers.Dense(args.latent_dim)
    ])
    encoder_sd = tf.keras.layers.Dense(units=args.latent_dim,
                                       activation=tf.nn.softplus)
    decoder_mu = keras.Sequential(
        [
            tf.keras.layers.InputLayer((args.latent_dim,)),
            tf.keras.layers.Dense(units=np.prod(output_shape)),
            tf.keras.layers.Reshape(output_shape)
        ]
    )

    ib = DVIB(latent_dim=args.latent_dim,
              encoder_net=encoder_net,
              encoder_mu=encoder_mu,
              encoder_sd=encoder_sd,
              decoder_mu=decoder_mu,
              starting_lambda=100.)

    opt = keras.optimizers.Adam(learning_rate=1e-3)
    ib.compile(optimizer=opt)

    # callback for stopping when it gets worse on validation data
    # note that I just used the test set here out of laziness (instead of a proper separate validation data set)
    early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_expected_ll", patience=10, mode="max",
                                                      restore_best_weights=True)
    # callback for decreasing lambda each x epochs by a factor
    lambda_cb = LambdaCallback(decrease_lambda_each=5, lambda_factor=0.95)

    try:
        train_history: dict = ib.fit(x=X, y=Y, epochs=args.num_epochs,
                                     validation_data=(X_test, Y_test),
                                     validation_freq=5,
                                     callbacks=[early_stopping_cb, lambda_cb]).history
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
    y_hat_mu, y_hat_sd = ib.predict(X_test)

    # plot outputs
    fig, axs = plt.subplots(1, 4)
    axs[0].imshow(Y.reshape((n_sqrt_train, n_sqrt_train)))
    axs[0].set_title("Train")
    axs[1].imshow(Y_test.reshape((n_sqrt_test, n_sqrt_test)))
    axs[1].set_title("Test")
    axs[2].imshow(y_hat_mu.reshape((n_sqrt_test, n_sqrt_test)))
    axs[2].set_title("Predicted")
    axs[3].imshow((y_hat_mu - Y_test).reshape((n_sqrt_test, n_sqrt_test)))
    axs[3].set_title("Error")
    for ax in axs:
        ax.set_axis_off()
    plt.show()


if __name__ == "__main__":
    # Parameters for our data set
    n_sqrt_train = 40
    n_sqrt_test = 60

    plt.rcParams['image.cmap'] = 'YlGnBu'
    floatType = 'float32'

    parser = argparse.ArgumentParser(description="DVIB Demo")

    parser.add_argument("--mb-size", default=400, type=int)
    parser.add_argument("--latent-dim", default=1, type=int)
    parser.add_argument("--num-epochs", default=500, type=int)

    parser.add_argument("--seed", default=1234, type=int)

    args = parser.parse_args()
    pprint.pformat(vars(args))

    tf.keras.backend.set_floatx(floatType)
    tf.get_logger().setLevel('ERROR')
    sns.set()

    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    main()

"""
Provides an example implementation of the Deep Variational Information Bottleneck [1].
Makes use of the Keras Subclass interface and the train_step / test_step / predict_step functions that can be
overwritten for a custom loss term.

Additionally tunes the lambda Hyperparameter, i.e. a weight term multiplied with the KL-divergence part of the loss.
For this, use the provided callback LambdaCallback.

For an example, see the file DVIB_example.

[1] Alemi, Alexander A., et al. "Deep variational information bottleneck." arXiv preprint arXiv:1612.00410 (2016).
"""
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from tensorflow.keras import layers

from typing import Tuple

tfd = tfp.distributions


class DVIB(keras.Model):
    """Basic Implementation of Deep Variational Information Bottleneck."""

    def __init__(self, latent_dim: int,
                 encoder_net: layers.Layer,
                 encoder_mu: layers.Layer,
                 encoder_sd: layers.Layer,
                 decoder_mu: layers.Layer,
                 starting_lambda: float = 100.
                 ) -> None:
        """
        Args:
            latent_dim (int):
            encoder_net (layers.Layer):
            encoder_mu (layers.Layer):
            encoder_sd (layers.Layer):
            decoder_mu (layers.Layer):
        """
        assert isinstance(starting_lambda, float)
        super().__init__(name="DVIB")

        self.latent_dim = latent_dim

        self.encoder_net = encoder_net

        self.encoder_mu = encoder_mu
        self.encoder_sd = encoder_sd
        self.decoder_mu = decoder_mu

        self.lambda_ = tf.Variable(starting_lambda, trainable=False)

        self.ll_tracker = keras.metrics.Mean(name="expected log-likelihood")
        self.kl_tracker = keras.metrics.Mean(name="kl-divergence")
        self.loss_tracker = keras.metrics.Mean(name="total loss")
        self.rmse_tracker = keras.metrics.RootMeanSquaredError(name="rmse")

        self.decoder_logsd = tf.Variable(0.0, trainable=True)

    def decoder_sd(self):
        """
        Wrapper for sd
        Returns:

        """
        return tf.exp(self.decoder_logsd + 1e-5)

    def q_z(self, x: tf.Tensor) -> tfd.MultivariateNormalDiag:
        """
        Args:
            x (tf.Tensor):
        """
        mean_branch, sd_branch = tf.split(self.encoder_net(x), num_or_size_splits=2, axis=1)

        scale_z = self.encoder_sd(sd_branch) + 1e-6  # [mb_size, latent_dim]
        mu_z = self.encoder_mu(mean_branch)  # [mb_size, latent_dim]
        encoded_posterior = tfd.MultivariateNormalDiag(mu_z, scale_z)

        return encoded_posterior

    def decode(self, z: tf.Tensor) -> tfd.Distribution:
        """
        Args:
            z (tf.Tensor):
        """
        mu_y_hat = self.decoder_mu(z)  # [sample_size, mb_size, 1]
        sigma_y_hat = self.decoder_sd()  # scalar

        pdf_y_hat = tfd.Normal(mu_y_hat, sigma_y_hat)

        # make Distribution object independent over y shape
        # -> sum over y dimension, mean over batch dimension, mean over n_samples
        #
        # reinterpreted_batch_ndims says which right-most dims are regarded as the event-size (i.e. the y shape)
        # the remaining are regarded as the 'batch' shape.
        pdf_y_hat = tfd.Independent(pdf_y_hat, reinterpreted_batch_ndims=1)
        # batch_shape=[n_samples, mb_size] event_shape=[output_dim]
        return pdf_y_hat

    def call(self, inputs, training=None, mask=None) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Args:
            inputs:
            training:
            mask:
        """
        x = inputs
        z_posterior = self.q_z(x)
        pdf_y_hat = self.decode(z=z_posterior.mean())
        return pdf_y_hat.mean(), pdf_y_hat.stddev()

    def train_step(self, data: Tuple[tf.Tensor, tf.Tensor]):
        """
        Args:
            data:
        """
        x, y = data

        with tf.GradientTape() as tape:
            z_prior = self.build_prior_z()
            z_posterior = self.q_z(x)

            pdf_y_hat = self.decode(z=z_posterior.sample())

            # Expected Log Likelihood / Reconstruction error
            exp_ll = tf.reduce_sum(pdf_y_hat.log_prob(y))  # [sample_size, mb_size] -> ()
            # KL Divergence
            kl_div = z_posterior.kl_divergence(z_prior)
            # Combined Loss, i.e. negative ELBO
            total_loss = -exp_ll + self.lambda_ * kl_div

        # update weights
        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # track losses
        self.ll_tracker.update_state(exp_ll)
        self.kl_tracker.update_state(kl_div)
        self.loss_tracker.update_state(total_loss)
        return {"loss": self.loss_tracker.result(),
                "expected_ll": self.ll_tracker.result(),
                "KL-div": self.kl_tracker.result(),
                "lambda": self.lambda_}

    def predict_step(self, data):
        x = data
        return self(x, training=False)

    def test_step(self, data):
        x, y = data

        z_prior = self.build_prior_z()
        z_posterior = self.q_z(x)

        pdf_y_hat = self.decode(z=z_posterior.sample())

        # Expected Log Likelihood / Reconstruction error
        exp_ll = tf.reduce_sum(pdf_y_hat.log_prob(y))  # [sample_size, mb_size] -> ()
        # KL Divergence
        kl_div = z_posterior.kl_divergence(z_prior)
        # Combined Loss, i.e. negative ELBO
        total_loss = -exp_ll + self.lambda_ * kl_div

        mean_prediction = self.decode(z_posterior.mean()).mean()

        # track losses
        self.ll_tracker.update_state(exp_ll)
        self.kl_tracker.update_state(kl_div)
        self.loss_tracker.update_state(total_loss)
        self.rmse_tracker.update_state(y_true=y, y_pred=mean_prediction)
        return {"loss": self.loss_tracker.result(),
                "expected_ll": self.ll_tracker.result(),
                "KL-div": self.kl_tracker.result(),
                "rmse": self.rmse_tracker.result()}

    def build_prior_z(self) -> tfd.MultivariateNormalDiag:
        """Returns: tfd.MultivariateNormalDiag"""
        mu = tf.zeros(self.latent_dim)
        rho = tf.ones(self.latent_dim)
        return tfd.MultivariateNormalDiag(mu, rho)


class LambdaCallback(keras.callbacks.Callback):
    def __init__(self, decrease_lambda_each: int = 5,
                 lambda_factor=0.9):
        """
            Callback class for lambda decrease.
        Args:
            decrease_lambda_each:
            lambda_factor:
        """
        super().__init__()
        self.decrease_lambda_each = decrease_lambda_each
        self.lambda_factor = lambda_factor

    def on_epoch_end(self, epoch, logs=None):
        self.model: DVIB
        if epoch % self.decrease_lambda_each == 0:
            self.model.lambda_.assign(self.model.lambda_ * self.lambda_factor)

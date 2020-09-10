import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from tensorflow.keras import layers

from typing import Tuple, Callable

tfd = tfp.distributions


class DVIB(keras.Model):
    """Basic Implementation of Deep Variational Information Bottleneck."""

    def __init__(self, latent_dim: int,
                 encoder_net: layers.Layer,
                 encoder_mu: layers.Layer,
                 encoder_sd: layers.Layer,
                 decoder_mu: layers.Layer
                 ) -> None:
        """
        Args:
            latent_dim (int):
            encoder_net (layers.Layer):
            encoder_mu (layers.Layer):
            encoder_sd (layers.Layer):
            decoder_mu (layers.Layer):
        """
        super().__init__(name="DVIB")

        self.latent_dim = latent_dim

        self.encoder_net = encoder_net

        self.encoder_mu = encoder_mu
        self.encoder_sd = encoder_sd
        self.decoder_mu = decoder_mu

        self.lambda_ = tf.Variable(1.0, trainable=False)

        self.ll_tracker = keras.metrics.Mean(name="expected log-likelihood")
        self.kl_tracker = keras.metrics.Mean(name="kl-divergence")
        self.loss_tracker = keras.metrics.Mean(name="total loss")

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

    def call(self, inputs, training=None, mask=None):
        """
        Args:
            inputs:
            training:
            mask:
        """
        x = inputs
        z_posterior = self.q_z(x)
        pdf_y_hat = self.decode(z=z_posterior.mean())
        return pdf_y_hat.mean()

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

    def build_prior_z(self) -> tfd.MultivariateNormalDiag:
        """Returns: tfd.MultivariateNormalDiag"""
        mu = tf.zeros(self.latent_dim)
        rho = tf.ones(self.latent_dim)
        return tfd.MultivariateNormalDiag(mu, rho)

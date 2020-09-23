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
from tensorflow import keras, Variable
from tensorflow.keras import layers
from collections import namedtuple
from typing import Tuple
from abc import abstractmethod, ABC

tfd = tfp.distributions


class DVIB(keras.Model, ABC):
    """
    Basic Abstract Class Implementation of Deep Variational Information Bottleneck.
    Requires subclass for defining the used likelihood.
    """
    Loss = namedtuple("Loss", ['expected_ll', 'kl_div', 'total'])

    def __init__(self, latent_dim: int,
                 encoder_net: layers.Layer,
                 encoder_mu: layers.Layer,
                 encoder_sd: layers.Layer,
                 decoder_mu: layers.Layer,
                 starting_lambda: float = 100.,
                 num_train_samples: int = 1,
                 num_predict_samples: int = 100,
                 num_evaluate_samples: int = 30,
                 *args,
                 **kwargs
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
        super().__init__(*args, **kwargs, name="DVIB")

        self.latent_dim = latent_dim

        # Neural Networks for encoder and decoder
        self.encoder_net = encoder_net
        self.encoder_mu = encoder_mu
        self.encoder_sd = encoder_sd
        self.decoder_mu = decoder_mu

        # The Lagrange parameter which weights the KL Loss.
        # Can and should be slowly decreased during training.
        self.lambda_ = tf.Variable(starting_lambda, trainable=False)

        # Our custom metrics for the IB
        self.ll_tracker = keras.metrics.Mean(name="expected_ll")
        self.kl_tracker = keras.metrics.Mean(name="kl_div")
        self.loss_tracker = keras.metrics.Mean(name="total")

        # how many samples will be used for training/prediction/evaluation..
        self.num_train_samples = num_train_samples
        self.num_predict_samples = num_predict_samples
        self.num_evaluate_samples = num_evaluate_samples

    def prior_z(self) -> tfd.MultivariateNormalDiag:
        """Returns: tfd.MultivariateNormalDiag"""
        mu = tf.zeros(self.latent_dim)
        rho = tf.ones(self.latent_dim)
        return tfd.MultivariateNormalDiag(mu, rho)

    def q_z(self, x: tf.Tensor) -> tfd.MultivariateNormalDiag:
        """
        Args:
            x (tf.Tensor):
        """
        mean_branch, sd_branch = tf.split(self.encoder_net(x), num_or_size_splits=2, axis=-1)

        scale_z = self.encoder_sd(sd_branch) + 1e-6  # [mb_size, latent_dim]
        mu_z = self.encoder_mu(mean_branch)  # [mb_size, latent_dim]
        encoded_posterior = tfd.MultivariateNormalDiag(mu_z, scale_z)

        return encoded_posterior

    def decode(self, z: tf.Tensor) -> tfd.Distribution:
        """
        Args:
            z (tf.Tensor):
        Returns pdf_y_hat
        """
        mu_y_hat = tf.map_fn(self.decoder_mu, z)  # [sample_size, mb_size, 1]
        pdf_y_hat = self.generate_likelihood(mu_y_hat)

        # make Distribution object independent over y shape
        # -> sum over y dimension, mean over batch dimension, mean over n_samples
        #
        # reinterpreted_batch_ndims says which right-most dims are regarded as the event-size (i.e. the y shape)
        # the remaining are regarded as the 'batch' shape.
        pdf_y_hat = tfd.Independent(pdf_y_hat, reinterpreted_batch_ndims=1)
        # batch_shape=[n_samples, mb_size] event_shape=[output_dim]
        return pdf_y_hat

    def calc_losses(self, x, y, num_samples=1):
        z_prior = self.prior_z()
        z_posterior = self.q_z(x)

        pdf_y_hat = self.decode(z=z_posterior.sample(num_samples))
        # Expected Log Likelihood / Reconstruction error
        exp_ll = tf.reduce_sum(pdf_y_hat.log_prob(y))  # [sample_size, mb_size] -> ()
        # KL Divergence
        kl_div = z_posterior.kl_divergence(z_prior)
        # Combined Loss, i.e. negative ELBO
        total_loss = -exp_ll + self.lambda_ * kl_div

        losses = self.Loss(expected_ll=exp_ll,
                           kl_div=kl_div,
                           total=total_loss)
        return losses, pdf_y_hat

    def train_step(self, data: Tuple[tf.Tensor, tf.Tensor]):
        """
        Args:
            data:
        """
        x, y = data
        y_transformed = self.target_transform(y)

        with tf.GradientTape() as tape:
            losses, pdf_y_hat = self.calc_losses(x, y_transformed, 1)

        # the actual prediction for our target
        y_pred = self._predict_transform(tf.reduce_mean(pdf_y_hat.mean(), axis=0))

        # update weights
        gradients = tape.gradient(losses.total, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # track losses
        self.compiled_metrics.update_state(y, y_pred)  # TODO
        loss_dict = self.update_custom_metrics(losses)
        loss_dict.update({m.name: m.result() for m in self.metrics})

        return loss_dict

    def test_step(self, data):
        x, y = data
        y_transformed = self.target_transform(y)

        # calculate losses
        losses, pdf_y_hat = self.calc_losses(x, y_transformed, 30)
        # the actual prediction for our target
        y_pred = self._predict_transform(tf.reduce_mean(pdf_y_hat.mean(), axis=0))

        # track losses
        self.compiled_metrics.update_state(y, y_pred)  # TODO
        loss_dict = self.update_custom_metrics(losses)
        loss_dict.update({m.name: m.result() for m in self.metrics})

        return loss_dict

    def call(self, inputs, training=None, mask=None) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Args:
            inputs:
            training:
            mask:
        Returns: pdf_y_hat.mean(), pdf_y_hat.stddev()
        """
        x = inputs
        z_posterior = self.q_z(x)
        pdf_y_hat = self.decode(z=z_posterior.mean()[tf.newaxis, ...])
        return pdf_y_hat.mean(), pdf_y_hat.stddev()

    def predict_step(self, data):
        x = data
        return self(x, training=False)

    def update_custom_metrics(self, losses: Loss) -> dict:
        """
        Update the all the IB Metrics during training or evaluation.
        Args:
            losses:

        Returns:

        """
        self.ll_tracker.update_state(losses.expected_ll)
        self.kl_tracker.update_state(losses.kl_div)
        self.loss_tracker.update_state(losses.total)

        loss_dict = losses._asdict()
        loss_dict.update({"lambda": self.lambda_})
        return loss_dict

    @abstractmethod
    def generate_likelihood(self, loc: tf.Tensor) -> tfd.Distribution:
        """
        Return a likelihood of which the loc is parameterized by the decoder_mu network.
        e.g. return tfd.Normal(loc, self.decoder_sd()) for regression
        Args:
            loc:

        Returns:

        """
        raise NotImplementedError("You have to reimplement this..")

    def target_transform(self, y: tf.Tensor):
        """
        Could include e.g. argmax for classification.
        Usually just the identity.
        Args:
            y:

        Returns:

        """
        return y

    @staticmethod
    def _predict_transform(mean_prediction):
        """
        Transformation of mean_over_samples(decoder( encoder(x).sample() )) , which is then used for the prediction.
        Usually just the identity, can be argmax for Classification.

        Args:
            mean_prediction:

        Returns:

        """
        return mean_prediction


class ClassificationDVIB(DVIB):
    def __init__(self, onehot_depth: int, *args, **kwargs):
        """

        Args:
            onehot_depth: depth for at-runtime onehot encoding of targets,
            i.e. 10 for MNIST
            *args:
            **kwargs:
        """
        super().__init__(*args, **kwargs)
        self.decoder_logsd = tf.Variable(0.0, trainable=True)
        self.onehot_depth = onehot_depth

    def target_transform(self, y):
        """

        Returns:

        """
        return tf.one_hot(y, depth=self.onehot_depth)

    def generate_likelihood(self, loc: tf.Tensor) -> tfd.Distribution:
        """
        Use Bernoulli Likelihood
        Args:
            loc:

        Returns:

        """
        return tfd.Bernoulli(loc)

    @staticmethod
    def _predict_transform(mean_prediction):
        """
        Transformation for the mean over the decoded samples, which is then used for the prediction.
        Can be argmax for Classification, just the identity for Regression or something else.

        Args:
            mean_prediction:

        Returns:

        """
        return tf.argmax(mean_prediction, -1)


class RegressionDVIB(DVIB):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.decoder_logsd = tf.Variable(0.0, trainable=True)

    def decoder_sd(self) -> tf.Tensor:
        """
        Wrapper for sd
        Returns:
        """
        return tf.exp(self.decoder_logsd + 1e-5)

    def generate_likelihood(self, loc: tf.Tensor) -> tfd.Distribution:
        """
        Use Gaussian Likelihood with scalar scale.
        Args:
            loc:

        Returns:

        """
        return tfd.Normal(loc, self.decoder_sd())


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

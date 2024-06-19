from typing import Optional, Any, Callable

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from flax import linen as nn

from scvi import REGISTRY_KEYS
from scvi._types import Tunable
from scvi.distributions import JaxNegativeBinomialMeanDisp as NegativeBinomial
from scvi.module.base import JaxBaseModuleClass, LossOutput, flax_configure

from .layers import BottomUpLayer, TopDownLayer, Dense
# from .hsic import compute_HSIC


class JaxEncoder(nn.Module):
    """Encoder for Jax VAE. Inference -> outputs changes to posterior from prior"""

    n_input: int
    n_bg_latent: int
    n_salient_latent: int
    n_hidden: int
    n_layer: int
    dropout_rate: int
    training: Optional[bool] = None

    def setup(self):
        """Setup encoder"""
        # self.bottom_up_layer_bg = BottomUpLayer(
        #     n_hidden=self.n_hidden,
        #     n_output=self.n_bg_latent,
        #     dropout_rate=self.dropout_rate,
        #     n_layer=self.n_layer + 1,
        # )
        self.bottom_up_layer_bg = [
            BottomUpLayer(
                self.n_hidden,
                self.n_hidden,
                dropout_rate=self.dropout_rate,
                n_layer=1,
            )
            for _ in range(self.n_layer)
        ]
        self.top_down_layer_bg = [
            TopDownLayer(
                self.n_hidden, self.n_bg_latent, dropout_rate=self.dropout_rate
            )
            for _ in range(self.n_layer)
        ]

        self.bottom_up_layer_salient = BottomUpLayer(
            self.n_hidden, self.n_hidden, dropout_rate=self.dropout_rate, n_layer=1
        )
        self.top_down_layer_salient = TopDownLayer(
            self.n_hidden, self.n_salient_latent, dropout_rate=self.dropout_rate
        )

    def __call__(
        self, x: jnp.ndarray, design: jnp.ndarray, training: Optional[bool] = None
    ):
        """Forward pass."""
        training = nn.merge_param("training", self.training, training)
        is_eval = not training
        x = jnp.log1p(x)
        # y = self.dense(design)
        pz = []
        qz = []
        z_bg = None

        # bottom_up = self.bottom_up_layer_bg(x, training=is_eval)
        for i in range(self.n_layer):
            bottom_up = self.bottom_up_layer_bg[i](x, training=is_eval)
            z_bg, pz_bg, qz_bg = self.top_down_layer_bg[i](
                z_bg, None, bottom_up[0], training=is_eval
            )

            # #Skip connections
            # if z_bg is None:
            #     z_bg = z_bg_
            # else:
            #     z_bg = z_bg + z_bg_
            pz.append(pz_bg)
            qz.append(qz_bg)

        bottom_up_salient = self.bottom_up_layer_salient(x, training=is_eval)

        z_salient, pz_salient, qz_salient = self.top_down_layer_salient(
            z_bg,
            design,
            bottom_up_salient[0],
            training=is_eval,
        )

        pz.append(pz_salient)
        qz.append(qz_salient)

        return z_bg, z_salient, pz, qz  # , pz_salient, qz_salient


class FlaxDecoder(nn.Module):
    """Decoder for Jax VAE."""

    n_input: int
    dropout_rate: float
    n_hidden: int
    training: Optional[bool] = None

    def setup(self):
        """Setup decoder."""
        self.dense1 = Dense(self.n_hidden)
        self.dense2 = Dense(self.n_hidden)
        self.dense3 = Dense(self.n_input)

        self.layernorm = nn.LayerNorm()
        # self.batchnorm2 = nn.BatchNorm(momentum=0.9)
        self.dropout = nn.Dropout(self.dropout_rate)

        self.disp = self.param(
            "disp", lambda rng, shape: jax.random.normal(rng, shape), (self.n_input, 1)
        )

    def __call__(
        self, z: jnp.ndarray, batch: jnp.ndarray, training: Optional[bool] = None
    ):
        """Forward pass."""
        # TODO(adamgayoso): Test this
        # training = True
        training = nn.merge_param("training", self.training, training)
        # is_eval = not training

        # z1, z2 = jnp.split(z, 2, axis = 1)
        # z2 = self.dropout(z2, deterministic=is_eval)
        # z = jnp.hstack([z1, z2])
        h = self.dense1(z)
        batch = self.dense2(batch)
        h += batch
        h = self.layernorm(h)
        h = nn.relu(h)
        # # # h = self.dropout1(h, deterministic=is_eval)
        # # h = self.dense3(h)
        # # # skip connection
        # # h += self.dense4(batch)
        # # h = self.batchnorm2(h)
        # # h = nn.relu(h)
        # # h = /self.dropout2(h, deterministic=is_eval)
        h = self.dense3(h)
        return h, self.disp.ravel()


@flax_configure
class JaxVAE(JaxBaseModuleClass):
    """Variational autoencoder model."""

    n_input: int
    n_batch: int
    n_cats_per_cov: Optional[list]
    n_hidden: Tunable[int] = 128
    n_bg_latent: Tunable[int] = 10
    n_salient_latent: Tunable[int] = 10
    dropout_rate: Tunable[float] = 0.1
    n_layer: Tunable[int] = 1
    gene_likelihood: Tunable[str] = "nb"
    eps: Tunable[float] = 1e-8
    training: bool = True

    def setup(self):
        """Setup model."""

        self.encoder = JaxEncoder(
            n_input=self.n_input,
            n_bg_latent=self.n_bg_latent,
            n_salient_latent=self.n_salient_latent,
            n_hidden=self.n_hidden,
            n_layer=self.n_layer,
            dropout_rate=self.dropout_rate,
        )

        self.decoder = FlaxDecoder(
            n_input=self.n_input,
            dropout_rate=0.0,
            n_hidden=self.n_hidden,
        )

    @property
    def required_rngs(self):
        return ("params", "dropout", "z")

    def _get_inference_input(self, tensors: dict[str, jnp.ndarray]):
        """Get input for inference."""
        x = tensors[REGISTRY_KEYS.X_KEY]
        design = tensors[REGISTRY_KEYS.CONT_COVS_KEY]

        input_dict = {"x": x, "design": design}
        return input_dict

    def inference(
        self, x: jnp.ndarray, design: jnp.ndarray, n_samples: int = 1
    ) -> dict:
        """Run inference model."""
        # print(design.shape)
        # what should appear here?
        z_bg, z_salient, pz, qz = self.encoder(x, design, training=self.training)
        # z_salient = jnp.zeros_like(z_salient)
        # z_salient = z_salient * jnp.expand_dims((design.sum(axis=1) >= 1), -1)
        z = jnp.hstack([z_bg, z_salient])

        return {"qz": qz, "pz": pz, "z": z}

    def _get_generative_input(
        self,
        tensors: dict[str, jnp.ndarray],
        inference_outputs: dict[str, jnp.ndarray],
    ):
        """Get input for generative model."""
        x = tensors[REGISTRY_KEYS.X_KEY]
        z = inference_outputs["z"]
        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]
        design = tensors[REGISTRY_KEYS.CONT_COVS_KEY]
        input_dict = {
            "x": x,
            "z": z,
            "batch_index": batch_index,
            "design": design,
        }
        return input_dict

    def generative(self, x, z, batch_index, design) -> dict:
        """Run generative model."""
        # one hot adds an extra dimension

        batch = jax.nn.one_hot(batch_index, self.n_batch).squeeze(-2)
        # z = jnp.tile(z, (self.len_latent, 1, 1))
        # mask = self.create_mask(z.shape, n_latents=self.n_latent)
        # z = z * mask
        rho_unnorm, disp = self.decoder(z, batch, training=self.training)
        # rho_unnorm, disp = self.vmap_decoder(z, batch, training=self.training)
        disp_ = jnp.exp(disp)
        rho = jax.nn.softmax(rho_unnorm, axis=-1)
        total_count = x.sum(-1)[:, jnp.newaxis]
        mu = total_count * rho

        if self.gene_likelihood == "nb":
            disp_ = jnp.exp(disp)
            px = NegativeBinomial(mean=mu, inverse_dispersion=disp_)
        else:
            px = dist.Poisson(mu)

        return {"px": px, "rho": rho}

    def loss(
        self,
        tensors,
        inference_outputs,
        generative_outputs,
        kl_weight: float = 1.0,
    ):
        """Compute loss."""
        # kl_weight = kl_weight / 10
        x = tensors[REGISTRY_KEYS.X_KEY]
        px = generative_outputs["px"]
        pz = inference_outputs["pz"]
        qz = inference_outputs["qz"]
        reconst_loss = -px.log_prob(x).sum(-1)

        # qzs_m = qz[-1].mean
        # qzbg_m = qz[-2].mean
        # HSIC = compute_HSIC(qzs_m, qzbg_m)
        # kl_weight = 1e-6
        # kl_weight = 1
        # reconst_loss = reconst_loss.mean(axis=0)
        # kl_divergence_z = jnp.zeros(reconst_loss.shape[0])
        # loc = qz.loc
        # scale = qz.scale
        # for latent in self.n_latent:
        #     kl_divergence_z += dist.kl_divergence(
        #         dist.Normal(loc[:, :latent], scale[:, :latent]), dist.Normal(0, 1)
        #     ).sum(-1)
        # kl_divergence_z = (qz.log_prob(z) - dist.Laplace(0, 1).log_prob(z)).sum(-1)
        kl_divergence_z = 0
        for i in range(len(pz)):
            kl_divergence_z += dist.kl_divergence(qz[i], pz[i]).sum(-1)

        # kl_divergence_z = kl_divergence_z + 10 * HSIC
        kl_local_for_warmup = kl_divergence_z
        weighted_kl_local = kl_weight * kl_local_for_warmup
        loss = jnp.mean(reconst_loss + weighted_kl_local)
        kl_local = kl_divergence_z
        return LossOutput(
            loss=loss, reconstruction_loss=reconst_loss, kl_local=kl_local
        )

    def create_mask(self, shape, n_latents):
        num_items = len(n_latents)
        mask = jnp.ones(shape, dtype=bool)
        for i in range(num_items):
            mask = mask.at[i, :, n_latents[i] :].set(False)
        return mask

    def get_jit_generative_fn(
        self,
        get_generative_input_kwargs: dict[str, Any] | None = None,
        generative_kwargs: dict[str, Any] | None = None,
    ) -> Callable[
        [dict[str, jnp.ndarray], dict[str, jnp.ndarray]], dict[str, jnp.ndarray]
    ]:
        """Create a method to run inference using the bound module.

        Parameters
        ----------
        get_generative_input_kwargs
            Keyword arguments to pass to subclass `_get_generative_input`
        generative_kwargs
            Keyword arguments  for subclass `generative` method

        Returns
        -------
        A callable taking rngs and array_dict as input and returning the output
        of the `inference` method. This callable runs `_get_inference_input`.
        """
        vars_in = {"params": self.params, **self.state}
        get_generative_input_kwargs = _get_dict_if_none(get_generative_input_kwargs)
        generative_kwargs = _get_dict_if_none(generative_kwargs)

        @jax.jit
        def _run_generative(rngs, array_dict, inference_outputs):
            module = self.clone()
            generative_input = module._get_generative_input(
                array_dict, inference_outputs
            )
            out = module.apply(
                vars_in,
                rngs=rngs,
                method=module.generative,
                **generative_input,
                **generative_kwargs,
            )
            return out

        return _run_generative


def _get_dict_if_none(param):
    param = {} if not isinstance(param, dict) else param

    return param

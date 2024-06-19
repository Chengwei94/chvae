
import jax.numpy as jnp
import numpyro.distributions as dist
from flax import linen as nn
from flax.linen.initializers import variance_scaling


class Dense(nn.Dense):
    """Jax dense layer."""

    def __init__(self, *args, **kwargs):
        # scale set to reimplement pytorch init
        scale = 1 / 3
        kernel_init = variance_scaling(scale, "fan_in", "uniform")
        # bias init can't see input shape so don't include here
        kwargs.update({"kernel_init": kernel_init})
        super().__init__(*args, **kwargs)


class MLPLayer(nn.Module):
    """A simple MLP layer"""

    n_hidden: int
    n_output: int
    dropout_rate: float
    normalisation: str = "layernorm"
    end_dense: bool = False

    def setup(self):
        self.dense = Dense(self.n_hidden)
        if self.normalisation == "layernorm":
            self.norm = nn.LayerNorm()
        else:
            self.norm = nn.BatchNorm(momentum=0.9)

        self.dropout = nn.Dropout(self.dropout_rate)
        if self.end_dense:
            self.dense1 = Dense(self.n_output)

    def __call__(self, x, training: bool):
        # y = x
        x = self.dense(x)
        x = self.dropout(x, deterministic=training)
        if self.normalisation == "layernorm":
            x = self.norm(x)
        else:
            x = self.norm(x, use_running_average=training)
        x = nn.relu(x)
        if self.end_dense:
            x = self.dense1(x)
        # if y.shape == x.shape:
        #     x = y + x
        return x


class BottomUpLayer(nn.Module):
    """Returns changes to the posterior distributions"""

    n_hidden: int
    n_output: int
    dropout_rate: float
    n_layer: int = 1

    def setup(self):
        """Setup encoder."""
        self.layers = [
            MLPLayer(
                n_hidden=self.n_hidden,
                n_output=self.n_output,
                dropout_rate=self.dropout_rate,
                end_dense=False,
            )
            for _ in range(self.n_layer)
        ]
        # self.layers1 = [
        #     MLPLayer(n_hidden=self.n_hidden, n_output=self.n_output, dropout_rate=self.dropout_rate, end_dense=False)
        #     for _ in range(self.n_layer)
        # ]
        self.denses = [Dense(self.n_hidden) for _ in range(self.n_layer)]

    def __call__(self, x, training: bool):
        ys = []
        for i in range(self.n_layer):
            x = self.layers[i](x, training=training)
            # x = self.layers1[i](x, training=training)
            ys.append(x)
        # bottom_up_std = jnp.sqrt(jnp.exp(bottom_up_log_var))
        return ys


class TopDownLayer(nn.Module):
    """Generative path. Takes in posterior from
    previous layer and outputs next layer prior and posterior"""

    n_hidden: int
    n_output: int
    dropout_rate: float

    def setup(self):
        # self.prior_layer = Dense(self.n_output * 2)
        self.posterior_layer = Dense(self.n_output * 2)
        self.prior_layer = MLPLayer(
            n_hidden=self.n_hidden,
            n_output=self.n_output * 2,
            dropout_rate=self.dropout_rate,
            end_dense=True,
        )
        # self.posterior_layer = MLPLayer(
        #     n_hidden=self.n_hidden, n_output=self.n_output * 2, dropout_rate=self.dropout_rate, end_dense=True
        # )
        self.dense = MLPLayer(
            n_hidden=self.n_hidden,
            n_output=self.n_hidden,
            dropout_rate=self.dropout_rate,
        )
        # self.dense = Dense(self.n_hidden)
        # self.posterior_hidden_layer = ML`PLayer(self.n_hidden, self.dropout_rate)
        # self.prior_layer = Dense(self.n_output * 2)
        # self.posterior_layer = Dense(self.n_output * 2)
        self.prior_design_layer = Dense(self.n_output, use_bias=False)
        self.posterior_design_layer = Dense(self.n_output)

    def __call__(self, z, design, bottom_up, training: bool):
        if z is not None:
            pz_vals = self.prior_layer(z, training=training)
            pz_mean, pz_log_var = jnp.split(pz_vals, 2, axis=1)
            pz_log_var = jnp.clip(pz_log_var, -7, 7)
            pz_std = jnp.sqrt(jnp.exp(pz_log_var))
            if design is not None:
                # pass
                pz_mean = pz_mean * self.prior_design_layer(design)
                pz_std = pz_std * jnp.expand_dims(
                    (design.sum(axis=1) >= 1), -1
                )  # + 1e-1
                # Add 0.01 so that its not KL(0, 0)
                pz_std = pz_std + (
                    jnp.expand_dims((design.sum(axis=1) <= 1), -1) * 5e-2
                )
            pz = dist.Normal(pz_mean, pz_std)
        else:
            pz_mean = 0
            pz_std = 1
            pz = dist.Normal(pz_mean, pz_std)

        if z is not None:
            z = self.dense(z, training=training)
            qz_vals = self.posterior_layer(jnp.hstack([bottom_up, z]))
        else:
            qz_vals = self.posterior_layer(bottom_up)

        qz_mean, qz_log_var = jnp.split(qz_vals, 2, axis=1)
        qz_log_var = jnp.clip(qz_log_var, -7, 7)
        qz_std = jnp.sqrt(jnp.exp(qz_log_var))

        if design is not None:
            # pass
            posterior_design = self.posterior_design_layer(design)
            qz_mean = qz_mean * (posterior_design)

        qz = dist.Normal(qz_mean, qz_std)
        z_rng = self.make_rng("z_rng")
        z = qz.rsample(z_rng)

        return z, pz, qz

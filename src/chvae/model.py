import logging
from collections.abc import Sequence
from typing import Literal, Optional

import jax.numpy as jnp
import numpy as np
from anndata import AnnData

from scvi import REGISTRY_KEYS
from scvi.data import AnnDataManager
from scvi.data.fields import CategoricalObsField, LayerField, NumericalJointObsField
from .module import JaxVAE
from scvi.utils import setup_anndata_dsp

from scvi.model.base import BaseModelClass, JaxTrainingMixin

logger = logging.getLogger(__name__)


class JaxSCVI(JaxTrainingMixin, BaseModelClass):
    """``EXPERIMENTAL`` single-cell Variational Inference :cite:p:`Lopez18`, but with a Jax backend.

    This implementation is in a very experimental state. API is completely subject to change.

    Parameters
    ----------
    adata
        AnnData object that has been registered via :meth:`~scvi.model.JaxSCVI.setup_anndata`.
    n_hidden
        Number of nodes per hidden layer.
    n_bg_latent
        Dimensionality of the background latent space.
    n_salient_latent
        Dimensionality of the salient latent space.
    dropout_rate
        Dropout rate for neural networks.
    n_layer
        Number of heirarchical layers for the background latent space.
    gene_likelihood
        One of:

        * ``'nb'`` - Negative binomial distribution
        * ``'poisson'`` - Poisson distribution
    **model_kwargs
        Keyword args for :class:`~scvi.module.JaxVAE`

    Examples
    --------
    >>> adata = anndata.read_h5ad(path_to_anndata)
    >>> scvi.model.JaxSCVI.setup_anndata(adata, batch_key="batch")
    >>> vae = scvi.model.JaxSCVI(adata)
    >>> vae.train()
    >>> adata.obsm["X_scVI"] = vae.get_latent_representation()
    """

    _module_cls = JaxVAE

    def __init__(
        self,
        adata: AnnData,
        n_hidden: int = 128,
        n_bg_latent: int = 10,
        n_salient_latent: int = 10,
        dropout_rate: float = 0.1,
        n_layer: int = 1,
        gene_likelihood: Literal["nb", "poisson"] = "nb",
        **model_kwargs,
    ):
        super().__init__(adata)

        n_batch = self.summary_stats.n_batch
        n_cats_per_cov = self.summary_stats.get("n_extra_continuous_covs", 0)

        self.module = self._module_cls(
            n_input=self.summary_stats.n_vars,
            n_batch=n_batch,
            n_cats_per_cov=n_cats_per_cov,
            n_hidden=n_hidden,
            n_bg_latent=n_bg_latent,
            n_salient_latent=n_salient_latent,
            dropout_rate=dropout_rate,
            gene_likelihood=gene_likelihood,
            n_layer=n_layer,
            **model_kwargs,
        )

        self._model_summary_string = ""
        self.init_params_ = self._get_init_params(locals())

    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
        cls,
        adata: AnnData,
        layer: Optional[str] = None,
        batch_key: Optional[str] = None,
        design: Optional[list] = None,
        **kwargs,
    ):
        """%(summary)s.

        Parameters
        ----------
        %(param_adata)s
        %(param_layer)s
        %(param_batch_key)s
        """
        setup_method_args = cls._get_setup_method_args(**locals())
        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=True),
            CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
            NumericalJointObsField(REGISTRY_KEYS.CONT_COVS_KEY, design),
        ]
        adata_manager = AnnDataManager(
            fields=anndata_fields, setup_method_args=setup_method_args
        )
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)

    def get_latent_representation(
        self,
        adata: Optional[AnnData] = None,
        layer: int = -1,
        indices: Optional[Sequence[int]] = None,
        return_scale: bool = False,
        n_samples: int = 1,
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        r"""Return the latent representation for each cell.

        This is denoted as :math:`z_n` in our manuscripts.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        layer
            which layer to use. -1 refers to the salient space and -2 refers to the background space
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        return_scale
            Whether to return the scale of the posterior distribution or a sample. Depreciated
        n_samples
            Number of samples to use for computing the latent representation.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.

        Returns
        -------
        latent_representation : np.ndarray
            Low-dimensional representation for each cell
        """
        self._check_if_trained(warn=False)

        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size, iter_ndarray=True
        )

        jit_inference_fn = self.module.get_jit_inference_fn(
            inference_kwargs={"n_samples": n_samples}
        )

        # vae = self.module.bind({"params": self.module.params})
        # encoder = vae.salient_prior_encoder

        latent = []
        latent_scale = []
        for array_dict in scdl:
            out = jit_inference_fn(self.module.rngs, array_dict)
            # mean, var = encoder(out["qz"].mean, array_dict[REGISTRY_KEYS.CONT_COVS_KEY], False)
            # print(mean)
            # mean = jnp.hstack([jnp.zeros_like(mean), mean])
            # var = jnp.hstack([jnp.ones_like(var), var])
            if not return_scale:
                qz = out["qz"]
                qz = qz[layer]

                # z = z * var + mean
                z = qz.mean
            else:
                z = out["qz"].loc
                scale = out["qz"].scale
                latent_scale.append(scale)
            latent.append(z)
        concat_axis = 0 if ((n_samples == 1) or return_scale) else 1
        latent = jnp.concatenate(latent, axis=concat_axis)

        if return_scale:
            latent_scale = jnp.concatenate(latent_scale, axis=concat_axis)
            return self.module.as_numpy_array(latent), self.module.as_numpy_array(
                latent_scale
            )

        return self.module.as_numpy_array(latent)

    def to_device(self, device):
        pass

    # @partial(jax.jit, static_argnums=(0,))
    # def jit_validation_step(
    #     self,
    #     state: TrainStateWithState,
    #     batch: dict[str, np.ndarray],
    #     rngs: dict[str, jnp.ndarray],
    #     **kwargs,
    # ):
    #     """Jit validation step."""
    #     vars_in = {"params": state.params, **state.state}
    #     outputs = self.module.apply(vars_in, batch, rngs=rngs, **kwargs)
    #     loss_output = outputs[2]

    #     return loss_output

    # def validation_step(self, batch, batch_idx):
    #     """Validation step for Jax."""
    #     self.module.eval()
    #     loss_output = self.jit_validation_step(
    #         self.module.train_state,
    #         batch,
    #         self.module.rngs,
    #         loss_kwargs=None,
    #     )
    #     loss_output = jax.tree_util.tree_map(
    #         lambda x: torch.tensor(jax.device_get(x)),
    #         loss_output,
    #     )
    #     self._training_plan_cls.log(
    #         "validation_loss",
    #         loss_output.loss,
    #         on_epoch=True,
    #         batch_size=loss_output.n_obs_minibatch,
    #     )
    #     self._training_plan_cls.compute_and_log_metrics(loss_output, self.val_metrics, "validation")


    # def get_negative_likelihood(
    #         self,
    #         adata: Optional[AnnData] = None,
    #         indices: Optional[Sequence[int]] = None,
    #         n_samples: int = 1,
    #         batch_size: Optional[int] = None,
    #     ) -> np.ndarray:
    #         r"""Return the latent representation for each cell.

    #         This is denoted as :math:`z_n` in our manuscripts.

    #         Parameters
    #         ----------
    #         adata
    #             AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
    #             AnnData object used to initialize the model.
    #         indices
    #             Indices of cells in adata to use. If `None`, all cells are used.
    #         return_scale
    #             Whether to return the scale of the posterior distribution or a sample.
    #         n_samples
    #             Number of samples to use for computing the latent representation.
    #         batch_size
    #             Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.

    #         Returns
    #         -------
    #         latent_representation : np.ndarray
    #             Low-dimensional representation for each cell
    #         """
    #         self._check_if_trained(warn=False)

    #         adata = self._validate_anndata(adata)
    #         scdl = self._make_data_loader(
    #             adata=adata, indices=indices, batch_size=batch_size, iter_ndarray=True
    #         )

    #         # jit_inference_fn = self.module.training_plan.get_jit_inference_fn(
    #         #     # inference_kwargs={"n_samples": n_samples}
    #         # )

    #         # vae = self.module.bind({"params": self.module.params})
    #         # encoder = vae.salient_prior_encoder

    #         latent = []
    #         latent_scale = []
    #         for array_dict in scdl:
    #             out = self.training_plan.validation_step(array_dict, None)
    #             print(out)
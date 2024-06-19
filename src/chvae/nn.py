# import collections
# from typing import Iterable, Literal

# import torch
# from torch import nn

# from scvi.nn._utils import one_hot

# def _identity(x):
#     return x


# class FCLayers(nn.Module):
#     """A helper class to build fully-connected layers for a neural network.

#     Parameters
#     ----------
#     n_in
#         The dimensionality of the input
#     n_out
#         The dimensionality of the output
#     n_cat_list
#         A list containing, for each category of interest,
#         the number of categories. Each category will be
#         included using a one-hot encoding.
#     n_layers
#         The number of fully-connected hidden layers
#     n_hidden
#         The number of nodes per hidden layer
#     dropout_rate
#         Dropout rate to apply to each of the hidden layers
#     use_batch_norm
#         Whether to have `BatchNorm` layers or not
#     use_layer_norm
#         Whether to have `LayerNorm` layers or not
#     use_activation
#         Whether to have layer activation or not
#     bias
#         Whether to learn bias in linear layers or not
#     inject_covariates
#         Whether to inject covariates in each layer, or just the first (default).
#     activation_fn
#         Which activation function to use
#     """

#     def __init__(
#         self,
#         n_in: int,
#         n_out: int,
#         n_cat_list: Iterable[int] = None,
#         n_layers: int = 1,
#         n_hidden: int = 128,
#         dropout_rate: float = 0.1,
#         use_batch_norm: bool = True,
#         use_layer_norm: bool = False,
#         use_activation: bool = True,
#         bias: bool = True,
#         inject_covariates: bool = True,
#         activation_fn: nn.Module = nn.ReLU,
#     ):
#         super().__init__()
#         self.inject_covariates = inject_covariates
#         layers_dim = [n_in] + (n_layers - 1) * [n_hidden] + [n_out]

#         if n_cat_list is not None:
#             # n_cat = 1 will be ignored
#             self.n_cat_list = [n_cat if n_cat > 1 else 0 for n_cat in n_cat_list]
#         else:
#             self.n_cat_list = []

#         cat_dim = sum(self.n_cat_list)
#         self.fc_layers = nn.Sequential(
#             collections.OrderedDict(
#                 [
#                     (
#                         f"Layer {i}",
#                         nn.Sequential(
#                             nn.Linear(
#                                 n_in + cat_dim * self.inject_into_layer(i),
#                                 n_out,
#                                 bias=bias,
#                             ),
#                             # non-default params come from defaults in original Tensorflow implementation
#                             nn.BatchNorm1d(n_out, momentum=0.01, eps=0.001)
#                             if use_batch_norm
#                             else None,
#                             nn.LayerNorm(n_out, elementwise_affine=False)
#                             if use_layer_norm
#                             else None,
#                             activation_fn() if use_activation else None,
#                             nn.Dropout(p=dropout_rate) if dropout_rate > 0 else None,
#                         ),
#                     )
#                     for i, (n_in, n_out) in enumerate(
#                         zip(layers_dim[:-1], layers_dim[1:])
#                     )
#                 ]
#             )
#         )

#     def inject_into_layer(self, layer_num) -> bool:
#         """Helper to determine if covariates should be injected."""
#         user_cond = layer_num == 0 or (layer_num > 0 and self.inject_covariates)
#         return user_cond

#     def set_online_update_hooks(self, hook_first_layer=True):
#         """Set online update hooks."""
#         self.hooks = []

#         def _hook_fn_weight(grad):
#             categorical_dims = sum(self.n_cat_list)
#             new_grad = torch.zeros_like(grad)
#             if categorical_dims > 0:
#                 new_grad[:, -categorical_dims:] = grad[:, -categorical_dims:]
#             return new_grad

#         def _hook_fn_zero_out(grad):
#             return grad * 0

#         for i, layers in enumerate(self.fc_layers):
#             for layer in layers:
#                 if i == 0 and not hook_first_layer:
#                     continue
#                 if isinstance(layer, nn.Linear):
#                     if self.inject_into_layer(i):
#                         w = layer.weight.register_hook(_hook_fn_weight)
#                     else:
#                         w = layer.weight.register_hook(_hook_fn_zero_out)
#                     self.hooks.append(w)
#                     b = layer.bias.register_hook(_hook_fn_zero_out)
#                     self.hooks.append(b)

#     def forward(self, x: torch.Tensor, *cat_list: int):
#         """Forward computation on ``x``.

#         Parameters
#         ----------
#         x
#             tensor of values with shape ``(n_in,)``
#         cat_list
#             list of category membership(s) for this sample

#         Returns
#         -------
#         :class:`torch.Tensor`
#             tensor of shape ``(n_out,)``
#         """
#         one_hot_cat_list = []  # for generality in this list many indices useless.

#         # if len(self.n_cat_list) > len(cat_list):
#         #     raise ValueError(
#         #         "nb. categorical args provided doesn't match init. params."
#         #     )
        
#         for n_cat, cat in zip(self.n_cat_list, cat_list):
#             # if n_cat and cat is None:
#                 # raise ValueError("cat not provided while n_cat != 0 in init. params.")
#             if n_cat > 1:  # n_cat = 1 will be ignored - no additional information
#                 if cat.size(1) != n_cat:
#                     one_hot_cat = one_hot(cat, n_cat)
#                 else:
#                     one_hot_cat = cat  # cat has already been one_hot encoded
#                 one_hot_cat_list += [one_hot_cat]
#         for i, layers in enumerate(self.fc_layers):
#             for layer in layers:
#                 if layer is not None:
#                     if isinstance(layer, nn.BatchNorm1d):
#                         if x.dim() == 3:
#                             x = torch.cat(
#                                 [(layer(slice_x)).unsqueeze(0) for slice_x in x], dim=0
#                             )
#                         else:
#                             x = layer(x)
#                     else:
#                         if isinstance(layer, nn.Linear) and self.inject_into_layer(i):
#                             if x.dim() == 3:
#                                 one_hot_cat_list_layer = [
#                                     o.unsqueeze(0).expand(
#                                         (x.size(0), o.size(0), o.size(1))
#                                     )
#                                     for o in one_hot_cat_list
#                                 ]
#                             else:
#                                 one_hot_cat_list_layer = one_hot_cat_list
#                             x = torch.cat((x, *one_hot_cat_list_layer), dim=-1)
#                         x = layer(x)
#         return x

# # Decoder
# class DecoderSCVI(nn.Module):
#     """Decodes data from latent space of ``n_input`` dimensions into ``n_output`` dimensions.

#     Uses a fully-connected neural network of ``n_hidden`` layers.

#     Parameters
#     ----------
#     n_input
#         The dimensionality of the input (latent space)
#     n_output
#         The dimensionality of the output (data space)
#     n_cat_list
#         A list containing the number of categories
#         for each category of interest. Each category will be
#         included using a one-hot encoding
#     n_layers
#         The number of fully-connected hidden layers
#     n_hidden
#         The number of nodes per hidden layer
#     dropout_rate
#         Dropout rate to apply to each of the hidden layers
#     inject_covariates
#         Whether to inject covariates in each layer, or just the first (default).
#     use_batch_norm
#         Whether to use batch norm in layers
#     use_layer_norm
#         Whether to use layer norm in layers
#     scale_activation
#         Activation layer to use for px_scale_decoder
#     **kwargs
#         Keyword args for :class:`~scvi.nn.FCLayers`.
#     """

#     def __init__(
#         self,
#         n_input: int,
#         n_output: int,
#         n_cat_list: Iterable[int] = None,
#         n_layers: int = 1,
#         n_hidden: int = 128,
#         inject_covariates: bool = True,
#         use_batch_norm: bool = False,
#         use_layer_norm: bool = False,
#         scale_activation: Literal["softmax", "softplus"] = "softmax",
#         **kwargs,
#     ):
#         super().__init__()
#         # self.px_decoder = FCLayers(
#         #     n_in=n_input,
#         #     n_out=n_hidden,
#         #     n_cat_list=n_cat_list,
#         #     n_layers=n_layers,
#         #     n_hidden=n_hidden,
#         #     dropout_rate=0,
#         #     inject_covariates=inject_covariates,
#         #     use_batch_norm=use_batch_norm,
#         #     use_layer_norm=use_layer_norm,
#         #     **kwargs,
#         # )

#         if n_cat_list is not None:
#             # n_cat = 1 will be ignored
#             self.n_cat_list = [n_cat if n_cat > 1 else 0 for n_cat in n_cat_list]
#         else:
#             self.n_cat_list = []

#         cat_dim = sum(self.n_cat_list)
#         self.px_decoder = torch.nn.Sequential(
#             nn.Linear(n_input + cat_dim, n_hidden),
#             nn.LayerNorm(n_hidden),
#             nn.ReLU()
#         )
#         # mean gamma
#         if scale_activation == "softmax":
#             px_scale_activation = nn.Softmax(dim=-1)
#         elif scale_activation == "softplus":
#             px_scale_activation = nn.Softplus()
#         self.px_scale_decoder = nn.Sequential(
#             nn.Linear(n_hidden, n_output),
#             px_scale_activation,
#         )

#         # dispersion: here we only deal with gene-cell dispersion case
#         self.px_r_decoder = nn.Linear(n_hidden, n_output)

#         # dropout
#         self.px_dropout_decoder = nn.Linear(n_hidden, n_output)

#     def forward(
#         self,
#         z: torch.Tensor,
#         dispersion: str,
#         library: torch.Tensor,
#         *cat_list: int,
#     ):
#         """The forward computation for a single sample.

#          #. Decodes the data from the latent space using the decoder network
#          #. Returns parameters for the ZINB distribution of expression
#          #. If ``dispersion != 'gene-cell'`` then value for that param will be ``None``

#         Parameters
#         ----------
#         dispersion
#             One of the following

#             * ``'gene'`` - dispersion parameter of NB is constant per gene across cells
#             * ``'gene-batch'`` - dispersion can differ between different batches
#             * ``'gene-label'`` - dispersion can differ between different labels
#             * ``'gene-cell'`` - dispersion can differ for every gene in every cell
#         z :
#             tensor with shape ``(n_input,)``
#         library_size
#             library size
#         cat_list
#             list of category membership(s) for this sample

#         Returns
#         -------
#         4-tuple of :py:class:`torch.Tensor`
#             parameters for the ZINB distribution of expression

#         """
#         # The decoder returns values for the parameters of the ZINB distribution
#         px = self.px_decoder(z, *cat_list)
#         px_scale = self.px_scale_decoder(px)
#         px_dropout = self.px_dropout_decoder(px)
#         # Clamp to high value: exp(12) ~ 160000 to avoid nans (computational stability)
#         px_rate = torch.exp(library) * px_scale  # torch.clamp( , max=12)
#         px_r = self.px_r_decoder(px) if dispersion == "gene-cell" else None
#         return px_scale, px_rate

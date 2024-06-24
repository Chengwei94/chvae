import pyro
from scvi.data import synthetic_iid

from chvae.model import JaxSCVI


def test_mymodel():
    adata = synthetic_iid()
    JaxSCVI.setup_anndata(adata, batch_key="batch", labels_key="labels")
    model = JaxSCVI(adata)
    model.train(1)
    # model.get_elbo()
    # model.get_latent_representation()
    # model.get_marginal_ll(n_mc_samples=5)
    # model.get_reconstruction_error()
    model.history
    # tests __repr__
    print(model)


# def test_mypyromodel():
#     adata = synthetic_iid()
#     pyro.clear_param_store()
#     MyPyroModel.setup_anndata(adata, batch_key="batch", labels_key="labels")
#     model = MyPyroModel(adata)
#     model.train(max_epochs=1, train_size=1)
#     model.get_latent(adata)
#     model.history

#     # tests __repr__
#     print(model)

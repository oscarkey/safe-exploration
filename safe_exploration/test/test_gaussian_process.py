"""Test the multi-output GP implementations."""


import pytest

from ..ssm_pytorch.gaussian_process import ZeroMeanWithGrad

try:
    import torch
    from safe_exploration.ssm_pytorch import MultiOutputGP, LinearMean, GPyTorchSSM
    import gpytorch
    from torch.nn.functional import softplus
except:
    pass


@pytest.fixture(autouse = True)
def check_has_ssm_pytorch_module(check_has_ssm_pytorch):
    pass


class TestLinearMean(object):

    def test_trainable(self):
        A = torch.randn((1, 3))
        mean = LinearMean(A, trainable=True)
        assert len(list(mean.parameters())) == 1

    def test_1d(self):
        x = torch.randn((10, 3))
        A = torch.randn((1, 3))

        mean = LinearMean(A, trainable=False)
        # Make sure matrix is not trainable
        assert not list(mean.parameters())
        assert mean.batch_size == 1

        out = mean(x)

        torch.testing.assert_allclose(out, (x @ A.t()).t()[0])

    def test_multidim(self):
        x = torch.randn((2, 10, 3))
        A = torch.randn((2, 3))

        mean = LinearMean(A, trainable=False)
        out = mean(x)

        assert mean.batch_size == 2
        # A @ x.T = (x.T @ A.T).T  for each x. The latter also works for multiple x.
        torch.testing.assert_allclose(out[[0]], (x[0] @ A[[0]].t()).t())
        torch.testing.assert_allclose(out[[1]], (x[1] @ A[[1]].t()).t())

try: # This requires the ssm_pytorch dependencies and throws an error.
     # However we do not use it anyways in this case hence no exception
     # handling required
    class ExactGPModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, cov, likelihood, mean = None):
            super().__init__(train_x, train_y, likelihood)
            if mean is None:
                mean = gpytorch.means.ConstantMean()
            self.mean_module = mean
            self.covar_module = cov

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
except:
    pass


class TestZeroMeanWithGrad:
    def test__forward__returns_zero(self):
        mean = ZeroMeanWithGrad()
        output = mean(torch.tensor([0, 10]))
        assert torch.all(torch.eq(output, torch.tensor([0, 0])))

    def test__forward__maintains_type_and_grad_of_input(self):
        mean = ZeroMeanWithGrad()

        output = mean(torch.tensor([100, 50], dtype=torch.double, requires_grad=True))

        assert output.dtype == torch.double
        assert output.requires_grad is True


class TestMultiOutputGP(object):

    def test_single_output_gp(self):
        kernel = gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=None, batch_size=1, active_dims=None, lengthscale_prior=None, param_transform=softplus, inv_param_transform=None, eps=1e-6)
        mean = LinearMean(torch.tensor([[0.5]]))
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        #likelihood.noise = torch.tensor(0.01 ** 2)

        train_x = torch.tensor([-0.5, -0.1, 0., 0.1, 1.])[:, None]
        train_y = 0.5 * train_x.t()

        model = MultiOutputGP(train_x, train_y, kernel, likelihood, mean=mean, num_outputs=1)
        model.eval()

        test_x = torch.linspace(-1, 2, 5)
        pred = model(test_x)

        true_mean = torch.tensor([-0.5, -0.125, 0.25, 0.6250, 1.0])[None, :]
        torch.testing.assert_allclose(pred.mean, true_mean)

    def test_multi_output_gp(self):
        # Setup composite mean
        mean1 = gpytorch.means.ConstantMean()
        mean2 = gpytorch.means.ConstantMean()
        mean = gpytorch.means.ConstantMean(batch_size=2)

        # Setup composite kernel
        cov1 = gpytorch.kernels.RBFKernel()
        cov2 = gpytorch.kernels.RBFKernel()
        kernel = gpytorch.kernels.RBFKernel(batch_size=2)

        # Training data
        train_x = torch.randn(5, 2)
        train_y = torch.randn(5, 2).t()

        # Combined GP
        likelihood = gpytorch.likelihoods.GaussianLikelihood(batch_size=2)
        gp = MultiOutputGP(torch.rand_like(train_x), torch.rand_like(train_y), kernel, likelihood, mean=mean, num_outputs=2)
        # We initialized with random training data so we can test set_train_data here.
        gp.set_train_data(train_x, train_y)

        # Individual GPs
        likelihood1 = gpytorch.likelihoods.GaussianLikelihood()
        gp1 = ExactGPModel(train_x, train_y[0], cov1, likelihood1, mean=mean1)
        gp2 = ExactGPModel(train_x, train_y[1], cov2, likelihood1, mean=mean2)

        # Evaluation mode
        gp.eval()
        gp1.eval()
        gp2.eval()

        # Evaluate
        test_x = torch.randn(10, 2)
        pred = gp(test_x)
        pred1 = gp1(test_x)
        pred2 = gp2(test_x)

        torch.testing.assert_allclose(pred.mean[0], pred1.mean)
        torch.testing.assert_allclose(pred.mean[1], pred2.mean)

        torch.testing.assert_allclose(pred.covariance_matrix[0],
                                      pred1.covariance_matrix)
        torch.testing.assert_allclose(pred.covariance_matrix[1],
                                      pred2.covariance_matrix)

        torch.testing.assert_allclose(pred.variance[0], pred1.variance)
        torch.testing.assert_allclose(pred.variance[1], pred2.variance)

        # Test optimization
        gp.train()
        optimizer = torch.optim.Adam([{'params': gp.parameters()}], lr=0.1)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp)
        optimizer.zero_grad()
        loss = gp.loss(mll)
        loss.backward()

        optimizer.step()

    def test__init__training_data_none__does_not_crash(self):
        state_dimen = 2
        kernel = gpytorch.kernels.RBFKernel(batch_size=state_dimen)
        likelihood = gpytorch.likelihoods.GaussianLikelihood(batch_size=state_dimen)

        MultiOutputGP(train_x=None, train_y=None, kernel=kernel, likelihood=likelihood, num_outputs=state_dimen)

    def test__set_train_data__training_data_none__does_not_crash(self):
        state_dimen = 2
        kernel = gpytorch.kernels.RBFKernel(batch_size=state_dimen)
        likelihood = gpytorch.likelihoods.GaussianLikelihood(batch_size=state_dimen)
        train_x = torch.tensor([[1, 2], [3, 4]])
        train_y = torch.tensor([10, 11])
        gp = MultiOutputGP(train_x, train_y, kernel, likelihood, num_outputs=state_dimen)

        gp.set_train_data(None, None)

@pytest.fixture()
def before_test_gpytorchssm(check_has_ssm_pytorch):

    n_s = 2
    n_u = 1

    kernel = gpytorch.kernels.MaternKernel()
    mean = LinearMean(torch.tensor([[0.5]]))
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    likelihood.noise = torch.tensor(0.01 ** 2)

    train_x = torch.tensor([-0.5, -0.1, 0., 0.1, 1.])[:, None]
    train_y = 0.5 * train_x.t()

    ssm = GPyTorchSSM(n_s,n_u,train_x,train_y,kernel,likelihood)#,mean)

    return ssm,n_s,n_u,train_x,train_y,kernel,likelihood


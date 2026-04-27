import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import torch as t
import math
import numpy as np

device = 'cuda' if t.cuda.is_available() else 'cpu'
# Network
class MLP(nn.Module):
    def __init__(self, input_dim,
                 output_dim,
                 hidden_size=(128, 64),
                 activation='LeakyReLU',
                 discrim=False, dropout=-1):

        super(MLP, self).__init__()
        dims = []
        dims.append(input_dim)
        dims.extend(hidden_size)
        dims.append(output_dim)
        self.layers = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'LeakyReLU':
            self.activation = nn.LeakyReLU()

        self.sigmoid = nn.Sigmoid() if discrim else None
        self.dropout = dropout

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            if i != len(self.layers) - 1:
                x = self.activation(x)
                if self.dropout != -1:
                    x = nn.Dropout(min(0.1, self.dropout / 3) if i == 1 else self.dropout)(x)
            elif self.sigmoid:
                x = self.sigmoid(x)
        return x

class ModuleWrapper(nn.Module):
    """Wrapper for nn.Module with support for arbitrary flags and a universal forward pass"""

    def __init__(self):
        super(ModuleWrapper, self).__init__()

    def set_flag(self, flag_name, value):
        setattr(self, flag_name, value)
        for m in self.children():
            if hasattr(m, 'set_flag'):
                m.set_flag(flag_name, value)

    def forward(self, x):
        for module in self.children():
            x = module(x)

        kl = 0.0
        for module in self.modules():
            if hasattr(module, 'kl_loss'):
                kl = kl + module.kl_loss()

        return x, kl

# 贝叶斯形式下的随机初始化线性层,(参考：PyTorch-BayesianCNN-master)
class BBBLinear(ModuleWrapper):
    def __init__(self, in_features, out_features, bias=True, priors=None):
        super(BBBLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        self.device = t.device("cuda:0" if t.cuda.is_available() else "cpu")

        if priors is None:
            priors = {
                'prior_mu': 0,
                'prior_sigma': 0.1,
                'posterior_mu_initial': (0, 0.1),
                'posterior_rho_initial': (-3, 0.1),
            }
        self.prior_mu = priors['prior_mu']
        self.prior_sigma = priors['prior_sigma']
        self.posterior_mu_initial = priors['posterior_mu_initial']
        self.posterior_rho_initial = priors['posterior_rho_initial']

        self.W_mu = Parameter(t.Tensor(out_features, in_features))
        self.W_rho = Parameter(t.Tensor(out_features, in_features))
        if self.use_bias:
            self.bias_mu = Parameter(t.Tensor(out_features))
            self.bias_rho = Parameter(t.Tensor(out_features))
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_rho', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.W_mu.data.normal_(*self.posterior_mu_initial)
        self.W_rho.data.normal_(*self.posterior_rho_initial)

        if self.use_bias:
            self.bias_mu.data.normal_(*self.posterior_mu_initial)
            self.bias_rho.data.normal_(*self.posterior_rho_initial)

    def forward(self, x, sample=True):

        self.W_sigma = t.log1p(t.exp(self.W_rho))
        if self.use_bias:
            self.bias_sigma = t.log1p(t.exp(self.bias_rho))
            bias_var = self.bias_sigma ** 2
        else:
            self.bias_sigma = bias_var = None

        act_mu = F.linear(x, self.W_mu, self.bias_mu)
        act_var = 1e-16 + F.linear(x ** 2, self.W_sigma ** 2, bias_var)
        act_std = t.sqrt(act_var)

        if self.training or sample:
            eps = t.empty(act_mu.size()).normal_(0, 1).to(self.device)
            return act_mu + act_std * eps
        else:
            return act_mu

    def kl_loss(self):
        kl = KL_DIV(self.prior_mu, self.prior_sigma, self.W_mu, self.W_sigma)
        if self.use_bias:
            kl += KL_DIV(self.prior_mu, self.prior_sigma, self.bias_mu, self.bias_sigma)
        return kl


# 贝叶斯形式下的未初始化线性层,(参考：PyTorch-BayesianCNN-master)
class BBBLinear_noneInit(ModuleWrapper):
    def __init__(self, in_features, out_features, bias=True, priors=None):
        super(BBBLinear_noneInit, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        self.device = t.device("cuda:0" if t.cuda.is_available() else "cpu")

        if priors is None:
            priors = {
                'prior_mu': 0,
                'prior_sigma': 0.1,
                'posterior_mu_initial': (0, 0.1),
                'posterior_rho_initial': (-3, 0.1),
            }
        self.prior_mu = priors['prior_mu']
        self.prior_sigma = priors['prior_sigma']
        self.posterior_mu_initial = priors['posterior_mu_initial']
        self.posterior_rho_initial = priors['posterior_rho_initial']

        self.W_mu = Parameter(t.empty((out_features, in_features), device=self.device))
        self.W_rho = Parameter(t.empty((out_features, in_features), device=self.device))

        if self.use_bias:
            self.bias_mu = Parameter(t.empty((out_features), device=self.device))
            self.bias_rho = Parameter(t.empty((out_features), device=self.device))
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_rho', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.W_mu.data.normal_(*self.posterior_mu_initial)
        self.W_rho.data.normal_(*self.posterior_rho_initial)

        if self.use_bias:
            self.bias_mu.data.normal_(*self.posterior_mu_initial)
            self.bias_rho.data.normal_(*self.posterior_rho_initial)

    def forward(self, input, sample=True):
        if self.training or sample:
            W_eps = t.empty(self.W_mu.size()).normal_(0, 1).to(self.device)
            self.W_sigma = t.log1p(t.exp(self.W_rho))
            weight = self.W_mu + W_eps * self.W_sigma

            if self.use_bias:
                bias_eps = t.empty(self.bias_mu.size()).normal_(0, 1).to(self.device)
                self.bias_sigma = t.log1p(t.exp(self.bias_rho))
                bias = self.bias_mu + bias_eps * self.bias_sigma
            else:
                bias = None
        else:
            weight = self.W_mu
            bias = self.bias_mu if self.use_bias else None

        return F.linear(input, weight, bias)

    def kl_loss(self):
        kl = KL_DIV(self.prior_mu, self.prior_sigma, self.W_mu, self.W_sigma)
        if self.use_bias:
            kl += KL_DIV(self.prior_mu, self.prior_sigma, self.bias_mu, self.bias_sigma)
        return kl

def KL_DIV(mu_q, sig_q, mu_p, sig_p):
    kl = 0.5 * (2 * t.log(sig_p / sig_q) - 1 + (sig_q / sig_p).pow(2) + ((mu_p - mu_q) / sig_p).pow(2)).sum()
    return kl

# 1月1日添加,(参考代码:bayes-by-backprop-master)
class ScaleMixtureGaussian(object):
    def __init__(self, pi, sigma1, sigma2):
        super().__init__()
        self.pi = pi
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.gaussian1 = t.distributions.Normal(0, sigma1)
        self.gaussian2 = t.distributions.Normal(0, sigma2)

    def log_prob(self, input):
        prob1 = t.exp(self.gaussian1.log_prob(input))
        prob2 = t.exp(self.gaussian2.log_prob(input))
        return (t.log(self.pi * prob1 + (1 - self.pi) * prob2)).sum()

class Gaussian(object):
    def __init__(self, mu, rho):
        super().__init__()
        self.mu = mu
        self.rho = rho
        self.normal = t.distributions.Normal(0, 1)

        if not isinstance(self.mu, t.Tensor):
            self.mu = t.tensor(self.mu).to(device)
        if not isinstance(self.rho, t.Tensor):
            self.rho = t.tensor(self.rho).to(device)

        self.distributions = t.distributions.Normal(self.mu, self.rho)
    @property
    def sigma(self):
        return t.log1p(t.exp(self.rho))

    def sample(self):
        # epsilon = self.normal.sample(self.rho.size()).to(device)
        # return self.mu + self.sigma * epsilon
        return self.distributions.sample()

    def log_prob(self, input):
        # return (-math.log(math.sqrt(2 * math.pi))
        #         - t.log(self.sigma)
        #         - ((input - self.mu) ** 2) / (2 * self.sigma ** 2)).sum()
        return self.distributions.log_prob(input)

    def prob(self, input):
        return t.exp(self.log_prob(input))

# bayes-by-backprop-master
# class BayesianLinear(nn.Module):
#     def __init__(self, in_features, out_features, args):
#         super().__init__()
#         self.args = args
#         self.in_features = in_features
#         self.out_features = out_features
#
#         # Hyper parameters
#         self.prior_sigma = np.sqrt(self.args.PI * self.args.SIGMA_1 ** 2 +
#                                    (1 - self.args.PI) * self.args.SIGMA_2 ** 2)
#
#         # Weight parameters
#         self.weight_mu = nn.Parameter(t.Tensor(out_features, in_features).normal_(0, self.prior_sigma))
#         self.weight_rho = nn.Parameter(t.Tensor(out_features, in_features))
#         nn.init.constant_(self.weight_rho, 0)
#         # self.weight = Gaussian(self.weight_mu, self.weight_rho)
#         # Bias parameters
#         self.bias_mu = nn.Parameter(t.Tensor(out_features).normal_(0, self.prior_sigma))
#         self.bias_rho = nn.Parameter(t.Tensor(out_features))
#         nn.init.constant_(self.bias_rho, 0)
#
#         # self.bias = Gaussian(self.bias_mu, self.bias_rho)
#         # Prior distributions
#         # self.weight_prior = ScaleMixtureGaussian(args.PI, args.SIGMA_1, args.SIGMA_2)
#         # self.bias_prior = ScaleMixtureGaussian(args.PI, args.SIGMA_1, args.SIGMA_2)
#         # self.log_prior_value = 0
#         # self.log_variational_posterior_value = 0
#         self.kl_loss_value = 0
#
#     def forward(self, input, sample=False):
#         '''
#         if self.training or sample:
#             weight = self.weight.sample()
#             bias = self.bias.sample()
#         else:
#             weight = self.weight.mu
#             bias = self.bias.mu
#
#         self.log_prior_value = self.weight_prior.log_prob(weight) + self.bias_prior.log_prob(bias)
#         self.log_variational_posterior_value = self.weight.log_prob(weight) + self.bias.log_prob(bias)
#
#         return F.linear(input, weight, bias)
#         '''
#         # 以下是参考bayes-linear-regression-master(1.13更改)
#         kernel_sigma = nn.functional.softplus(self.weight_mu)
#         # kernel = self.weight_mu + kernel_sigma * t.randn(self.weight_mu.shape).to(device)
#         kernel = Gaussian(mu=self.weight_mu, rho=kernel_sigma).sample()
#
#         bias_sigma = nn.functional.softplus(self.bias_mu)
#         # bias = self.bias_mu + bias_sigma * t.randn(self.bias_mu.shape).to(device)
#         bias = Gaussian(mu=self.bias_mu, rho=bias_sigma).sample()
#
#         # kl loss compute
#         kl_loss_kernel = self.kl_loss_fn(kernel, self.weight_mu, kernel_sigma)
#         kl_loss_bias = self.kl_loss_fn(bias, self.bias_mu, bias_sigma)
#         self.kl_loss_value = kl_loss_kernel + kl_loss_bias
#
#         return t.matmul(input, kernel.T) + bias
#
#     def kl_loss_fn(self, w, mu, sigma):
#         # log(q(w|θ))
#         variational_dist = Gaussian(mu, sigma)
#         prob1 = variational_dist.log_prob(w)
#         # log(p(w)) 混合先验
#         prior_1_dist = Gaussian(0.0, self.args.SIGMA_1)
#         prior_2_dist = Gaussian(0.0, self.args.SIGMA_2)
#         prob2 = t.log(self.args.PI * prior_1_dist.prob(w) + (1.0-self.args.PI) * prior_2_dist.prob(w))
#
#         return t.mean(prob1 - prob2)
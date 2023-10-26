import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
import ipdb


class DDPM(nn.Module):

    def __init__(self, input_dim, T, beta_min, beta_max, device):
        super().__init__()

        hidden = 64
        self.T = T
        self.input_dim = input_dim
        self.beta = torch.linspace(beta_min, beta_max, T).to(device) # 1e-4, 0.02
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cat([torch.prod(self.alpha[:i]).view(1) for i in range(1, T+1)])

        if len(input_dim) > 1:
            self.f_pre = nn.Sequential(nn.Conv2d(3, hidden, 3, stride=1, padding=1),
                                       nn.ReLU(),
                                       nn.Conv2d(hidden, hidden, 3, stride=1, padding=1),
                                       nn.ReLU(),
                                       nn.Conv2d(hidden, hidden, 3, stride=1, padding=1),
                                       nn.ReLU(),
                                       nn.Conv2d(hidden, hidden, 3, stride=1, padding=1))
            self.film_layer = nn.Linear(1, hidden*2)
            self.f = nn.Sequential(nn.Conv2d(hidden, hidden, 3, stride=1, padding=1),
                                   nn.ReLU(),
                                   nn.Conv2d(hidden, hidden, 3, stride=1, padding=1),
                                   nn.ReLU(),
                                   nn.Conv2d(hidden, hidden, 3, stride=1, padding=1),
                                   nn.ReLU(),
                                   nn.Conv2d(hidden, 3, 3, stride=1, padding=1))
        else:
            self.film_layer = None
            self.f = nn.Sequential(nn.Linear(input_dim[0] + 1, hidden),
                                   nn.ReLU(),
                                   nn.Linear(hidden, hidden),
                                   nn.ReLU(),
                                   nn.Linear(hidden, hidden),
                                   nn.ReLU(),
                                   nn.Linear(hidden, input_dim[0]))

    def forward(self, x, t):

        if self.film_layer is not None:
            h = self.f_pre(x)
            z = self.film_layer(t.float())
            beta = torch.unsqueeze(torch.unsqueeze(z[:, :int(z.shape[-1] / 2)], -1), -1)
            gamma = torch.unsqueeze(torch.unsqueeze(z[:, int(z.shape[-1] / 2):], -1), -1)
            h = gamma * h + beta
        else:
            h = torch.cat([x, t], -1)

        return self.f(h)

    def get_loss(self, x0, t, epsilon):
        alpha_bar_t1 = self.alpha_bar[t-1] if len(x0.shape) == 2 else torch.unsqueeze(torch.unsqueeze(self.alpha_bar[t-1], 2), 3)
        xt = torch.sqrt(alpha_bar_t1) * x0 + torch.sqrt(1 - alpha_bar_t1)*epsilon
        #epsilon_hat = self.f(torch.cat([xt, t], -1))
        epsilon_hat = self.forward(xt, t)

        loss = torch.mean((epsilon - epsilon_hat) ** 2)

        return loss

    def sample(self, n_samples, device):
        dims = [n_samples]
        dims.extend(self.input_dim)
        x = torch.randn(tuple(dims)).to(device)
        x_time = [x.detach().cpu().numpy()]
        for t in range(self.T, 0, -1):
            x = self.reverse_diffusion_step(x, t, device)
            x_time.append(x.detach().cpu().numpy())
        return x_time

    def reverse_diffusion_step(self, xt, t, device):
        z = torch.randn(xt.shape).to(device)
        z = z * 0 if t == 1 else z
        # sigma_t = (self.beta[t - 1] ** 2) * torch.eye(xt.shape[-1]).to(device)
        sigma_t = (self.beta[t - 1] ** 2) * torch.eye(np.prod(xt.shape[1:])).to(device)
        # epsilon_hat = self.f(torch.cat([xt, torch.ones((xt.shape[0], 1)).float().to(device) * t], -1))
        epsilon_hat = self.forward(xt, torch.ones((xt.shape[0], 1)).float().to(device) * t)
        xt1 = (1 / torch.sqrt(self.alpha[t - 1])) * (xt - (1 - self.alpha[t - 1]) / torch.sqrt(1 - self.alpha_bar[t - 1]) * epsilon_hat)
        if len(self.input_dim) > 1:
            xt1 += (z.view(z.shape[0], -1) @ sigma_t).view(xt.shape)
        else:
            xt1 += z @ sigma_t
        return xt1



















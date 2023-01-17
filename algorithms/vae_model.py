import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, in_dim, hidden_layers, latent_dim, device):
        super(Encoder, self).__init__()

        shape_layers = [in_dim] + hidden_layers
        layers = [
            [nn.Linear(i, j,  device=device), nn.ReLU()] for i, j in zip(shape_layers[:-1], shape_layers[1:])
        ]
        layers = [
            layer for c_layers in layers for layer in c_layers
        ]    

        self.base_model = nn.Sequential(*layers)
        self.latent_mean = nn.Linear(hidden_layers[-1], latent_dim, device=device)
        self.latent_std = nn.Linear(hidden_layers[-1], latent_dim, device=device)
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X_base = self.base_model(X)
        return self.latent_mean(X_base), self.latent_std(X_base)

class Decoder(nn.Module):
    def __init__(self, in_dim, hidden_layers, latent_dim, device):
        super(Decoder, self).__init__()

        shape_layers = [latent_dim] + hidden_layers
        layers = [
            [nn.Linear(i, j,  device=device), nn.ReLU()] for i, j in zip(shape_layers[:-1], shape_layers[1:])
        ]
        layers = [
            i for j in layers for i in j
        ]
        layers.append(nn.Linear(shape_layers[-1], in_dim, device=device))

        self.model = nn.Sequential(*layers)
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.model(X)



class VAE:
    def __init__(
        self,
        train_data,
        test_data,
        in_dim,
        encoder_hidden_layers,
        decoder_hidden_layers,
        latent_dim,
        device=None,
    ):
        # device
        self.name = "VAE"
        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = device
        self.latent_dim = latent_dim
        self.encoder_hidden_layers = encoder_hidden_layers
        self.decoder_hidden_layers = decoder_hidden_layers
        self.in_dim = in_dim

        # initialize encoder/decoder weights and biases
        self.initialize_encoder()
        self.initialize_decoder()

        # config dataset
        self.train_data = train_data
        data = next(iter(train_data))
        self.example_size = data.size()
        self.test_data = test_data

    def train(self, batch_size, max_epoch, lr, weight_decay):
        optimizer = self._get_optimizer(lr, weight_decay)
        hist_loss = []

        train_dataloader = DataLoader(
            self.train_data, batch_size, shuffle=True, drop_last=True, num_workers=0
        )

        # print initial loss
        data = next(iter(train_dataloader))
        Xground = data.view((batch_size, -1)).to(self.device)
        loss = self._vae_loss(Xground)

        tk = tqdm(range(max_epoch))
        for epoch in tk:
            res = []
            for ii, data in enumerate(train_dataloader):
                Xground = data.view((batch_size, -1)).to(self.device)

                optimizer.zero_grad()
                loss = self._vae_loss(Xground)

                # backward propagate
                loss.backward()
                optimizer.step()
                res.append(loss.item())
            hist_loss.append(res)
            tk.set_postfix({"val_loss": res[-1], "epoch": epoch})

        return np.array(hist_loss)

    def generate_from_test_data(self, batch_size):
        """data reconstruction test"""
        test_dataloader = DataLoader(
            self.test_data, batch_size, shuffle=True, drop_last=True, num_workers=0
        )

        data = next(iter(test_dataloader))
        Xground = data.view((batch_size, -1)).to(self.device)

        z_mean, z_logstd = self._encoding(Xground)

        epsi = torch.randn(z_logstd.size()).to(self.device)
        z_star = z_mean + torch.exp(0.5 * z_logstd) * epsi

        Xstar = self._decoding(z_star)
        Xstar = torch.sigmoid(Xstar)

        Xstar = Xstar.view(data.size())

        return data, Xstar

    def generate_from_latent_space(self, batch_size):
        """distribution transformation test(generate artificial dataset from random noises)"""
        Z = torch.randn((batch_size, self.latent_dim)).to(self.device)
        Xstar = self._decoding(Z).view((-1, *self.example_size))

        return Xstar

    def _vae_loss(self, Xground):
        """compute VAE loss = kl_loss + likelihood_loss"""

        # KL loss
        z_mean, z_logstd = self._encoding(Xground)
        kl_loss = 0.5 * torch.sum(
            1 + z_logstd - z_mean ** 2 - torch.exp(z_logstd), dim=1
        )

        # likelihood loss
        epsi = torch.randn(z_logstd.size()).to(self.device)
        z_star = z_mean + z_logstd * epsi  # torch.exp(0.5 * z_logstd) # reparameterize trick
        Xstar = self._decoding(z_star)

        llh_loss = Xground * torch.log(1e-12 + Xstar) + (1 - Xground) * torch.log(
            1e-12 + 1 - Xstar
        )
        llh_loss = torch.sum(llh_loss, dim=1)

        var_loss = -torch.mean(kl_loss + llh_loss)

        return var_loss

    def _get_optimizer(self, lr, weight_decay):
        # adding weights encoder/decoder to optimization paramters list
        opt_params = list(self.encoder.parameters()) + list(self.decoder.parameters())

        return Adam(opt_params, lr=lr, weight_decay=weight_decay)

    def _encoding(self, X):
        # Kingma Supplemtary C.2
        mean_output, logstd_output = self.encoder(X)

        return mean_output, logstd_output

    def _decoding(self, Z):
        Xstar = self.decoder(Z)
        Xstar = torch.sigmoid(Xstar)

        return Xstar
    
    def initialize_encoder(self):
        self.encoder = Encoder(self.in_dim, self.encoder_hidden_layers, self.latent_dim, self.device)
    
    def initialize_decoder(self):
        self.decoder = Decoder(self.in_dim, self.decoder_hidden_layers, self.latent_dim, self.device)

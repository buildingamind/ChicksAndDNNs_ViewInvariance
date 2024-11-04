import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from models.common import create_decoder, create_encoder


class BetaVAE(pl.LightningModule):
    def __init__(self, height, width, input_channels, latent_dim, beta=1.0):
        super().__init__()
        self.save_hyperparameters()

        self.channels = [input_channels, 32, 32, 64, 64]
        self.h_out = int(height / (2 ** len(self.channels[1:])))
        self.w_out = int(width / (2 ** len(self.channels[1:])))
        self.hidden_dim = int(self.h_out * self.w_out * self.channels[-1])

        # Create convolutional encoder.
        self.encoder = create_encoder(self.channels)

        # Create fc layers.
        self.fc_mu = nn.Linear(self.hidden_dim, latent_dim)
        self.fc_var = nn.Linear(self.hidden_dim, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, self.hidden_dim)

        # Create convolutional decoder.
        self.decoder = create_decoder(self.channels[::-1])

    def encode(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var

    def decode(self, z):
        z = F.leaky_relu(self.fc_decode(z))
        z = z.view(-1, self.channels[-1], self.h_out, self.w_out)
        return self.decoder(z)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x):
        mu, log_var = self.encode(x)
        sigma = torch.exp(log_var)
        eps = torch.randn_like(sigma)
        z = eps * sigma + mu
        recon_x = self.decode(z)
        return recon_x, mu, log_var

    def sample(self, num_samples):
        with torch.no_grad():
            z = torch.randn(num_samples, self.hparams.latent_dim)
            samples = self.decode(z)
            return samples

    def loss_function(self, recon_x, x, mu, log_var):
        recon_loss = F.mse_loss(recon_x, x, size_average=False)
        kld_loss = -0.5 * torch.sum(1 + 2 * log_var - mu.pow(2) - (2 * log_var).exp())
        loss = recon_loss + self.hparams.beta * kld_loss
        return {
            'loss': loss,
            'recon_loss': recon_loss,
            'kld_loss': kld_loss
        }

    def training_step(self, batch, batch_idx):
        images, _ = batch
        recon, mu, log_var = self(images)

        loss = self.loss_function(recon, images, mu, log_var)
        result = pl.TrainResult(loss['loss'])
        for k, v in loss.items():
            result.log('Train/'+k, v)
        return result

    def validation_step(self, batch, batch_idx):
        images, _ = batch
        recon, mu, log_var = self(images)

        loss = self.loss_function(recon, images, mu, log_var)
        result = pl.EvalResult(checkpoint_on=loss['loss'])
        for k, v in loss.items():
            result.log('Eval/'+k, v)
        return result

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

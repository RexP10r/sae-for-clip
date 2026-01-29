import torch
import torch.nn as nn

class SAE(nn.Module):
    def __init__(self, input_dim: int, latent_upsample: int):
        super().__init__()
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim * latent_upsample, bias=False),
            nn.ReLU()
        )
        self._decoder = nn.Linear(input_dim * latent_upsample, input_dim)
        self.to(self._device)

    def encode(self, embed):
        return self._encoder(embed)

    def decode(self, latent):
        return self._decoder(latent)

    def get_metrics_names(self):
        return self._metrics_names

    def forward(self, input):
        latent = self._encoder(input)
        recon = self._decoder(latent)
        return recon, latent

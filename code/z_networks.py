import torch
from torch import nn
class s_encoder(nn.Module):
    def __init__(self, num_embed_size, hidden_size, duration_size, mag_size, time_size, output_size):
        super(s_encoder, self).__init__()

        self.hidden_size = hidden_size

        # Embedding for spike_nums (categorical input with 13 classes)
        self.num_embeddings = nn.Embedding(13, num_embed_size)

        # Networks for processing spike_durations, spike_magnitudes, and time_sin_cos
        self.duration_net = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, duration_size),
            nn.ReLU()
        )

        self.mag_net = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, mag_size),
            nn.ReLU()
        )

        self.time_net = nn.Sequential(
            nn.Linear(2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, time_size),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(num_embed_size + duration_size + mag_size + time_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, spike_nums, spike_durations, spike_magnitudes, time_sin_cos):
        num = self.num_embeddings(spike_nums)
        num = num.expand(-1, 12, -1)

        # Parallelize the following computations
        duration, magnitude, time = self.duration_net(spike_durations.unsqueeze(-1)), self.mag_net(spike_magnitudes.unsqueeze(-1)), self.time_net(time_sin_cos)

        x = torch.cat([num, duration, magnitude, time], dim=-1)
        out = self.fc(x)
        return out


class s_decoder(nn.Module):
    def __init__(self, latent_size, hidden_size, z_size, mix_size, id_size, weather_size, date_size):
        super(s_decoder, self).__init__()

        # Networks for processing ID_statistics, weather, and date_sin_cos
        self.id_net = nn.Sequential(
            nn.Linear(6, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, id_size),
            nn.ReLU()
        )

        self.weather_net = nn.Sequential(
            nn.Linear(3, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, weather_size),
            nn.ReLU()
        )

        self.date_net = nn.Sequential(
            nn.Linear(2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, date_size),
            nn.ReLU()
        )

        self.z_f_net = nn.Sequential(
            nn.Linear(12 * latent_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, z_size),
            nn.ReLU()
        )

        self.mix_net = nn.Sequential(
            nn.Linear(id_size + weather_size + date_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, mix_size),
            nn.ReLU()
        )

        # Fully connected layers for reconstructing the output
        self.spike_num_recon = nn.Sequential(
            nn.Linear(mix_size + z_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 13)
        )

        self.spike_durations_recon = nn.Sequential(
            nn.Linear(mix_size + z_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 12),
        )

        self.spike_magnitudes_recon = nn.Sequential(
            nn.Linear(mix_size + z_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 12),
        )

        self.time_sin_cos_recon = nn.Sequential(
            nn.Linear(id_size + weather_size + date_size + latent_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2)
        )
    
    def forward(self, z, ID_statistics, weather, date_sin, date_cos):
        # Parallelize the following computations
        id, w, date_sin_cos = self.id_net(ID_statistics), self.weather_net(weather), self.date_net(torch.cat([date_sin, date_cos], dim=-1))

        z_f = self.z_f_net(z.view(z.size(0), -1))

        mix = self.mix_net(torch.cat([id, w, date_sin_cos], dim=-1))
        concat = torch.cat([z_f, mix], dim=-1)

        id_12 = id.unsqueeze(1).expand(-1, 12, -1)
        w_12 = w.unsqueeze(1).expand(-1, 12, -1)
        date_12 = date_sin_cos.unsqueeze(1).expand(-1, 12, -1)

        concat_12 = torch.cat([z, id_12, w_12, date_12], dim=-1)

        # Reconstruct the output
        spike_num_recon = self.spike_num_recon(concat)
        spike_durations_recon = self.spike_durations_recon(concat)
        spike_magnitudes_recon = self.spike_magnitudes_recon(concat)
        time_sin_cos_recon = self.time_sin_cos_recon(concat_12)

        return spike_num_recon, spike_durations_recon, spike_magnitudes_recon, time_sin_cos_recon


class s_VAE(nn.Module):
    def __init__(self, num_embed_size, hidden_size_e, duration_size, mag_size, time_size, latent_size, hidden_size_d, z_size, mix_size, id_size, weather_size, date_size):
        super(s_VAE, self).__init__()

        self.encoder = s_encoder(num_embed_size, hidden_size_e, duration_size, mag_size, time_size, output_size=latent_size * 2)
        self.decoder = s_decoder(latent_size, hidden_size_d, z_size, mix_size, id_size, weather_size, date_size)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def forward(self, spike_nums, spike_durations, spike_magnitudes, ID_statistics, weather, date_sin, date_cos, time_sin_cos):
        encoder_out = self.encoder(spike_nums, spike_durations, spike_magnitudes, time_sin_cos)
        mu, logvar = encoder_out.chunk(2, dim=2)

        # Reparameterization trick to sample from the latent space
        z = self.reparameterize(mu, logvar)

        # Decode the latent variable to reconstruct the input
        spike_num_recon, spike_durations_recon, spike_magnitudes_recon, time_sin_cos_recon = self.decoder(z, ID_statistics, weather, date_sin, date_cos)

        return spike_num_recon, spike_durations_recon, spike_magnitudes_recon, time_sin_cos_recon, mu, logvar

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 1e-3)

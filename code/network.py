import torch
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class m_Encoder(nn.Module):
    '''
    Encoder for the medoid VAE

    To encode the energy profile [batch_size, sequence_length, 1] into a latent space [batch_size, sequence_length, latent_dim * 2]

    Args:
    hidden_size: int
        The size of the hidden layer for the fully connected layers
    lstm_size: int
        The size of the hidden layer for the LSTM
    num_layers: int
        The number of layers for the LSTM
    output_size: int
        The size of the output, the latent dimension
    '''
    def __init__(self, hidden_size, lstm_size, num_layers, output_size):
        super(m_Encoder, self).__init__()

        self.hidden_size = hidden_size

        self.num_layers = num_layers
        self.lstm_size = lstm_size

        self.lstm = nn.LSTM(1, lstm_size, num_layers, batch_first=True, bidirectional=True)

        self.fc = nn.Linear(lstm_size * 2, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.lstm_size).to(device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.lstm_size).to(device)

        out, _ = self.lstm(x, (h0, c0))

        out = self.fc(out)

        return out

class m_Decoder(nn.Module):
    '''
    Decoder for the medoid VAE

    To decode the latent space into the energy profile [batch_size, sequence_length, 1]

    Args:
    input_size: int
        The size of the input, the latent dimension
    hidden_size: int
        The size of the hidden layer for the fully connected layers
    cluster_embed_size: int
        The size of the embedding for the cluster, 10 clusters in total
    lstm_size: int
        The size of the hidden layer for the LSTM
    num_layers: int
        The number of layers for the LSTM

    
    '''
    def __init__(self, input_size, hidden_size, cluster_embed_size, time_size, lstm_size, num_layers):
        super(m_Decoder, self).__init__()

        self.hidden_size = hidden_size

        self.lstm_size = lstm_size
        self.num_layers = num_layers

        self.Embedding = nn.Embedding(20, cluster_embed_size)

        self.lstm = nn.LSTM(input_size, lstm_size, num_layers, batch_first=True, bidirectional=True)

        self.time_net = nn.Sequential(
            nn.Linear(4, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, time_size),
            nn.ReLU()
        )

        self.fc1 = nn.Sequential(
            nn.Linear(lstm_size * 2 + cluster_embed_size + time_size + 3, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 128),
            nn.ReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(128 + cluster_embed_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.ReLU()
        )

    def forward(self, x, weather, cluster, time):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.lstm_size).to(device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.lstm_size).to(device)

        out, _ = self.lstm(x, (h0, c0))

        cluster = cluster.squeeze(2)
        c = self.Embedding(cluster)
        t = self.time_net(time)

        out = torch.cat((out, weather, c, t), dim=-1)
        out = self.fc1(out)
        out = torch.cat((out, c), dim=-1)
        out = self.fc2(out)

        return out
    
class m_VAE(nn.Module):
    '''
    VAE for the medoid

    Args:
    encoder_hidden: int
        The size of the hidden layer for the encoder
    encoder_lstm_size: int
        The size of the hidden layer for the LSTM in the encoder
    encoder_lstm_layers: int
        The number of layers for the encoder
    latent_dim: int
        The size of the latent dimension
    decoder_hidden_size: int
        The size of the hidden layer for the decoder
    embed_size: int
        The size of the embedding for the cluster, 10 clusters in total
    decoder_lstm_size: int
        The size of the hidden layer for the LSTM in the decoder
    decoder_lstm_layers: int
        The number of layers for the decoder
    
    '''
    def __init__(self, encoder_hidden, encoder_lstm_size, encoder_lstm_layers, latent_dim, decoder_hidden_size, embed_size, time_size, decoder_lstm_size, decoder_lstm_layers):
        super(m_VAE, self).__init__()


        self.lstm_encoder = m_Encoder(encoder_hidden, encoder_lstm_layers, encoder_lstm_size, latent_dim * 2)
        
        self.lstm_decoder = m_Decoder(latent_dim, decoder_hidden_size, embed_size, time_size, decoder_lstm_size, decoder_lstm_layers)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, weather, cluster, time, real):
        out = self.lstm_encoder(real)

        mu, logvar = out.chunk(2, dim=2)
        z = self.reparameterize(mu, logvar)

        r = self.lstm_decoder(z, weather, cluster, time)
        return r, mu, logvar
    
    
# VAE for spikes
# This is to improve the quality of the reconstructions for the spikes
# Encode the regular energy profile, decode with the spike profile
class m2s_Encoder(nn.Module):
    '''
    Encoder for the regular-to-spike VAE
    
    To encode the energy profile [batch_size, sequence_length, 1] into a latent space [batch_size, sequence_length, latent_dim * 2]
    
    Args:
    hidden_size: int
        The size of the hidden layer for the fully connected layers
    lstm_size: int
        The size of the hidden layer for the LSTM
    num_layers: int
        The number of layers for the LSTM
    output_size: int
        The size of the output, the latent dimension
    '''
    def __init__(self, hidden_size, lstm_size, num_layers, output_size):
        super(m2s_Encoder, self).__init__()

        self.hidden_size = hidden_size

        self.lstm_size = lstm_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(1, lstm_size, num_layers, batch_first=True, bidirectional=True)

        self.fc1 = nn.Sequential(
            nn.Linear(lstm_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.ReLU()
        )

    def forward(self, x):

        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.lstm_size).to(device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.lstm_size).to(device)

        out, _ = self.lstm(x, (h0, c0))

        out = self.fc1(out)
        out = self.fc2(out)

        return out
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.orthogonal_(param)  # For input-hidden weights
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param)  # For hidden-hidden weights
                    elif 'bias' in name:
                        nn.init.constant_(param, 1e-3)  # Biases initialized to a small constant
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 1e-3)

class m2s_Decoder(nn.Module):
    '''
    Decoder for the medoid-to-regular VAE
    
    To decode the latent space into the energy profile [batch_size, sequence_length, 1]

    Args:
    input_size: int
        The size of the input, the latent dimension
    hidden_size: int
        The size of the hidden layer for the fully connected layers
    embed_size_cluster: int
        The size of the embedding for the cluster, 10 clusters in total
    embed_size_spike: int
        The size of the embedding for the spike, 4 types of spikes in total
    time_size: int
        The size of the output layer for the time
    statistics_size: int
        The size of the output layer for the user statistics
    gradient_size: int
        The size of the output layer for the gradient_net
    lstm_size: int
        The size of the hidden layer for the LSTM
    num_layers: int
        The number of layers for the LSTM
    lstm_dropout: float
        The dropout rate for the LSTM
    fc_dropout: float
        The dropout rate for the fully connected layers
    '''
    def __init__(self, input_size, hidden_size, embed_size_cluster, embed_size_spike, time_size, statistics_size, gradient_size, num_layers, lstm_size, lstm_dropout, fc_dropout):
        super(m2s_Decoder, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm_size = lstm_size

        spike_magnitude_size = 1
        gradient_o = 1
        statistics_o = 6
        weather_o = 3
        
        self.Embedding_spike = nn.Embedding(5, embed_size_spike)
        self.Embedding_cluster = nn.Embedding(20, embed_size_cluster)

        self.lstm = nn.LSTM(input_size, lstm_size, num_layers, batch_first=True, dropout=lstm_dropout if num_layers > 1 else 0, bidirectional=True)

        self.time_net = nn.Sequential(
            nn.Linear(4, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, time_size),
            nn.ReLU()
        )

        self.statistical_net = nn.Sequential(
            nn.Linear(6, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, statistics_size),
            nn.ReLU()
        )

        self.g_net = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, gradient_size),
            nn.ReLU()
        )

        self.fc1 = nn.Sequential(
            nn.Linear(lstm_size * 2 + embed_size_cluster + embed_size_spike + weather_o + statistics_o + time_size +  spike_magnitude_size + statistics_size + gradient_size + gradient_o, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(fc_dropout),
            nn.ReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(hidden_size + embed_size_spike + 1 + gradient_size + 1, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(fc_dropout),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.ReLU()
            # nn.LayerNorm(hidden_size),
            # nn.ReLU()
        )

        # self.fc3 = nn.Sequential(
        #     nn.Linear(hidden_size, hidden_size),
        #     nn.LayerNorm(hidden_size),
        #     nn.ReLU(),
        #     nn.Linear(hidden_size, 1),
        #     nn.ReLU()
        # )

    def forward(self, x, weather, cluster, time, statistical, spike_type, spike_magnitude):

        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.lstm_size).to(device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.lstm_size).to(device)
        out, _ = self.lstm(x, (h0, c0))

        sp = self.Embedding_spike(spike_type).squeeze(2)
        c = self.Embedding_cluster(cluster).squeeze(2)
        t = self.time_net(time)
        s = self.statistical_net(statistical)

        gradient = statistical[:, :, -1:]
        g = self.g_net(gradient)
        
        out = torch.cat((out, weather, c, t, s, statistical, g, gradient, sp, spike_magnitude), dim=-1)
        out = self.fc1(out)
        out = torch.cat((out, sp, g, gradient, spike_magnitude), dim=-1)
        out = self.fc2(out)
        # out = self.fc3(out)
        return out
    
class m2s_VAE(nn.Module):
    '''
    Regular-to-spike VAE

    Args:
    encoder_hidden: int
        The size of the hidden layer for the fully connected layers in the encoder
    encoder_lstm_size: int
        The size of the hidden layer for the LSTM in the encoder
    encoder_lstm_layers: int
        The number of layers for the LSTM in the encoder
    latent_dim: int
        The size of the latent dimension
    decoder_hidden: int
        The size of the hidden layer for the fully connected layers in the decoder
    embed_size_cluster: int
        The size of the embedding for the cluster, 10 clusters in total
    embed_size_spike: int
        The size of the embedding for the spike, 4 types of spikes in total
    time_size: int
        The size of the output layer for the time
    statistics_size: int
        The size of the output layer for the user statistics
    gradient_size: int
        The size of the output layer for the gradient_net
    decoder_lstm_size: int
        The size of the hidden layer for the LSTM in the decoder
    decoder_layers: int
        The number of layers for the LSTM in the decoder
    decoder_lstm_dropout: float
        The dropout rate for the LSTM in the decoder
    fc_dropout: float
        The dropout rate for the fully connected layers
    '''
    def __init__(self, 
                 encoder_hidden, encoder_lstm_size, encoder_lstm_layers, 
                 latent_dim, 
                 decoder_hidden, embed_size_cluster, embed_size_spike, time_size, statistics_size, gradient_size, decoder_lstm_size, decoder_layers, decoder_lstm_dropout, fc_dropout):
        
        super(m2s_VAE, self).__init__()

        self.lstm_encoder = m2s_Encoder(encoder_hidden, encoder_lstm_size, encoder_lstm_layers, latent_dim * 2)
        
        self.lstm_decoder = m2s_Decoder(
            latent_dim, 
            decoder_hidden, 
            embed_size_cluster, embed_size_spike, 
            time_size, statistics_size, gradient_size,
            decoder_layers, decoder_lstm_size, decoder_lstm_dropout,
            fc_dropout
            )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, weather, cluster, time, statistical, spike_type, spike_magnitude):
    # def forward(self, x, statistical, spike_type, spike_magnitude):
        out = self.lstm_encoder(x)
        mu, logvar = out.chunk(2, dim=2)
        z = self.reparameterize(mu, logvar)

        r = self.lstm_decoder(z, weather, cluster, time, statistical, spike_type, spike_magnitude)
        #r = self.lstm_decoder(z, statistical, spike_type, spike_magnitude)
        return r, mu, logvar
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.orthogonal_(param)  # For input-hidden weights
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param)  # For hidden-hidden weights
                    elif 'bias' in name:
                        nn.init.constant_(param, 1e-3)  # Biases initialized to a small constant
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 1e-3)
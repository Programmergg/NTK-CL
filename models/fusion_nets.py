import torch
import torch.nn as nn
from torch.nn import functional as F

class FusionNetwork(nn.Module):
    def __init__(self, embedding_dim):
        super(FusionNetwork, self).__init__()
        self.fc1 = nn.Linear(embedding_dim * 2, embedding_dim)

    def forward(self, channel_emb, patch_emb):
        concatenated = torch.cat((channel_emb, patch_emb), dim=1)
        fused = torch.relu(self.fc1(concatenated))
        fused = 0.5 * fused + channel_emb + patch_emb
        return fused

class MultimodalFusion(nn.Module):
    def __init__(self, embedding_dim):
        super(MultimodalFusion, self).__init__()
        self.fc1 = nn.Linear(embedding_dim, embedding_dim)
        self.fc2 = nn.Linear(embedding_dim * 3, embedding_dim)
        self.interact = nn.Parameter(torch.randn(embedding_dim))

    def forward(self, channel_emb, patch_emb):
        interaction = F.relu(self.fc1(torch.mul(channel_emb, patch_emb)))
        interaction = interaction * self.interact
        concatenated = torch.cat((channel_emb, patch_emb, interaction), dim=1)
        fused = F.relu(self.fc2(concatenated))
        fused = 0.1 * fused + channel_emb + patch_emb
        return fused

class FusionVAE(nn.Module):
    def __init__(self, embedding_dim):
        super(FusionVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim // 2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim // 2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )

    def forward(self, channel_emb, patch_emb):
        x = torch.cat((channel_emb, patch_emb), dim=1)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        fused = 0.1 * decoded + channel_emb + patch_emb
        return fused

class AttentionFusion(nn.Module):
    def __init__(self, embedding_dim, num_heads=8):
        super(AttentionFusion, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        assert self.head_dim * num_heads == embedding_dim, "embedding_dim must be divisible by num_heads"
        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, channel_emb, patch_emb):
        batch_size = channel_emb.size(0)
        # Query, Key, Value computations
        queries = self.query(patch_emb).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        keys = self.key(channel_emb).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        values = self.value(channel_emb).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        # Scaled Dot-Product Attention
        attention_scores = torch.softmax((queries @ keys.transpose(-2, -1)) / (self.head_dim ** 0.5), dim=-1)
        attention_scores = self.dropout(attention_scores)
        fused = attention_scores @ values
        fused = fused.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)
        fused = fused.view(batch_size, self.num_heads * self.head_dim)
        # Apply normalization and residual connection
        fused = 0.5 * fused + 1.0 * channel_emb + 0.1 * patch_emb
        return fused

class RecurrentFusion(nn.Module):
    def __init__(self, embedding_dim, hidden_dim=None, num_layers=1, rnn_type='LSTM', batch_first=True):
        super(RecurrentFusion, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim if hidden_dim is not None else embedding_dim
        self.num_layers = num_layers
        self.batch_first = batch_first
        if rnn_type.upper() == 'LSTM':
            self.rnn = nn.LSTM(input_size=embedding_dim, hidden_size=self.hidden_dim, num_layers=num_layers, batch_first=batch_first)
        elif rnn_type.upper() == 'GRU':
            self.rnn = nn.GRU(input_size=embedding_dim, hidden_size=self.hidden_dim, num_layers=num_layers, batch_first=batch_first)
        else:
            raise ValueError("Unsupported RNN type. Please choose 'LSTM' or 'GRU'.")

    def forward(self, channel_emb, patch_emb):
        x = torch.stack([channel_emb, patch_emb], dim=1).reshape(-1, 2, self.embedding_dim)
        _, hn = self.rnn(x)
        if isinstance(self.rnn, nn.LSTM):
            hn = hn[0].squeeze(0)
        else:
            hn = hn.squeeze(0)
        fused_output = hn + channel_emb + patch_emb
        return fused_output

class StateSpaceModule(nn.Module):
    def __init__(self, state_dim, num_states):
        super(StateSpaceModule, self).__init__()
        self.state_dim = state_dim
        self.num_states = num_states
        self.transition_matrices = nn.ParameterList([
            nn.Parameter(torch.randn(state_dim, state_dim)) for _ in range(num_states)
        ])
        self.observation_matrices = nn.ParameterList([
            nn.Parameter(torch.randn(state_dim, state_dim)) for _ in range(num_states)
        ])

    def forward(self, x, state_idx):
        batch_size = x.shape[0]
        outputs = []
        for i in range(batch_size):
            A = self.transition_matrices[state_idx[i]]
            C = self.observation_matrices[state_idx[i]]
            state = F.relu(torch.matmul(x[i:i + 1], A))  # Simple state transition with non-linearity
            observation = torch.matmul(state, C)  # Observation from state
            outputs.append(observation)
        return torch.cat(outputs, dim=0)

class MambaRecurrentFusion(nn.Module):
    def __init__(self, embedding_dim, state_dim=3072, num_states=5, num_layers=1):
        super(MambaRecurrentFusion, self).__init__()
        self.embedding_dim = embedding_dim
        self.state_dim = state_dim
        self.num_states = num_states
        self.num_layers = num_layers
        self.state_space_module = StateSpaceModule(state_dim, num_states)
        self.rnn = nn.GRU(input_size=state_dim, hidden_size=embedding_dim, num_layers=num_layers, batch_first=True)
        self.state_selector = nn.Linear(2 * embedding_dim, num_states)  # Decides which state to use

    def forward(self, channel_emb, patch_emb):
        combined_emb = torch.cat([channel_emb, patch_emb], dim=-1)
        state_idx = torch.argmax(self.state_selector(combined_emb), dim=-1)  # Choose state based on input
        # Apply the selected state space transformation
        transformed = self.state_space_module(combined_emb, state_idx)
        x = transformed.unsqueeze(1)  # Prepare for RNN
        _, hn = self.rnn(x)
        hn = hn.squeeze(0) + channel_emb + patch_emb
        return hn
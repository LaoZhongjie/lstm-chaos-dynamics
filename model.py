import torch
import torch.nn as nn
import torch.nn.functional as F
import config
from seed_utils import HierarchicalSeedManager

class LSTM(nn.Module):
    def __init__(self, vocab_size=4000, embedding_dim=32, hidden_size=60, num_classes=1, 
                 embedding_weights=None, fc_weights=None, seed_manager=None):
        super(LSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim
        self.seed_manager = seed_manager or HierarchicalSeedManager(config.RANDOM_SEED)
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if embedding_weights is not None:
            print("\n[lstm]Embedding Layer: loaded, not trainable... ")
            self.embedding.weight = nn.Parameter(torch.tensor(embedding_weights, dtype=torch.float32), requires_grad=False)
        else:
            print("\n[lstm]Embedding Layer: uniform init")
            with self.seed_manager.local_seed("model.init.embedding"):
                nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
            self.embedding.weight.data[0].fill_(0)  # Padding token
        
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_size,
            batch_first=True,
            bidirectional=False
        )
        
        self.fc = nn.Linear(hidden_size, num_classes)
        if fc_weights is not None:
            print("[lstm]FC Layer: loaded, not trainable...\n")
            self.fc.weight = nn.Parameter(torch.tensor(fc_weights[0], dtype=torch.float32), requires_grad=False)
            self.fc.bias = nn.Parameter(torch.tensor(fc_weights[1], dtype=torch.float32), requires_grad=False)
        else:
            print("[lstm]FC Layer: micro normal init...\n")
            with self.seed_manager.local_seed("model.init.fc"):
                nn.init.normal_(self.fc.weight, mean=0.0, std=1e-2)
            nn.init.zeros_(self.fc.bias)

        self._init_recurrent_weights()
        
    def _init_recurrent_wweights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight_hh' in name:
                hidden_size = param.shape[1]
                for i in range(4):
                    gate_weight = param.data[i * hidden_size:(i + 1) * hidden_size]
                    with self.seed_manager.local_seed(f"model.init.lstm.{name}.gate_{i}"):
                        nn.init.orthogonal_(gate_weight)
                param.data *= 0.2

            elif 'weight_ih' in name:
                with self.seed_manager.local_seed(f"model.init.lstm.{name}"):
                    nn.init.normal_(param, mean=0.0, std=1e-2)

            elif 'bias' in name:
                param.data.zero_()
                n = param.size(0)
                # forget gate bias
                param.data[n//4:n//2].fill_(-1.0)
    
    def forward(self, x, lengths, hidden=None):
        batch_size = x.size(0)
        
        embedded = self.embedding(x)
        
        if hidden is None:
            hidden = self.init_hidden(batch_size)
        
        lstm_out, hidden = self.lstm(embedded, hidden)
        
        # Use last true output for classification
        idx = (lengths - 1).clamp(min=0)
        last_output = lstm_out[torch.arange(batch_size, device=x.device), idx, :]
        
        output = self.fc(last_output)
        
        return output, hidden
    
    def init_hidden(self, batch_size):

        device = next(self.parameters()).device
        h_0 = torch.zeros(1, batch_size, self.hidden_size, device=device)
        c_0 = torch.zeros(1, batch_size, self.hidden_size, device=device)
        return (h_0, c_0)
    
    def get_hidden_output(self, x, hidden=None):
        batch_size = x.size(0)
        
        # Embedding
        embedded = self.embedding(x)
        
        # Initialize hidden state if not provided
        if hidden is None:
            hidden = self.init_hidden(batch_size)
        
        # lstm forward pass
        h_t, final_hc = self.lstm(embedded, hidden)
        
        return h_t, final_hc
    
    def continue_iteration(self, initial_hc, timesteps, input_dim=32):
        device = next(self.parameters()).device
        batch_size = initial_hc[0].size(1)
        
        # Create zero input for continued iteration
        zero_input = torch.zeros(batch_size, timesteps, input_dim, device=device)
        
        # Continue lstm iteration
        hidden_states, _ = self.lstm(zero_input, initial_hc)
        
        return hidden_states

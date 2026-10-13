import torch
import torch.nn as nn
import numpy as np
import random
from torch.nn.functional import softmax, relu

def encoder(input_dimension, output_dimension):
    l1 = nn.Linear(input_dimension, output_dimension)
    l2 = nn.ReLU()
    model = nn.Sequential(l1, l2)
    return model

class IQN_Policy(nn.Module):

    def __init__(self, self_dimension, static_dimension, dynamic_dimension, self_feature_dimension, static_feature_dimension, dynamic_feature_dimension, hidden_dimension, action_size, device='cpu', seed=0):
        super().__init__()
        self.self_dimension = self_dimension
        self.static_dimension = static_dimension
        self.dynamic_dimension = dynamic_dimension
        self.self_feature_dimension = self_feature_dimension
        self.static_feature_dimension = static_feature_dimension
        self.dynamic_feature_dimension = dynamic_feature_dimension
        self.concat_feature_dimension = self_feature_dimension + static_feature_dimension + dynamic_feature_dimension
        self.hidden_dimension = hidden_dimension
        self.action_size = action_size
        self.device = device
        self.seed_id = seed
        self.seed = torch.manual_seed(seed)
        self.self_encoder = encoder(self_dimension, self_feature_dimension)
        self.static_encoder = encoder(static_dimension, static_feature_dimension)
        self.dynamic_encoder = encoder(dynamic_dimension, dynamic_feature_dimension)
        self.K = 32
        self.n = 64
        self.pis = torch.FloatTensor([np.pi * i for i in range(self.n)]).view(1, 1, self.n).to(device)
        self.cos_embedding = nn.Linear(self.n, self.concat_feature_dimension)
        self.hidden_layer = nn.Linear(self.concat_feature_dimension, hidden_dimension)
        self.hidden_layer_2 = nn.Linear(hidden_dimension, hidden_dimension)
        self.output_layer = nn.Linear(hidden_dimension, action_size)

    def calc_cos(self, batch_size, num_tau=8, cvar=1.0):
        taus = torch.rand(batch_size, num_tau).to(self.device).unsqueeze(-1)
        taus = taus * cvar
        cos = torch.cos(taus * self.pis)
        assert cos.shape == (batch_size, num_tau, self.n), 'cos shape is incorrect'
        return (cos, taus)

    def forward(self, x, num_tau=8, cvar=1.0):
        batch_size = x.shape[0]
        self_states = x[:, :self.self_dimension]
        static_states = x[:, self.self_dimension:self.self_dimension + self.static_dimension]
        dynamic_states = x[:, self.self_dimension + self.static_dimension:]
        self_features = self.self_encoder(self_states)
        static_features = self.static_encoder(static_states)
        dynamic_features = self.dynamic_encoder(dynamic_states)
        features = torch.cat((self_features, static_features, dynamic_features), 1)
        cos, taus = self.calc_cos(batch_size, num_tau, cvar)
        cos = cos.view(batch_size * num_tau, self.n)
        cos_features = relu(self.cos_embedding(cos)).view(batch_size, num_tau, self.concat_feature_dimension)
        features = (features.unsqueeze(1) * cos_features).view(batch_size * num_tau, self.concat_feature_dimension)
        features = relu(self.hidden_layer(features))
        features = relu(self.hidden_layer_2(features))
        quantiles = self.output_layer(features)
        return (quantiles.view(batch_size, num_tau, self.action_size), taus)

class IEQN_Policy(nn.Module):

    def __init__(self, self_dimension, static_dimension, dynamic_dimension, self_feature_dimension, static_feature_dimension, dynamic_feature_dimension, hidden_dimension, action_size, mapper_hidden_dim=64, device='cpu', seed=0):
        super().__init__()
        self.self_dimension = self_dimension
        self.static_dimension = static_dimension
        self.dynamic_dimension = dynamic_dimension
        self.self_feature_dimension = self_feature_dimension
        self.static_feature_dimension = static_feature_dimension
        self.dynamic_feature_dimension = dynamic_feature_dimension
        self.concat_feature_dimension = self_feature_dimension + static_feature_dimension + dynamic_feature_dimension
        self.hidden_dimension = hidden_dimension
        self.action_size = action_size
        self.device = device
        self.seed_id = seed
        self.set_seed(seed)
        self.self_encoder = encoder(self_dimension, self_feature_dimension)
        self.static_encoder = encoder(static_dimension, static_feature_dimension)
        self.dynamic_encoder = encoder(dynamic_dimension, dynamic_feature_dimension)
        self.n = 64
        self.pis = torch.FloatTensor([np.pi * i for i in range(self.n)]).view(1, 1, self.n).to(device)
        self.cos_embedding = nn.Linear(self.n, self.concat_feature_dimension)
        self.hidden_layer = nn.Linear(self.concat_feature_dimension, hidden_dimension)
        self.hidden_layer_2 = nn.Linear(hidden_dimension, hidden_dimension)
        self.output_layer = nn.Linear(hidden_dimension, action_size)
        self.mapper_hidden_dim = mapper_hidden_dim
        self.mapper = nn.Sequential(nn.Linear(self.concat_feature_dimension, self.mapper_hidden_dim), nn.ReLU(), nn.Linear(self.mapper_hidden_dim, self.concat_feature_dimension))

    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def calc_cos(self, batch_size, num_tau=8, cvar=1.0):
        taus = torch.rand(batch_size, num_tau).to(self.device).unsqueeze(-1)
        taus = taus * cvar
        cos = torch.cos(taus * self.pis)
        assert cos.shape == (batch_size, num_tau, self.n), 'cos shape is incorrect'
        return (cos, taus)

    def forward(self, x, num_tau=8, cvar=1.0):
        batch_size = x.shape[0]
        self_states = x[:, :self.self_dimension]
        static_states = x[:, self.self_dimension:self.self_dimension + self.static_dimension]
        dynamic_states = x[:, self.self_dimension + self.static_dimension:]
        self_features = self.self_encoder(self_states)
        static_features = self.static_encoder(static_states)
        dynamic_features = self.dynamic_encoder(dynamic_states)
        state_features = torch.cat((self_features, static_features, dynamic_features), 1)
        cos, taus = self.calc_cos(batch_size, num_tau, cvar)
        cos = cos.view(batch_size * num_tau, self.n)
        cos_features = relu(self.cos_embedding(cos)).view(batch_size, num_tau, self.concat_feature_dimension)
        state_fraction_features_expectile = (state_features.unsqueeze(1) * cos_features).view(batch_size * num_tau, self.concat_feature_dimension)
        hidden_out_expectile = relu(self.hidden_layer(state_fraction_features_expectile))
        hidden_out_expectile = relu(self.hidden_layer_2(hidden_out_expectile))
        expectiles = self.output_layer(hidden_out_expectile)
        mapped_cos_features = cos_features + self.mapper(cos_features)
        state_fraction_features_quantile = (state_features.unsqueeze(1) * mapped_cos_features).view(batch_size * num_tau, self.concat_feature_dimension)
        hidden_out_quantile = relu(self.hidden_layer(state_fraction_features_quantile))
        hidden_out_quantile = relu(self.hidden_layer_2(hidden_out_quantile))
        quantiles = self.output_layer(hidden_out_quantile)
        return {'expectiles': expectiles.view(batch_size, num_tau, self.action_size), 'quantiles': quantiles.view(batch_size, num_tau, self.action_size), 'taus': taus}

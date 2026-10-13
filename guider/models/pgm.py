import torch
import torch.nn as nn

class ImprovedDualTaskTransformer(nn.Module):

    def __init__(self):
        super().__init__()
        self.self_car_dim = 2
        self.goal_dim = 2
        self.static_obs_dim = 3
        self.dynamic_veh_dim = 4
        self.self_car_embed = self._build_embedding(self.self_car_dim)
        self.goal_embed = self._build_embedding(self.goal_dim)
        self.static_obs_embed = self._build_embedding(self.static_obs_dim)
        self.dynamic_veh_embed = self._build_embedding(self.dynamic_veh_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, 12, 128))
        encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=8, dim_feedforward=512, dropout=0.3)
        self.transformer = nn.TransformerEncoder(encoder_layer, 4)
        self.fine_classifier = self._build_fine_head()
        self.coarse_classifier = self._build_coarse_head()

    def _build_embedding(self, input_dim):
        return nn.Sequential(nn.Linear(input_dim, 64), nn.GELU(), nn.Linear(64, 128))

    def _build_fine_head(self):
        return nn.Sequential(nn.Linear(128, 512), nn.GELU(), nn.Dropout(0.3), nn.Linear(512, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.3 / 2), nn.Linear(256, 9))

    def _build_coarse_head(self):
        return nn.Sequential(nn.Linear(128, 256), nn.GELU(), nn.Dropout(0.3 / 2), nn.Linear(256, 3))

    def forward(self, x):
        bs = x.size(0)
        components = [x[:, :self.self_car_dim].unsqueeze(1), x[:, self.self_car_dim:self.self_car_dim + self.goal_dim].unsqueeze(1), x[:, self.self_car_dim + self.goal_dim:self.self_car_dim + self.goal_dim + 5 * self.static_obs_dim].view(bs, 5, self.static_obs_dim), x[:, self.self_car_dim + self.goal_dim + 5 * self.static_obs_dim:].view(bs, 5, self.dynamic_veh_dim)]
        embeddings = []
        for embed, data in zip([self.self_car_embed, self.goal_embed, self.static_obs_embed, self.dynamic_veh_embed], components):
            embeddings.append(embed(data))
        embeddings = torch.cat(embeddings, dim=1) + self.pos_embed
        encoded = self.transformer(embeddings.permute(1, 0, 2)).permute(1, 0, 2)
        pooled = encoded[:, 0, :]
        return (self.fine_classifier(pooled), self.coarse_classifier(pooled))

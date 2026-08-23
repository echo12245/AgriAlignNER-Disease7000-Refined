import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet152

class RegionLevelVisualEncoder(nn.Module):
    def __init__(self, output_dim=768):
        super().__init__()
        self.resnet = resnet152(weights=None)
        local_weights = torch.load('pretrained_model/resnet152.pth', map_location='cpu')
        self.resnet.load_state_dict(local_weights, strict=False)
        self.output_dim = output_dim
        self.global_proj = nn.Linear(2048, output_dim)
        self.local_proj = nn.Linear(2048, output_dim)
        self.grid_size = 7
    def forward(self, images, aux_images=None):
        batch_size = images.shape[0]
        features = self._extract_features(images)
        global_feat = F.adaptive_avg_pool2d(features, 1).view(batch_size, -1)
        global_feat = self.global_proj(global_feat).unsqueeze(1)
        local_feats = features.view(batch_size, 2048, -1).permute(0, 2, 1)
        local_feats = self.local_proj(local_feats)
        if aux_images is not None:
            num_aux = aux_images.shape[1]
            aux_images_flat = aux_images.view(-1, 3, 224, 224)
            aux_features = self._extract_features(aux_images_flat)
            del aux_images_flat
            aux_features = F.adaptive_avg_pool2d(aux_features, 1).view(batch_size, num_aux, -1)
            aux_features = self.local_proj(aux_features)
            region_features = torch.cat([global_feat, local_feats, aux_features], dim=1)
            del global_feat, local_feats, aux_features
        else:
            region_features = torch.cat([global_feat, local_feats], dim=1)
            del global_feat, local_feats

        return region_features

    def _extract_features(self, x):
        with torch.no_grad():
            for name, layer in self.resnet.named_children():
                if name in ['fc', 'avgpool']:
                    continue
                x = layer(x)
        return x.detach().clone()
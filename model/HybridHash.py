import torch
import torch.nn as nn
import timm

class HybridHash(nn.Module):
    def __init__(self, config, num_levels=3, embed_dims=(128, 256, 512), 
                 num_heads=(4, 8, 16), depths=(2, 2, 15)):
        super(HybridHash, self).__init__()
        # Tạo model NesT base với các tham số được chỉ định
        self.backbone = timm.create_model(
            'nest_base',
            pretrained=False,
            # num_levels=num_levels,
            # embed_dims=embed_dims,
            # num_heads=num_heads,
            # depths=depths,
            img_size=config['img_size'],
            patch_size=config['patch_size'],
            in_chans=config['in_chans']
        )
        
        # Load pretrained weights nếu có
        if config['pretrained_dir']:
            state_dict = torch.load(config['pretrained_dir'])
            self.backbone.load_state_dict(state_dict)
        
        # Thay đổi layer cuối cùng để phù hợp với bài toán hash
        num_features = self.backbone.head.in_features
        self.backbone.head = nn.Linear(num_features, config['bit_list'][-1])  # Sử dụng số bit lớn nhất trong bit_list
        
    def forward(self, x):
        features = self.backbone(x)
        return features 
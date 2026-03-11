import torch
from models.multi_view.encoder.multi_encoder import MultiEncoder

if __name__ == "__main__":
    encoder_configs = {
    "dino": {
        "fuse_layer_encoder": [[0,1,2,3,4,5,6,7]],
        "target_layers": [2,3,4,5,6,7,8,9],
        "backbone": "dinov2reg_vit_base_14",
        "n": 10,          # 至少保证 outputs[0..9] 可取
        "norm": True,
        "trainable": False,
    },
    "clip": {
        "fuse_layer_encoder": [[0,1,2,3,4,5,6,7]],
        "target_layers": [2,3,4,5,6,7,8,9],
        "arch": "ViT-B-16",
        "pretrained": "openai",
        "n": 10,          # 同理至少 10
        "norm": False,
        "trainable": False,
    },
    "resnet": {
        "fuse_layer_encoder": [[2,3]],
        "target_layers": [2,3],  # 以当前 ResNetExtractor，这里通常只能是 0..3
        "arch": "wide_resnet50_2",
        "pretrained": True,
        "n": 4,
        "trainable": False,
    },
    "donut": {
        "fuse_layer_encoder": [[2,3]],
        "target_layers": [2,3],  # DonutExtractor 通常也是 4 个 stage -> 0..3
        "model_name": "naver-clova-ix/donut-base",
        "local_files_only": True,   # 如果你要完全离线
        "do_resize": True,
        "trainable": False,
    },
    }
    multi_encoder = MultiEncoder(encoder_configs=encoder_configs)
    x = torch.randn(1, 3, 518, 518)
    outputs = multi_encoder(x)
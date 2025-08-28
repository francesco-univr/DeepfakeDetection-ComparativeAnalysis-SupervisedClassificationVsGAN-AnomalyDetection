import torch.nn as nn
import torchvision.models as tv

# CBAM
try:
    from mycode.cbam import CBAM 
except ImportError:
    try:
        from cbam import CBAM 
    except ImportError:
        CBAM = None

# Helper function for torchvision version compatibility
def _resolve_resnet_weights_enum(arch: str):
   
    
    if arch.startswith("resnet"):
        num = arch.replace("resnet", "")
        enum_name = f"ResNet{num}_Weights"
        if hasattr(tv, enum_name):
            enum_cls = getattr(tv, enum_name)
            if hasattr(enum_cls, "IMAGENET1K_V1"):
                return enum_cls.IMAGENET1K_V1
    # Universal fallback 
    return "IMAGENET1K_V1"


def get_model(
    use_cbam: bool = False,
    arch: str = "resnet18",
    pretrained: bool = False,
    num_classes: int = 2,
    drop_p: float = 0.5,
):
   
    # Weights
    weights = None
    if pretrained:
        try:
            weights = _resolve_resnet_weights_enum(arch)
        except Exception:
            weights = "IMAGENET1K_V1" 

    # Instantiate the backbone model
    if not hasattr(tv, arch):
        raise ValueError(f"Unrecognized architecture for torchvision.models: '{arch}'")
    backbone_ctor = getattr(tv, arch)
    model = backbone_ctor(weights=weights)

    # Add a CBAM attention module
    if use_cbam:
        if CBAM is None:
            raise ImportError("CBAM not found")
        # prendi l'ultimo BasicBlock di layer4 per ricavare i canali
        last_block = list(model.layer4.children())[-1]
        if hasattr(last_block, "conv2") and hasattr(last_block.conv2, "out_channels"):
            ch = last_block.conv2.out_channels
        else:
            # fallback conservativo
            ch = getattr(model.fc, "in_features", 512)
        model.layer4.add_module("cbam", CBAM(ch))

    # Replace the classification head
    if not hasattr(model, "fc"):
        raise RuntimeError("Model needs fc")
    in_feats = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=drop_p),
        nn.Linear(in_feats, num_classes),
    )
    
    return model

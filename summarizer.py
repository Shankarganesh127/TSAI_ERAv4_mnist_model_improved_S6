
# summarizer.py
import torch.nn as nn
from torchsummary import summary

# -----------------------------
# 8. Model Architecture Checks
# -----------------------------
def model_checks(model):
    print('--- Model Architecture Checks ---')
    # Total Parameter Count
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total Parameter Count in Model: {total_params}\n')
    print('Layer-wise Parameter Details (in model order):')
    print('-'*80)
    # Get layers in order as defined in model
    for name, module in model.named_children():
        if isinstance(module, nn.Sequential):
            print(f"\nBlock: {name} (nn.Sequential)")
            for subname, submodule in module.named_children():
                layer_name = f'{name}.{subname} ({submodule.__class__.__name__})'
                layer_params = sum(p.numel() for p in submodule.parameters())
                details = ''
                if isinstance(submodule, nn.Conv2d):
                    details = f'Convolution: {submodule.in_channels} input channels, {submodule.out_channels} output channels, kernel size {submodule.kernel_size}, bias {submodule.bias is not None}'
                elif isinstance(submodule, nn.BatchNorm2d):
                    details = f'BatchNorm: {submodule.num_features} features, affine {submodule.affine}'
                elif isinstance(submodule, nn.ReLU):
                    details = 'Activation: ReLU (no parameters)'
                elif isinstance(submodule, nn.MaxPool2d):
                    details = f'MaxPooling: kernel size {submodule.kernel_size}, stride {submodule.stride}'
                elif isinstance(submodule, nn.Dropout):
                    details = f'Dropout: probability {submodule.p}'
                print(f'  {layer_name:40} | Params: {layer_params:6d} | {details}')
        else:
            layer_name = f'{name} ({module.__class__.__name__})'
            layer_params = sum(p.numel() for p in module.parameters())
            details = ''
            if isinstance(module, nn.AdaptiveAvgPool2d):
                details = f'Global Average Pooling: output size {module.output_size} (no parameters)'
            elif isinstance(module, nn.Linear):
                details = f'Fully Connected: {module.in_features} input features, {module.out_features} output features, bias {module.bias is not None}'
            print(f'  {layer_name:40} | Params: {layer_params:6d} | {details}')
    print('-'*80)
    print('\nSummary:')
    # BatchNorm
    bn_layers = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
    print(f'BatchNorm2d layers used: {len(bn_layers)}')
    # Dropout
    dropout_layers = [m for m in model.modules() if isinstance(m, nn.Dropout)]
    print(f'Dropout layers used: {len(dropout_layers)}')
    # Fully Connected & GAP
    fc_layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
    gap_layers = [m for m in model.modules() if isinstance(m, nn.AdaptiveAvgPool2d)]
    print(f'Fully Connected (Linear) layers used: {len(fc_layers)}')
    print(f'Global Average Pooling layers used: {len(gap_layers)}')
    print('---------------------------------')



import torch
import torch.nn as nn
from model import IMFModel, DenseFeatureEncoder, LatentTokenEncoder, LatentTokenDecoder, ImplicitMotionAlignment, FrameDecoder, ResNetFeatureExtractor

def test_gradient_flow(model, input_shape):
    print(f"\nTesting gradient flow for {type(model).__name__}")
    x = torch.randn(input_shape)
    x.requires_grad_(True)
    
    if isinstance(model, IMFModel):
        x_current = x
        x_reference = torch.randn_like(x)
        output = model(x_current, x_reference)
    elif isinstance(model, (DenseFeatureEncoder, LatentTokenEncoder, ResNetFeatureExtractor)):
        output = model(x)
    elif isinstance(model, LatentTokenDecoder):
        latent = torch.randn(input_shape[0], model.const.shape[1])  # Adjust size as needed
        output = model(latent)
    elif isinstance(model, ImplicitMotionAlignment):
        m_c = torch.randn_like(x)
        m_r = torch.randn_like(x)
        f_r = torch.randn_like(x)
        output = model(m_c, m_r, f_r)
    elif isinstance(model, FrameDecoder):
        # Assuming FrameDecoder expects a list of tensors
        features = [torch.randn_like(x) for _ in range(4)]  # Adjust number as needed
        output = model(features)
    else:
        raise ValueError(f"Unsupported model type: {type(model).__name__}")
    
    if isinstance(output, tuple):
        loss = sum(o.sum() for o in output if isinstance(o, torch.Tensor))
    elif isinstance(output, list):
        loss = sum(o.sum() for o in output)
    else:
        loss = output.sum()
    
    loss.backward()
    
    for name, param in model.named_parameters():
        if param.grad is None:
            print(f"Warning: No gradient for {name}")
        else:
            grad_norm = param.grad.norm().item()
            print(f"{name}: gradient norm = {grad_norm:.6f}")
            assert grad_norm != 0, f"Zero gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"
            assert not torch.isinf(param.grad).any(), f"Inf gradient for {name}"
    
    print(f"Gradient flow test passed for {type(model).__name__}")

def test_feature_extractor(feature_extractor, input_shape):
    print(f"\nTesting {type(feature_extractor).__name__}")
    x = torch.randn(input_shape)
    features = feature_extractor(x)
    
    assert isinstance(features, list), f"Expected list output, got {type(features)}"
    print(f"Number of feature maps: {len(features)}")
    for i, feature in enumerate(features):
        print(f"Feature map {i} shape: {feature.shape}")
    
    # Test gradient flow
    loss = sum(feature.sum() for feature in features)
    loss.backward()
    
    for name, param in feature_extractor.named_parameters():
        if param.grad is None:
            print(f"Warning: No gradient for {name}")
        else:
            grad_norm = param.grad.norm().item()
            print(f"{name}: gradient norm = {grad_norm:.6f}")
            assert grad_norm != 0, f"Zero gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"
            assert not torch.isinf(param.grad).any(), f"Inf gradient for {name}"
    
    print(f"Gradient flow test passed for {type(feature_extractor).__name__}")

def test_implicit_motion_alignment_modules(use_mlgffn=False):
    print("\nTesting ImplicitMotionAlignment modules")
    motion_dims = [128, 256, 512, 512]
    batch_size = 1
    spatial_sizes = [64, 32, 16, 8]  # Example spatial sizes for different levels
    
    for i, dim in enumerate(motion_dims):
        model = ImplicitMotionAlignment(
            feature_dim=dim,
            motion_dim=dim,
            depth=4,
            num_heads=8,
            window_size=8,
            mlp_ratio=4,
            use_mlgffn=use_mlgffn
        )
        
        spatial_size = spatial_sizes[i]
        input_shape = (batch_size, dim, spatial_size, spatial_size)
        
        m_c = torch.randn(input_shape)
        m_r = torch.randn(input_shape)
        f_r = torch.randn(input_shape)
        
        output = model(m_c, m_r, f_r)
        loss = output.sum()
        loss.backward()
        
        print(f"Testing ImplicitMotionAlignment for dim={dim}, spatial_size={spatial_size}")
        for name, param in model.named_parameters():
            if param.grad is None:
                print(f"Warning: No gradient for {name}")
            else:
                grad_norm = param.grad.norm().item()
                print(f"{name}: gradient norm = {grad_norm:.6f}")
                assert grad_norm != 0, f"Zero gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"
                assert not torch.isinf(param.grad).any(), f"Inf gradient for {name}"
        
        print(f"Gradient flow test passed for ImplicitMotionAlignment with dim={dim}")

def run_all_gradient_flow_tests():
    # Test IMFModel
    model = IMFModel()
    test_gradient_flow(model, (1, 3, 256, 256))
    
    # Test DenseFeatureEncoder
    dense_feature_encoder = DenseFeatureEncoder(output_channels=[128, 256, 512, 512])
    test_feature_extractor(dense_feature_encoder, (1, 3, 256, 256))
    
    # Test ResNetFeatureExtractor
    resnet_feature_extractor = ResNetFeatureExtractor(output_channels=[128, 256, 512, 512])
    test_feature_extractor(resnet_feature_extractor, (1, 3, 256, 256))
    
    # Test LatentTokenEncoder
    latent_token_encoder = LatentTokenEncoder(
        initial_channels=64,
        output_channels=[128, 256, 512, 512, 512, 512],
        dm=32
    )
    test_gradient_flow(latent_token_encoder, (1, 3, 256, 256))
    
    # Test LatentTokenDecoder
    latent_token_decoder = LatentTokenDecoder()
    test_gradient_flow(latent_token_decoder, (1, 32))  # Adjust latent dim as needed
    
    # Test ImplicitMotionAlignment modules
    test_implicit_motion_alignment_modules()
    
    # Test FrameDecoder
    frame_decoder = FrameDecoder()
    test_gradient_flow(frame_decoder, (1, 512, 32, 32))

if __name__ == "__main__":
    run_all_gradient_flow_tests()
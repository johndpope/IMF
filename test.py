import torch
import torch.nn as nn
from model import LatentTokenEncoder,DenseFeatureEncoder
# 
# Test  latent_token_encoder
def test_latent_token_encoder():
    batch_size = 1
    input_channels = 3
    input_size = 256
    dm = 32  # "The latent token dimension is dm = 32 for all experiments except for ablation studies."

    et = LatentTokenEncoder(in_channels=input_channels, dm=dm)
    x = torch.randn(batch_size, input_channels, input_size, input_size)

    output = et(x)

    print(f"\nDebugging ET output shape: {output.shape}")

    # Assert the output is a 2D tensor (latent token)
    assert len(output.shape) == 2, f"ET output should be a 2D tensor (batch_size, dm), but got shape {output.shape}"
    assert output.shape[0] == batch_size, f"First dimension should be batch_size ({batch_size}), but got {output.shape[0]}"

    # Assert that dm is within the expected range
    assert 32 <= output.shape[1] <= 1024, f"dm should be between 32 and 1024, got {output.shape[1]}"

    print("All assertions for Latent Token Encoder (ET) passed!")

# Test the adjusted DenseFeatureEncoder
def test_dense_feature_encoder():
    batch_size = 1
    input_channels = 3
    input_size = 256
    
    ef = DenseFeatureEncoder()
    x = torch.randn(batch_size, input_channels, input_size, input_size)
    
    outputs = ef(x)
    
    print("\nFinal output shapes:")
    for i, output in enumerate(outputs):
        print(f"Output {i} shape: {output.shape}")
    
    # Assert the number of outputs (fr1, fr2, fr3, fr4)
    assert len(outputs) == 4, f"EF should produce 4 outputs, but got {len(outputs)}"
    
    # Assert the dimensions of each output
    expected_shapes = [
        (batch_size, 128, 64, 64),
        (batch_size, 256, 32, 32),
        (batch_size, 512, 16, 16),
        (batch_size, 512, 8, 8)
    ]
    
    for i, (output, expected_shape) in enumerate(zip(outputs, expected_shapes)):
        assert output.shape == expected_shape, f"fr{i+1} should be {expected_shape}, but got {output.shape}"
    
    print("All assertions for Dense Feature Encoder (EF) passed!")


# Run the tests
if __name__ == "__main__":
    test_dense_feature_encoder()
    test_latent_token_encoder()
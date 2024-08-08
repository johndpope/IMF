import torch
import torch.nn as nn
from model import LatentTokenEncoder, LatentTokenDecoder, DenseFeatureEncoder,FrameDecoder
# 
# Test  latent_token_encoder
def test_latent_token_encoder():
    batch_size = 1
    input_channels = 3
    input_size = 256
    dm = 32  # "The latent token dimension is dm = 32 for all experiments except for ablation studies."

    et = LatentTokenEncoder(dm=dm)
    x = torch.randn(batch_size, input_channels, input_size, input_size)

    output = et(x)

    print(f"\nDebugging ET output shape: {output.shape}")

    # Assert the output is a 2D tensor (latent token)
    assert len(output.shape) == 2, f"❌ET output should be a 2D tensor (batch_size, dm), but got shape {output.shape}"
    assert output.shape[0] == batch_size, f"❌First dimension should be batch_size ({batch_size}), but got {output.shape[0]}"

    # Assert that dm is within the expected range
    assert 32 <= output.shape[1] <= 1024, f"❌dm should be between 32 and 1024, got {output.shape[1]}"

    print("✅ All assertions for Latent Token Encoder (ET) passed!")

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
    assert len(outputs) == 4, f"❌ EF should produce 4 outputs, but got {len(outputs)}"
    
    # Assert the dimensions of each output
    expected_shapes = [
        (batch_size, 128, 64, 64),
        (batch_size, 256, 32, 32),
        (batch_size, 512, 16, 16),
        (batch_size, 512, 8, 8)
    ]
    
    for i, (output, expected_shape) in enumerate(zip(outputs, expected_shapes)):
        assert output.shape == expected_shape, f"❌ fr{i+1} should be {expected_shape}, but got {output.shape}"
    
    print("✅ All assertions for Dense Feature Encoder (EF) passed!")


# Test LatentTokenDecoder
def test_latent_token_decoder():
    batch_size = 1
    dm = 32  # The latent token dimension
    style_dim = 128  # Example style vector dimension
    input_size = 256


    # current
    et = LatentTokenEncoder(dm=dm)
    x_current = torch.randn(batch_size, 3, input_size, input_size)
    t_c = et(x_current)
    # print("t_c:",t_c.shape)
    
    # ref
    et = LatentTokenEncoder(dm=dm)
    x_ref = torch.randn(batch_size, 3, input_size, input_size)
    t_r = et(x_ref)
    # print("t_r:",t_r.shape)

    ef = DenseFeatureEncoder()
    features = ef(x_current)
    # print("\nFinal output shapes:")
    for i, f_r in enumerate(features):
        print(f"f_r {i} shape: {f_r}")

    latent_token_decoder = LatentTokenDecoder(latent_dim=dm)
    m_r = latent_token_decoder(t_r)
    m_c = latent_token_decoder(t_c)
    for i, m_r_x in enumerate(m_r):
        print(f"m_r_x {i} shape: {m_r_x}")


    print("✅ All assertions for Latent Token Decoder passed!")


def test_frame_decoder():
    # Assuming input sizes. These may need adjustment based on actual input dimensions.
    f_c4 = torch.randn(1, 512, 8, 8)
    f_c3 = torch.randn(1, 512, 16, 16)
    f_c2 = torch.randn(1, 512, 32, 32)
    f_c1 = torch.randn(1, 256, 64, 64)



    model = FrameDecoder()
    output = model([f_c4, f_c3, f_c2, f_c1])
    # Add assertions to verify output shape
    assert output.shape == (1, 3, 256, 256), f"Expected output shape (1, 3, 256, 256), but got {output.shape}"
    print("✅ All assertions passed. Model output shape is correct.")

# Run the tests
if __name__ == "__main__":
    test_dense_feature_encoder()
    test_latent_token_encoder()
    test_latent_token_decoder()
    test_frame_decoder()
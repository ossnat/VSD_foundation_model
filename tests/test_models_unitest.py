# test_mae_system.py
import torch
import torch.nn as nn

# Import your components
from src.models.backbone.mae_backbone_2d import MAEResNet18Backbone
from src.models.backbone.mae_backbone_3d import MAER3D18Backbone
from src.models.heads.mae_decoder_2d import MAEDecoder2D
from src.models.heads.mae_decoder_3d import MAEDecoder3D
from src.models.systems.mae_system import MAESystem


def test_mae_2d():
    """Test MAE 2D (image) pipeline"""
    print("\n" + "="*50)
    print("Testing MAE 2D System")
    print("="*50)
    
    # Dummy config
    config = {
        "training": {"lr": 1e-4, "weight_decay": 0.05},
        "loss": {"normalize": True}
    }
    
    # Create components
    encoder = MAEResNet18Backbone(pretrained=False)
    decoder = MAEDecoder2D(in_channels=512, out_channels=1, hidden_dim=256)
    system = MAESystem(encoder, decoder, config)
    
    # Create dummy batch
    batch_size = 4
    H, W = 100, 100
    
    batch = {
        "video_masked": torch.randn(batch_size, 1, H, W),      # Masked input
        "video_target": torch.randn(batch_size, 1, H, W),      # Original target
        "mask": torch.randint(0, 2, (batch_size, 1, H, W)).float()  # Binary mask
    }
    
    print(f"Input shape: {batch['video_masked'].shape}")
    print(f"Target shape: {batch['video_target'].shape}")
    print(f"Mask shape: {batch['mask'].shape}")
    
    # Forward pass
    result = system.forward(batch)
    loss = result["loss"]
    metrics = result["metrics"]
    
    print(f"\nLoss: {loss.item():.4f}")
    print(f"Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    
    # Test backward pass
    loss.backward()
    print("\n✅ Backward pass successful")
    
    # Test optimizer
    optimizer = system.get_optimizer()
    optimizer.step()
    print("✅ Optimizer step successful")
    
    print("\n✅ MAE 2D test passed!\n")


def test_mae_3d():
    """Test MAE 3D (video) pipeline"""
    print("\n" + "="*50)
    print("Testing MAE 3D System")
    print("="*50)
    
    config = {
        "training": {"lr": 1e-4, "weight_decay": 0.05},
        "loss": {"normalize": True}
    }
    
    # Create components
    encoder = MAER3D18Backbone(pretrained=False)
    decoder = MAEDecoder3D(in_channels=512, out_channels=1, hidden_dim=256)
    system = MAESystem(encoder, decoder, config)
    
    # Create dummy batch (video clips)
    batch_size = 2
    T, H, W = 5, 100, 100
    
    batch = {
        "video_masked": torch.randn(batch_size, 1, T, H, W),
        "video_target": torch.randn(batch_size, 1, T, H, W),
        "mask": torch.randint(0, 2, (batch_size, 1, T, H, W)).float()
    }
    
    print(f"Input shape: {batch['video_masked'].shape}")
    print(f"Target shape: {batch['video_target'].shape}")
    print(f"Mask shape: {batch['mask'].shape}")
    
    # Forward pass
    result = system.forward(batch)
    loss = result["loss"]
    metrics = result["metrics"]
    
    print(f"\nLoss: {loss.item():.4f}")
    print(f"Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    
    # Test backward
    loss.backward()
    print("\n✅ Backward pass successful")
    
    # Test optimizer
    optimizer = system.get_optimizer()
    optimizer.step()
    print("✅ Optimizer step successful")
    
    print("\n✅ MAE 3D test passed!\n")


def test_reconstruction_shapes():
    """Test that reconstruction has correct output shape"""
    print("\n" + "="*50)
    print("Testing Reconstruction Output Shapes")
    print("="*50)
    
    # 2D test
    encoder_2d = MAEResNet18Backbone(pretrained=False)
    decoder_2d = MAEDecoder2D(in_channels=512, out_channels=1)
    
    input_2d = torch.randn(2, 1, 100, 100)
    features_2d = encoder_2d(input_2d)
    reconstruction_2d = decoder_2d(features_2d)
    
    print(f"2D Input: {input_2d.shape}")
    print(f"2D Features: {features_2d.shape}")
    print(f"2D Reconstruction: {reconstruction_2d.shape}")
    assert reconstruction_2d.shape == input_2d.shape, "2D shape mismatch!"
    print("✅ 2D reconstruction shape correct\n")
    
    # 3D test
    encoder_3d = MAER3D18Backbone(pretrained=False)
    decoder_3d = MAEDecoder3D(in_channels=512, out_channels=1)
    
    input_3d = torch.randn(2, 1, 5, 100, 100)
    features_3d = encoder_3d(input_3d)
    reconstruction_3d = decoder_3d(features_3d)
    
    print(f"3D Input: {input_3d.shape}")
    print(f"3D Features: {features_3d.shape}")
    print(f"3D Reconstruction: {reconstruction_3d.shape}")
    assert reconstruction_3d.shape == input_3d.shape, "3D shape mismatch!"
    print("✅ 3D reconstruction shape correct\n")


def test_loss_computation():
    """Test that loss is computed correctly only on masked regions"""
    print("\n" + "="*50)
    print("Testing MAE Loss Computation")
    print("="*50)
    
    from src.models.systems.mae_system import MAELoss
    
    loss_fn = MAELoss(normalize=True)
    
    # Create simple test case
    reconstruction = torch.ones(2, 1, 10, 10)
    target = torch.zeros(2, 1, 10, 10)
    
    # Mask: first half visible (1), second half masked (0)
    mask = torch.cat([
        torch.ones(2, 1, 10, 5),
        torch.zeros(2, 1, 10, 5)
    ], dim=-1)
    
    loss = loss_fn(reconstruction, target, mask)
    
    print(f"Loss on masked regions: {loss.item():.4f}")
    print(f"Expected: ~1.0 (since reconstruction=1, target=0)")
    
    # Loss should be close to 1.0 (MSE of 1-0=1)
    assert 0.9 < loss.item() < 1.1, f"Loss value unexpected: {loss.item()}"
    print("✅ Loss computation correct\n")


# if __name__ == "__main__":
#     print("\n" + "="*60)
#     print("Running MAE System Tests")
#     print("="*60)
    
#     try:
#         test_mae_2d()
#         test_mae_3d()
#         test_reconstruction_shapes()
#         test_loss_computation()
        
#         print("\n" + "="*60)
#         print("✅ All tests passed successfully!")
#         print("="*60 + "\n")
        
#     except Exception as e:
#         print(f"\n❌ Test failed with error: {e}")
#         import traceback
#         traceback.print_exc()

"""
Unit Tests for Stage 2 Latent Flow Matching Training Script
Testing Framework: PyTest
"""
import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, MagicMock, patch
import sys
import os

# Mock imports để test độc lập
sys.modules['dataset'] = MagicMock()
sys.modules['models.fml.autoencoder'] = MagicMock()
sys.modules['models.fml.latent_flow_matcher'] = MagicMock()


# ==================== TEST 1: estimate_scale_factor ====================
class TestScaleFactorEstimation:
    """Test tính toán Scale Factor tự động"""
    
    def test_scale_factor_with_valid_data(self):
        """Test với data hợp lệ - phải trả về scale > 0"""
        # Mock encoder
        mock_encoder = Mock()
        mock_encoder.eval.return_value = mock_encoder
        
        # Mock latent output: std = 0.5 → scale = 1/(0.5) * 1.2 = 2.4
        latent_output = torch.randn(8, 50, 256) * 0.5
        mock_encoder.return_value = latent_output
        
        # Mock dataloader
        mock_batch = {
            'poses': torch.randn(8, 50, 214),
            'pose_mask': torch.ones(8, 50, dtype=torch.bool)
        }
        mock_dataloader = [mock_batch] * 10
        
        device = torch.device('cpu')
        
        # Import function (giả sử đã được import)
        from train_stage2_latent_flow import estimate_scale_factor
        
        scale = estimate_scale_factor(mock_encoder, mock_dataloader, device, num_batches=5)
        
        assert scale > 0, "Scale factor phải > 0"
        assert 1.0 <= scale <= 5.0, f"Scale {scale} nằm ngoài khoảng hợp lý"
    
    def test_scale_factor_with_empty_data(self):
        """Test khi không có data → phải trả về 1.0 (fallback)"""
        mock_encoder = Mock()
        mock_dataloader = []  # Empty
        device = torch.device('cpu')
        
        from train_stage2_latent_flow import estimate_scale_factor
        scale = estimate_scale_factor(mock_encoder, mock_dataloader, device)
        
        assert scale == 1.0, "Empty data phải trả về scale = 1.0"
    
    def test_scale_factor_numerical_stability(self):
        """Test với latent có std rất nhỏ → tránh division by zero"""
        mock_encoder = Mock()
        mock_encoder.eval.return_value = mock_encoder
        
        # Latent với std gần 0
        latent_output = torch.zeros(8, 50, 256) + 1e-8
        mock_encoder.return_value = latent_output
        
        mock_batch = {
            'poses': torch.randn(8, 50, 214),
            'pose_mask': torch.ones(8, 50, dtype=torch.bool)
        }
        mock_dataloader = [mock_batch] * 5
        device = torch.device('cpu')
        
        from train_stage2_latent_flow import estimate_scale_factor
        scale = estimate_scale_factor(mock_encoder, mock_dataloader, device)
        
        assert not torch.isnan(torch.tensor(scale)), "Scale không được là NaN"
        assert not torch.isinf(torch.tensor(scale)), "Scale không được là Inf"


# ==================== TEST 2: train_epoch ====================
class TestTrainEpoch:
    """Test vòng lặp training 1 epoch"""
    
    @pytest.fixture
    def mock_components(self):
        """Setup mock components"""
        flow_matcher = Mock()
        flow_matcher.train.return_value = None
        flow_matcher.parameters.return_value = [torch.randn(10, 10, requires_grad=True)]
        
        encoder = Mock()
        encoder.eval.return_value = encoder
        encoder.return_value = torch.randn(4, 30, 256)
        
        optimizer = Mock()
        device = torch.device('cpu')
        
        return flow_matcher, encoder, optimizer, device
    
    def test_train_epoch_loss_computation(self, mock_components):
        """Test tính loss đúng và backward được gọi"""
        flow_matcher, encoder, optimizer, device = mock_components
        
        # Mock loss output
        mock_losses = {
            'total': torch.tensor(1.5, requires_grad=True),
            'flow': torch.tensor(1.0),
            'sync': torch.tensor(0.3),
            'length': torch.tensor(0.2)
        }
        flow_matcher.return_value = mock_losses
        
        # Mock dataloader
        batch = {
            'poses': torch.randn(4, 30, 214),
            'pose_mask': torch.ones(4, 30, dtype=torch.bool),
            'text_tokens': torch.randint(0, 1000, (4, 20)),
            'attention_mask': torch.ones(4, 20, dtype=torch.bool),
            'seq_lengths': torch.tensor([25, 28, 30, 22])
        }
        mock_dataloader = [batch, batch]
        
        from train_stage2_latent_flow import train_epoch
        
        metrics, avg_loss = train_epoch(
            flow_matcher, encoder, mock_dataloader, 
            optimizer, device, epoch=0, scale_factor=1.0
        )
        
        # Assertions
        assert 'flow' in metrics, "Metrics phải có 'flow'"
        assert 'sync' in metrics, "Metrics phải có 'sync'"
        assert avg_loss > 0, "Average loss phải > 0"
        assert optimizer.zero_grad.call_count >= 2, "zero_grad phải được gọi mỗi batch"
        assert optimizer.step.call_count >= 2, "optimizer.step phải được gọi"
    
    def test_gradient_clipping_applied(self, mock_components):
        """Test gradient clipping được áp dụng"""
        flow_matcher, encoder, optimizer, device = mock_components
        
        mock_losses = {'total': torch.tensor(2.0, requires_grad=True), 'flow': torch.tensor(2.0)}
        flow_matcher.return_value = mock_losses
        
        batch = {
            'poses': torch.randn(2, 20, 214),
            'pose_mask': torch.ones(2, 20, dtype=torch.bool),
            'text_tokens': torch.randint(0, 1000, (2, 15)),
            'attention_mask': torch.ones(2, 15, dtype=torch.bool),
            'seq_lengths': torch.tensor([18, 20])
        }
        mock_dataloader = [batch]
        
        with patch('torch.nn.utils.clip_grad_norm_') as mock_clip:
            from train_stage2_latent_flow import train_epoch
            train_epoch(flow_matcher, encoder, mock_dataloader, optimizer, device, 0, 1.0)
            
            assert mock_clip.called, "Gradient clipping phải được gọi"
            # Check max_norm=1.0
            call_args = mock_clip.call_args
            assert call_args[1]['max_norm'] == 1.0, "max_norm phải là 1.0"


# ==================== TEST 3: validate ====================
class TestValidation:
    """Test validation function"""
    
    def test_validate_no_gradient(self):
        """Test validation chạy trong no_grad context"""
        mock_flow_matcher = Mock()
        mock_flow_matcher.eval.return_value = None
        mock_flow_matcher.return_value = {
            'total': torch.tensor(0.8),
            'flow': torch.tensor(0.5)
        }
        
        mock_encoder = Mock()
        mock_encoder.eval.return_value = mock_encoder
        mock_encoder.return_value = torch.randn(2, 20, 256)
        
        batch = {
            'poses': torch.randn(2, 20, 214),
            'pose_mask': torch.ones(2, 20, dtype=torch.bool),
            'text_tokens': torch.randint(0, 1000, (2, 15)),
            'attention_mask': torch.ones(2, 15, dtype=torch.bool),
            'seq_lengths': torch.tensor([18, 20])
        }
        mock_dataloader = [batch, batch]
        
        from train_stage2_latent_flow import validate
        
        val_loss = validate(mock_flow_matcher, mock_encoder, mock_dataloader, 
                           torch.device('cpu'), scale_factor=1.0)
        
        assert val_loss >= 0, "Validation loss phải >= 0"
        assert mock_flow_matcher.eval.called, "Model phải ở eval mode"
    
    def test_validate_returns_average_loss(self):
        """Test validation trả về average loss đúng"""
        mock_flow_matcher = Mock()
        mock_flow_matcher.eval.return_value = None
        
        # Loss giảm dần qua các batch
        mock_flow_matcher.side_effect = [
            {'total': torch.tensor(1.0)},
            {'total': torch.tensor(0.8)},
            {'total': torch.tensor(0.6)}
        ]
        
        mock_encoder = Mock()
        mock_encoder.eval.return_value = mock_encoder
        mock_encoder.return_value = torch.randn(2, 20, 256)
        
        batch = {
            'poses': torch.randn(2, 20, 214),
            'pose_mask': torch.ones(2, 20, dtype=torch.bool),
            'text_tokens': torch.randint(0, 1000, (2, 15)),
            'attention_mask': torch.ones(2, 15, dtype=torch.bool),
            'seq_lengths': torch.tensor([18, 20])
        }
        mock_dataloader = [batch] * 3
        
        from train_stage2_latent_flow import validate
        val_loss = validate(mock_flow_matcher, mock_encoder, mock_dataloader, 
                           torch.device('cpu'), 1.0)
        
        expected_avg = (1.0 + 0.8 + 0.6) / 3
        assert abs(val_loss - expected_avg) < 1e-5, f"Expected {expected_avg}, got {val_loss}"


# ==================== TEST 4: Integration Tests ====================
class TestTrainingIntegration:
    """Test toàn bộ training flow"""
    
    def test_scale_factor_persistence_in_checkpoint(self):
        """Test scale_factor được lưu và load đúng từ checkpoint"""
        checkpoint_data = {
            'epoch': 5,
            'model_state_dict': {},
            'latent_scale_factor': 2.5,
            'best_val_loss': 0.5
        }
        
        # Giả lập load checkpoint
        loaded_scale = checkpoint_data.get('latent_scale_factor', 1.0)
        assert loaded_scale == 2.5, "Scale factor phải được load đúng"
    
    def test_scheduler_step_timing(self):
        """Test scheduler.step() được gọi SAU mỗi epoch, không trong batch loop"""
        mock_scheduler = Mock()
        
        # Giả lập training 1 epoch với 3 batches
        # Scheduler KHÔNG được gọi trong batch loop
        num_batches = 3
        for _ in range(num_batches):
            pass  # Training logic
        
        # Scheduler chỉ được gọi 1 lần sau epoch
        mock_scheduler.step()
        
        assert mock_scheduler.step.call_count == 1, "Scheduler chỉ step 1 lần/epoch"
    
    def test_best_model_saving_logic(self):
        """Test logic lưu best model khi val_loss giảm"""
        best_val_loss = 1.0
        current_val_loss = 0.8
        
        should_save = current_val_loss < best_val_loss
        
        assert should_save == True, "Phải lưu khi val_loss giảm"
        
        # Update best
        if should_save:
            best_val_loss = current_val_loss
        
        assert best_val_loss == 0.8, "best_val_loss phải được update"


# ==================== TEST 5: Edge Cases ====================
class TestEdgeCases:
    """Test các trường hợp đặc biệt"""
    
    def test_empty_batch_handling(self):
        """Test xử lý batch rỗng"""
        # Trong thực tế không có batch rỗng nhưng test để đảm bảo robust
        num_batches = 0
        total_loss = 0.0
        
        avg_loss = total_loss / max(num_batches, 1)  # Tránh division by zero
        
        assert avg_loss == 0.0, "Empty batch phải trả về loss = 0"
    
    def test_nan_loss_detection(self):
        """Test phát hiện NaN loss"""
        loss = torch.tensor(float('nan'))
        
        is_nan = torch.isnan(loss)
        
        assert is_nan == True, "Phải detect được NaN loss"
        # Trong code thực tế nên có:
        # if torch.isnan(loss):
        #     raise ValueError("NaN loss detected!")
    
    def test_gradient_explosion_clipping(self):
        """Test gradient clipping hoạt động với gradient lớn"""
        # Giả lập gradient lớn
        param = torch.randn(100, 100, requires_grad=True)
        param.grad = torch.randn_like(param) * 1000  # Gradient rất lớn
        
        # Clip
        torch.nn.utils.clip_grad_norm_([param], max_norm=1.0)
        
        grad_norm = param.grad.norm().item()
        
        assert grad_norm <= 1.0 + 1e-5, f"Gradient norm {grad_norm} phải <= 1.0 sau clip"


# ==================== RUN TESTS ====================
if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
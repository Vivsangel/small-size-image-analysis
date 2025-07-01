# Modern Machine Learning Ecosystem: Advanced Insights for Small Data Industrial Applications

## 1. Foundation Models Revolution in Computer Vision

### Vision Transformers (ViTs) and Their Variants
The paradigm has shifted dramatically from CNNs to transformer-based architectures:

```python
# Modern ViT with advanced techniques
import timm
from transformers import AutoImageProcessor, AutoModel

# CLIP-based approaches for zero-shot classification
import clip
import torch

class ModernVisionFoundationModel:
    def __init__(self, model_name="microsoft/swin-large-patch4-window7-224"):
        # Swin Transformers show superior performance on small datasets
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
        
        # DINOv2 for self-supervised features (excellent for industrial data)
        self.dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        
        # CLIP for semantic understanding
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32")
    
    def extract_multimodal_features(self, image, text_context=None):
        """Extract features using multiple foundation models"""
        # Standard supervised features
        supervised_features = self.backbone(image)
        
        # Self-supervised features (often better for anomaly detection)
        with torch.no_grad():
            dino_features = self.dino_model(image)
        
        # CLIP features for semantic understanding
        if text_context:
            text_features = self.clip_model.encode_text(clip.tokenize(text_context))
            image_features = self.clip_model.encode_image(image)
            clip_similarity = torch.cosine_similarity(image_features, text_features)
        
        return {
            'supervised': supervised_features,
            'self_supervised': dino_features,
            'clip_features': image_features if text_context else None,
            'semantic_similarity': clip_similarity if text_context else None
        }
```

### Key Insights:
- **DINOv2** shows remarkable performance on industrial images without fine-tuning
- **Swin Transformers** often outperform CNNs on small datasets due to better inductive biases
- **CLIP** enables zero-shot classification using natural language descriptions of defects

## 2. Meta-Learning and Few-Shot Learning Advances

### Model-Agnostic Meta-Learning (MAML) 2.0
```python
import torch
import torch.nn as nn
from collections import OrderedDict

class MAML_Plus(nn.Module):
    """Enhanced MAML with modern improvements"""
    
    def __init__(self, model, inner_lr=0.01, meta_lr=0.001):
        super().__init__()
        self.model = model
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        
        # Learnable inner learning rates (MAML++)
        self.inner_lrs = nn.ParameterDict({
            name: nn.Parameter(torch.tensor(inner_lr))
            for name, _ in model.named_parameters()
        })
    
    def inner_update(self, loss, params=None):
        """Perform inner loop update with learnable learning rates"""
        if params is None:
            params = OrderedDict(self.model.named_parameters())
        
        grads = torch.autograd.grad(loss, params.values(), create_graph=True)
        
        updated_params = OrderedDict()
        for (name, param), grad in zip(params.items(), grads):
            # Use learnable learning rate
            lr = self.inner_lrs[name]
            updated_params[name] = param - lr * grad
        
        return updated_params
    
    def forward(self, support_x, support_y, query_x, n_inner_steps=5):
        """Meta-learning forward pass"""
        # Clone model parameters
        fast_weights = OrderedDict(self.model.named_parameters())
        
        # Inner loop adaptation
        for _ in range(n_inner_steps):
            # Forward pass with current weights
            logits = self.functional_forward(support_x, fast_weights)
            loss = F.cross_entropy(logits, support_y)
            
            # Update weights
            fast_weights = self.inner_update(loss, fast_weights)
        
        # Query set prediction with adapted weights
        query_logits = self.functional_forward(query_x, fast_weights)
        return query_logits
    
    def functional_forward(self, x, weights):
        """Forward pass using functional weights"""
        # Implement functional forward pass here
        pass

# Prototypical Networks with attention
class PrototypicalNetworkWithAttention(nn.Module):
    def __init__(self, backbone, feature_dim=512):
        super().__init__()
        self.backbone = backbone
        self.attention = nn.MultiheadAttention(feature_dim, num_heads=8)
        
    def compute_prototypes(self, support_features, support_labels, n_classes):
        """Compute class prototypes with attention"""
        prototypes = []
        
        for c in range(n_classes):
            class_features = support_features[support_labels == c]
            
            # Self-attention over class samples
            attended_features, _ = self.attention(
                class_features, class_features, class_features
            )
            
            # Prototype is attention-weighted mean
            prototype = attended_features.mean(dim=0)
            prototypes.append(prototype)
        
        return torch.stack(prototypes)
```

## 3. Advanced Data Augmentation Strategies

### Modern Augmentation Techniques
```python
import torchvision.transforms as T
from torchvision.transforms import autoaugment, InterpolationMode
import albumentations as A

class AdvancedAugmentationPipeline:
    """State-of-the-art augmentation strategies"""
    
    @staticmethod
    def get_autoaugment_policy():
        """AutoAugment optimized for industrial images"""
        return autoaugment.AutoAugmentPolicy.IMAGENET
    
    @staticmethod
    def get_randaugment(n=2, m=9):
        """RandAugment - simple but effective"""
        return autoaugment.RandAugment(
            num_ops=n, magnitude=m, 
            interpolation=InterpolationMode.BILINEAR
        )
    
    @staticmethod
    def get_trivialaugment():
        """TrivialAugment - one random augmentation per image"""
        return autoaugment.TrivialAugmentWide(
            interpolation=InterpolationMode.BILINEAR
        )
    
    @staticmethod
    def get_mixup_cutmix():
        """Mixup and CutMix for regularization"""
        return T.Compose([
            T.RandomChoice([
                # MixUp implemented in training loop
                T.Lambda(lambda x: x),  # Placeholder
                # CutMix
                A.CoarseDropout(
                    max_holes=1, max_height=0.3, max_width=0.3,
                    min_holes=1, min_height=0.1, min_width=0.1,
                    fill_value=0, mask_fill_value=0, p=0.5
                )
            ])
        ])

# Advanced geometric augmentations for microscopy
class MicroscopyAugmentations:
    @staticmethod
    def get_physics_based_augmentations():
        """Physics-based augmentations for microscopy"""
        return A.Compose([
            # Simulate different illumination conditions
            A.RandomBrightnessContrast(
                brightness_limit=(-0.2, 0.3),
                contrast_limit=(-0.3, 0.4),
                p=0.8
            ),
            
            # Simulate optical aberrations
            A.OneOf([
                A.OpticalDistortion(distort_limit=0.1, shift_limit=0.1),
                A.GridDistortion(num_steps=5, distort_limit=0.1),
                A.PiecewiseAffine(scale=(0.01, 0.05)),
            ], p=0.5),
            
            # Simulate focus variations
            A.OneOf([
                A.MotionBlur(blur_limit=(3, 7)),
                A.GaussianBlur(blur_limit=(1, 3)),
                A.Defocus(radius=(1, 3), alias_blur=(0.1, 0.5)),
            ], p=0.4),
            
            # Simulate sensor noise
            A.OneOf([
                A.GaussNoise(var_limit=(10, 50)),
                A.ISONoise(color_shift=(0.01, 0.1), intensity=(0.1, 0.8)),
                A.MultiplicativeNoise(multiplier=(0.9, 1.1), per_channel=True),
            ], p=0.3),
            
            # Color space variations (important for different staining/imaging)
            A.HueSaturationValue(
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=0.5
            ),
        ])
```

## 4. Self-Supervised Learning Revolution

### Modern Self-Supervised Approaches
```python
import torch
import torch.nn as nn
from lightly.models.modules import SimCLRProjectionHead
from lightly.loss import NTXentLoss

class ModernSSLFramework:
    """Comprehensive self-supervised learning framework"""
    
    def __init__(self, backbone):
        self.backbone = backbone
        
        # Multiple SSL heads for different pretext tasks
        self.simclr_head = SimCLRProjectionHead(512, 512, 128)
        self.rotation_head = nn.Linear(512, 4)  # 0, 90, 180, 270 degrees
        self.jigsaw_head = nn.Linear(512, 100)  # Jigsaw puzzle permutations
        
        # Losses
        self.contrastive_loss = NTXentLoss()
        self.rotation_loss = nn.CrossEntropyLoss()
        self.jigsaw_loss = nn.CrossEntropyLoss()
    
    def forward_simclr(self, x1, x2):
        """SimCLR contrastive learning"""
        # Extract features
        f1 = self.backbone(x1)
        f2 = self.backbone(x2)
        
        # Project to contrastive space
        z1 = self.simclr_head(f1)
        z2 = self.simclr_head(f2)
        
        # Contrastive loss
        loss = self.contrastive_loss(z1, z2)
        return loss
    
    def forward_rotation(self, x, rotation_labels):
        """Rotation prediction pretext task"""
        features = self.backbone(x)
        logits = self.rotation_head(features)
        loss = self.rotation_loss(logits, rotation_labels)
        return loss
    
    def forward_jigsaw(self, x_patches, permutation_labels):
        """Jigsaw puzzle pretext task"""
        # Process patches and predict permutation
        patch_features = []
        for patch in x_patches:
            patch_features.append(self.backbone(patch))
        
        combined_features = torch.cat(patch_features, dim=1)
        logits = self.jigsaw_head(combined_features)
        loss = self.jigsaw_loss(logits, permutation_labels)
        return loss

# BYOL (Bootstrap Your Own Latent) implementation
class BYOL(nn.Module):
    """Bootstrap Your Own Latent - works without negative samples"""
    
    def __init__(self, backbone, projection_dim=256, hidden_dim=4096):
        super().__init__()
        
        # Online network
        self.online_encoder = backbone
        self.online_projector = nn.Sequential(
            nn.Linear(backbone.num_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim)
        )
        self.online_predictor = nn.Sequential(
            nn.Linear(projection_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim)
        )
        
        # Target network (EMA of online network)
        self.target_encoder = self._copy_network(self.online_encoder)
        self.target_projector = self._copy_network(self.online_projector)
        
        # Disable gradients for target network
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        for param in self.target_projector.parameters():
            param.requires_grad = False
    
    def _copy_network(self, network):
        """Create a copy of the network"""
        import copy
        return copy.deepcopy(network)
    
    def _update_target_network(self, tau=0.99):
        """Exponential moving average update of target network"""
        for online_param, target_param in zip(
            self.online_encoder.parameters(), 
            self.target_encoder.parameters()
        ):
            target_param.data = tau * target_param.data + (1 - tau) * online_param.data
        
        for online_param, target_param in zip(
            self.online_projector.parameters(), 
            self.target_projector.parameters()
        ):
            target_param.data = tau * target_param.data + (1 - tau) * online_param.data
    
    def forward(self, x1, x2):
        """BYOL forward pass"""
        # Online network predictions
        online_proj1 = self.online_projector(self.online_encoder(x1))
        online_pred1 = self.online_predictor(online_proj1)
        
        online_proj2 = self.online_projector(self.online_encoder(x2))
        online_pred2 = self.online_predictor(online_proj2)
        
        # Target network projections
        with torch.no_grad():
            target_proj1 = self.target_projector(self.target_encoder(x1))
            target_proj2 = self.target_projector(self.target_encoder(x2))
        
        # Symmetric loss
        loss1 = self._compute_loss(online_pred1, target_proj2.detach())
        loss2 = self._compute_loss(online_pred2, target_proj1.detach())
        
        return (loss1 + loss2) / 2
    
    def _compute_loss(self, pred, target):
        """Normalized mean squared error"""
        pred_norm = nn.functional.normalize(pred, dim=1)
        target_norm = nn.functional.normalize(target, dim=1)
        return 2 - 2 * (pred_norm * target_norm).sum(dim=1).mean()
```

## 5. Advanced Neural Architecture Search (NAS)

### Efficient NAS for Small Datasets
```python
import torch
import torch.nn as nn
from functools import partial

class DifferentiableNAS:
    """Differentiable NAS for finding optimal architectures"""
    
    def __init__(self, search_space, num_classes=2):
        self.search_space = search_space
        self.num_classes = num_classes
        
        # Architecture parameters (learnable)
        self.arch_params = nn.ParameterList([
            nn.Parameter(torch.randn(len(ops)) / len(ops))
            for ops in search_space
        ])
    
    def create_mixed_operation(self, ops, arch_param):
        """Create mixed operation based on architecture parameters"""
        def mixed_op(x):
            # Softmax to get operation weights
            weights = torch.softmax(arch_param, dim=0)
            
            # Weighted combination of operations
            output = sum(w * op(x) for w, op in zip(weights, ops))
            return output
        
        return mixed_op
    
    def sample_architecture(self):
        """Sample discrete architecture from continuous space"""
        sampled_arch = []
        for arch_param in self.arch_params:
            # Gumbel softmax for discrete sampling
            logits = arch_param
            sampled_op_idx = torch.multinomial(
                torch.softmax(logits, dim=0), 1
            ).item()
            sampled_arch.append(sampled_op_idx)
        
        return sampled_arch

# Progressive shrinking for efficient architecture search
class ProgressiveShrinkingNAS:
    """Progressive shrinking to find efficient architectures"""
    
    def __init__(self, supernet, target_flops=None, target_params=None):
        self.supernet = supernet
        self.target_flops = target_flops
        self.target_params = target_params
    
    def progressive_shrinking_step(self, dataloader, criterion, optimizer):
        """One step of progressive shrinking"""
        # Sample sub-networks with different sizes
        subnets = self._sample_subnets()
        
        total_loss = 0
        for subnet_config in subnets:
            # Configure supernet to subnet
            self._configure_subnet(subnet_config)
            
            # Forward pass
            for batch in dataloader:
                inputs, targets = batch
                outputs = self.supernet(inputs)
                loss = criterion(outputs, targets)
                
                # Add efficiency constraints
                if self.target_flops:
                    flops_penalty = self._compute_flops_penalty(subnet_config)
                    loss += flops_penalty
                
                total_loss += loss
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        return total_loss.item()
```

## 6. Modern MLOps and Production Systems

### Advanced MLOps Pipeline
```python
from typing import Dict, Any, List
import mlflow
import wandb
from dataclasses import dataclass
import yaml

@dataclass
class ModelConfig:
    """Model configuration with validation"""
    model_name: str
    learning_rate: float
    batch_size: int
    epochs: int
    augmentation_policy: str
    
    def __post_init__(self):
        # Validation logic
        assert self.learning_rate > 0, "Learning rate must be positive"
        assert self.batch_size > 0, "Batch size must be positive"

class MLOpsFramework:
    """Comprehensive MLOps framework"""
    
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.setup_experiment_tracking()
        self.setup_model_registry()
    
    def _load_config(self, config_path: str) -> ModelConfig:
        """Load and validate configuration"""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return ModelConfig(**config_dict)
    
    def setup_experiment_tracking(self):
        """Setup experiment tracking with multiple backends"""
        # MLflow for model lifecycle management
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("semiconductor_analysis")
        
        # Weights & Biases for advanced visualization
        wandb.init(
            project="semiconductor-ml",
            config=self.config.__dict__,
            tags=["small-data", "semiconductor", "industrial-ai"]
        )
    
    def log_metrics(self, metrics: Dict[str, float], step: int):
        """Log metrics to multiple tracking systems"""
        # MLflow
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=step)
        
        # Weights & Biases
        wandb.log(metrics, step=step)
    
    def save_model_checkpoint(self, model, optimizer, scheduler, epoch, metrics):
        """Save comprehensive model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config.__dict__
        }
        
        # Save to MLflow
        mlflow.pytorch.log_model(
            model, 
            f"model_epoch_{epoch}",
            registered_model_name="SemiconductorClassifier"
        )
        
        # Save artifacts
        torch.save(checkpoint, f"checkpoint_epoch_{epoch}.pth")
        mlflow.log_artifact(f"checkpoint_epoch_{epoch}.pth")

# Advanced model monitoring
class ModelMonitor:
    """Production model monitoring system"""
    
    def __init__(self, model, reference_dataset):
        self.model = model
        self.reference_features = self._extract_reference_features(reference_dataset)
        
    def _extract_reference_features(self, dataset):
        """Extract features from reference dataset"""
        features = []
        self.model.eval()
        with torch.no_grad():
            for batch in dataset:
                inputs, _ = batch
                feature_maps = self.model.extract_features(inputs)
                features.append(feature_maps.cpu())
        
        return torch.cat(features, dim=0)
    
    def detect_drift(self, new_batch, threshold=0.1):
        """Detect data drift using feature statistics"""
        # Extract features from new batch
        with torch.no_grad():
            new_features = self.model.extract_features(new_batch)
        
        # Compute distribution statistics
        ref_mean = self.reference_features.mean(dim=0)
        ref_std = self.reference_features.std(dim=0)
        
        new_mean = new_features.mean(dim=0)
        new_std = new_features.std(dim=0)
        
        # KL divergence approximation
        kl_div = torch.sum(
            torch.log(new_std / ref_std) + 
            (ref_std**2 + (ref_mean - new_mean)**2) / (2 * new_std**2) - 0.5
        )
        
        drift_detected = kl_div > threshold
        
        return {
            'drift_detected': drift_detected.item(),
            'kl_divergence': kl_div.item(),
            'drift_score': min(kl_div.item() / threshold, 1.0)
        }
```

## 7. Edge Deployment and Optimization

### Model Optimization for Edge Deployment
```python
import torch
import torch.quantization as quantization
from torch.ao.quantization import get_default_qconfig_mapping
import onnx
import onnxruntime as ort

class EdgeOptimizer:
    """Comprehensive edge optimization pipeline"""
    
    def __init__(self, model):
        self.model = model
    
    def quantize_model(self, calibration_loader, backend='qnnpack'):
        """Post-training quantization"""
        # Set backend
        torch.backends.quantized.engine = backend
        
        # Prepare model for quantization
        self.model.eval()
        
        # Get default quantization configuration
        qconfig_mapping = get_default_qconfig_mapping(backend)
        
        # Prepare for post-training quantization
        model_prepared = quantization.prepare(self.model, qconfig_mapping)
        
        # Calibration
        model_prepared.eval()
        with torch.no_grad():
            for inputs, _ in calibration_loader:
                model_prepared(inputs)
        
        # Convert to quantized model
        quantized_model = quantization.convert(model_prepared)
        
        return quantized_model
    
    def prune_model(self, pruning_ratio=0.3):
        """Structured and unstructured pruning"""
        import torch.nn.utils.prune as prune
        
        # Define pruning strategy
        parameters_to_prune = []
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                parameters_to_prune.append((module, 'weight'))
        
        # Apply global magnitude pruning
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=pruning_ratio,
        )
        
        # Make pruning permanent
        for module, param_name in parameters_to_prune:
            prune.remove(module, param_name)
        
        return self.model
    
    def knowledge_distillation(self, teacher_model, student_model, 
                             train_loader, temperature=4.0, alpha=0.7):
        """Knowledge distillation for model compression"""
        
        teacher_model.eval()
        student_model.train()
        
        criterion_ce = nn.CrossEntropyLoss()
        criterion_kd = nn.KLDivLoss(reduction='batchmean')
        
        optimizer = torch.optim.Adam(student_model.parameters())
        
        for epoch in range(50):  # Fewer epochs for distillation
            for inputs, targets in train_loader:
                # Teacher predictions
                with torch.no_grad():
                    teacher_outputs = teacher_model(inputs)
                    teacher_soft = F.softmax(teacher_outputs / temperature, dim=1)
                
                # Student predictions
                student_outputs = student_model(inputs)
                student_soft = F.log_softmax(student_outputs / temperature, dim=1)
                
                # Combined loss
                hard_loss = criterion_ce(student_outputs, targets)
                soft_loss = criterion_kd(student_soft, teacher_soft) * (temperature ** 2)
                
                total_loss = alpha * soft_loss + (1 - alpha) * hard_loss
                
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
        
        return student_model
    
    def export_to_onnx(self, model, dummy_input, output_path):
        """Export model to ONNX format"""
        model.eval()
        
        # Export with dynamic axes for batch size
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            },
            opset_version=14
        )
        
        # Optimize ONNX model
        import onnxoptimizer
        
        onnx_model = onnx.load(output_path)
        optimized_model = onnxoptimizer.optimize(onnx_model)
        onnx.save(optimized_model, output_path.replace('.onnx', '_optimized.onnx'))
        
        return output_path.replace('.onnx', '_optimized.onnx')
    
    def benchmark_inference(self, model_path, test_loader):
        """Benchmark inference performance"""
        import time
        
        # Load ONNX Runtime session
        session = ort.InferenceSession(model_path)
        
        inference_times = []
        
        for inputs, _ in test_loader:
            start_time = time.time()
            
            # Run inference
            ort_inputs = {session.get_inputs()[0].name: inputs.numpy()}
            outputs = session.run(None, ort_inputs)
            
            end_time = time.time()
            inference_times.append(end_time - start_time)
        
        return {
            'mean_inference_time': np.mean(inference_times),
            'std_inference_time': np.std(inference_times),
            'throughput_fps': len(test_loader) / sum(inference_times)
        }
```

## 8. Interpretability and Explainable AI

### Advanced Interpretability Methods
```python
import torch
import torch.nn.functional as F
from captum.attr import (
    IntegratedGradients, GradCAM, GuidedGradCam, 
    LayerGradCam, Occlusion, Lime, ShapleyValueSampling
)
from captum.attr import visualization as viz

class ExplainabilityFramework:
    """Comprehensive model interpretability framework"""
    
    def __init__(self, model, target_layer=None):
        self.model = model
        self.target_layer = target_layer or self._get_target_layer()
        
        # Initialize attribution methods
        self.integrated_gradients = IntegratedGradients(model)
        self.gradcam = GradCAM(model, self.target_layer)
        self.guided_gradcam = GuidedGradCam(model, self.target_layer)
        self.occlusion = Occlusion(model)
        self.shapley = ShapleyValueSampling(model)
    
    def _get_target_layer(self):
        """Automatically determine target layer for CAM methods"""
        for name, module in reversed(list(self.model.named_modules())):
            if isinstance(module, nn.Conv2d):
                return module
        return None
    
    def explain_prediction(self, input_tensor, target_class=None, methods='all'):
        """Generate comprehensive explanations"""
        
        if target_class is None:
            # Use predicted class
            with torch.no_grad():
                output = self.model(input_tensor)
                target_class = output.argmax(dim=1).item()
        
        explanations = {}
        
        if methods in ['all', 'gradients']:
            # Integrated Gradients
            ig_attrs = self.integrated_gradients.attribute(
                input_tensor, target=target_class, n_steps=50
            )
            explanations['integrated_gradients'] = ig_attrs
        
        if methods in ['all', 'cam']:
            # GradCAM
            gradcam_attrs = self.gradcam.attribute(
                input_tensor, target=target_class
            )
            explanations['gradcam'] = gradcam_attrs
            
            # Guided GradCAM
            guided_attrs = self.guided_gradcam.attribute(
                input_tensor, target=target_class
            )
            explanations['guided_gradcam'] = guided_attrs
        
        if methods in ['all', 'occlusion']:
            # Occlusion-based explanations
            occlusion_attrs = self.occlusion.attribute(
                input_tensor, 
                target=target_class,
                strides=(3, 8, 8),
                sliding_window_shapes=(3, 15, 15)
            )
            explanations['occlusion'] = occlusion_attrs
        
        if methods in ['all', 'shapley']:
            # Shapley Value Sampling
            shapley_attrs = self.shapley.attribute(
                input_tensor,
                target=target_class,
                n_samples=100
            )
            explanations['shapley'] = shapley_attrs
        
        return explanations
    
    def visualize_explanations(self, input_tensor, explanations, 
                             save_path=None, figsize=(15, 10)):
        """Create comprehensive visualization"""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        
        # Original image
        original_img = input_tensor.squeeze().permute(1, 2, 0).detach().cpu()
        axes[0, 0].imshow(original_img)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Integrated Gradients
        if 'integrated_gradients' in explanations:
            ig_viz = viz.visualize_image_attr(
                explanations['integrated_gradients'].squeeze().permute(1, 2, 0).detach().cpu(),
                original_img,
                method='blended_heat_map',
                show_colorbar=True,
                sign='positive',
                plt_fig_axis=(fig, axes[0, 1])
            )
            axes[0, 1].set_title('Integrated Gradients')
        
        # GradCAM
        if 'gradcam' in explanations:
            gradcam_viz = viz.visualize_image_attr(
                explanations['gradcam'].squeeze().detach().cpu(),
                original_img,
                method='blended_heat_map',
                show_colorbar=True,
                plt_fig_axis=(fig, axes[0, 2])
            )
            axes[0, 2].set_title('GradCAM')
        
        # Additional visualizations...
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return fig

# Counterfactual explanations
class CounterfactualExplainer:
    """Generate counterfactual explanations"""
    
    def __init__(self, model, feature_ranges=None):
        self.model = model
        self.feature_ranges = feature_ranges
    
    def generate_counterfactual(self, input_tensor, target_class, 
                              max_iterations=1000, lr=0.01, lambda_reg=0.01):
        """Generate counterfactual explanation using optimization"""
        
        # Initialize counterfactual as copy of input
        counterfactual = input_tensor.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([counterfactual], lr=lr)
        
        for iteration in range(max_iterations):
            optimizer.zero_grad()
            
            # Forward pass
            output = self.model(counterfactual)
            prediction = F.softmax(output, dim=1)
            
            # Loss: maximize target class probability + minimize distance
            target_loss = -torch.log(prediction[0, target_class] + 1e-8)
            distance_loss = F.mse_loss(counterfactual, input_tensor)
            
            total_loss = target_loss + lambda_reg * distance_loss
            
            total_loss.backward()
            optimizer.step()
            
            # Check if target class is predicted
            if prediction[0, target_class] > 0.5:
                break
        
        return counterfactual.detach(), iteration
```

## 9. Regulatory Compliance and AI Governance

### AI Governance Framework
```python
from typing import Dict, List, Any
import json
from datetime import datetime
import hashlib

class AIGovernanceFramework:
    """Comprehensive AI governance and compliance framework"""
    
    def __init__(self, regulatory_standard='FDA_510k'):
        self.regulatory_standard = regulatory_standard
        self.audit_trail = []
        self.compliance_checks = []
        
    def document_model_development(self, model_config, training_data_info, 
                                 validation_results):
        """Document complete model development process"""
        
        documentation = {
            'timestamp': datetime.now().isoformat(),
            'model_configuration': model_config,
            'training_data': {
                'size': training_data_info['size'],
                'distribution': training_data_info['distribution'],
                'quality_metrics': training_data_info['quality_metrics'],
                'data_lineage': training_data_info['lineage']
            },
            'validation_results': validation_results,
            'regulatory_standard': self.regulatory_standard
        }
        
        # Create immutable hash for integrity
        doc_hash = hashlib.sha256(
            json.dumps(documentation, sort_keys=True).encode()
        ).hexdigest()
        
        documentation['integrity_hash'] = doc_hash
        self.audit_trail.append(documentation)
        
        return doc_hash
    
    def perform_bias_assessment(self, model, test_dataset, protected_attributes):
        """Comprehensive bias assessment"""
        
        bias_metrics = {}
        
        # Demographic parity
        for attr in protected_attributes:
            group_predictions = {}
            group_labels = {}
            
            for data, labels in test_dataset:
                predictions = model(data).argmax(dim=1)
                
                # Group by protected attribute
                for value in torch.unique(data[attr]):
                    mask = data[attr] == value
                    if value.item() not in group_predictions:
                        group_predictions[value.item()] = []
                        group_labels[value.item()] = []
                    
                    group_predictions[value.item()].extend(
                        predictions[mask].tolist()
                    )
                    group_labels[value.item()].extend(
                        labels[mask].tolist()
                    )
            
            # Calculate metrics for each group
            group_metrics = {}
            for group, preds in group_predictions.items():
                labels = group_labels[group]
                
                # Positive prediction rate
                positive_rate = sum(preds) / len(preds)
                
                # Accuracy
                accuracy = sum(p == l for p, l in zip(preds, labels)) / len(preds)
                
                group_metrics[group] = {
                    'positive_rate': positive_rate,
                    'accuracy': accuracy,
                    'sample_size': len(preds)
                }
            
            bias_metrics[attr] = group_metrics
        
        return bias_metrics
    
    def generate_model_card(self, model, training_info, performance_metrics, 
                          limitations, intended_use):
        """Generate comprehensive model card"""
        
        model_card = {
            'model_details': {
                'name': training_info.get('name', 'SemiconductorClassifier'),
                'version': training_info.get('version', '1.0'),
                'date': datetime.now().isoformat(),
                'type': 'Image Classification',
                'architecture': training_info['architecture'],
                'parameters': sum(p.numel() for p in model.parameters()),
                'framework': 'PyTorch'
            },
            
            'intended_use': {
                'primary_uses': intended_use['primary'],
                'primary_users': intended_use['users'],
                'out_of_scope': intended_use['out_of_scope']
            },
            
            'factors': {
                'relevant_factors': [
                    'Image quality', 'Lighting conditions', 
                    'Magnification level', 'Sample preparation'
                ],
                'evaluation_factors': training_info['evaluation_factors']
            },
            
            'metrics': {
                'performance_measures': performance_metrics,
                'decision_thresholds': training_info.get('thresholds'),
                'confidence_intervals': performance_metrics.get('confidence_intervals')
            },
            
            'training_data': {
                'dataset_description': training_info['dataset_description'],
                'preprocessing': training_info['preprocessing_steps']
            },
            
            'evaluation_data': {
                'dataset_description': training_info['eval_dataset_description'],
                'motivation': training_info['eval_motivation']
            },
            
            'quantitative_analyses': {
                'unitary_results': performance_metrics['unitary_results'],
                'intersectional_results': performance_metrics.get('intersectional_results')
            },
            
            'ethical_considerations': {
                'risks_and_harms': limitations['risks'],
                'use_cases': limitations['inappropriate_uses'],
                'fairness_assessment': performance_metrics.get('bias_assessment')
            },
            
            'recommendations': limitations['recommendations']
        }
        
        return model_card

# Continuous monitoring for production models
class ProductionMonitor:
    """Continuous monitoring of production AI systems"""
    
    def __init__(self, model, reference_dataset):
        self.model = model
        self.reference_stats = self._compute_reference_stats(reference_dataset)
        self.alerts = []
    
    def _compute_reference_stats(self, dataset):
        """Compute reference statistics from training data"""
        features = []
        predictions = []
        
        self.model.eval()
        with torch.no_grad():
            for batch in dataset:
                inputs, labels = batch
                outputs = self.model(inputs)
                features.append(inputs.flatten(1).cpu())
                predictions.append(F.softmax(outputs, dim=1).cpu())
        
        features = torch.cat(features, dim=0)
        predictions = torch.cat(predictions, dim=0)
        
        return {
            'feature_mean': features.mean(dim=0),
            'feature_std': features.std(dim=0),
            'prediction_mean': predictions.mean(dim=0),
            'prediction_std': predictions.std(dim=0)
        }
    
    def monitor_batch(self, batch_inputs, batch_outputs=None):
        """Monitor a batch of predictions"""
        
        # Feature drift detection
        current_features = batch_inputs.flatten(1).cpu()
        feature_drift = self._detect_feature_drift(current_features)
        
        # Prediction drift detection
        with torch.no_grad():
            current_predictions = F.softmax(
                self.model(batch_inputs), dim=1
            ).cpu()
        
        prediction_drift = self._detect_prediction_drift(current_predictions)
        
        # Performance monitoring (if labels available)
        performance_degradation = None
        if batch_outputs is not None:
            performance_degradation = self._detect_performance_degradation(
                current_predictions, batch_outputs.cpu()
            )
        
        # Generate alerts if thresholds exceeded
        self._check_alerts(feature_drift, prediction_drift, performance_degradation)
        
        return {
            'feature_drift': feature_drift,
            'prediction_drift': prediction_drift,
            'performance_degradation': performance_degradation,
            'alerts': self.alerts[-10:]  # Last 10 alerts
        }
    
    def _detect_feature_drift(self, current_features, threshold=0.1):
        """Detect drift in input features"""
        current_mean = current_features.mean(dim=0)
        current_std = current_features.std(dim=0)
        
        # Wasserstein distance approximation
        mean_drift = torch.norm(current_mean - self.reference_stats['feature_mean'])
        std_drift = torch.norm(current_std - self.reference_stats['feature_std'])
        
        total_drift = (mean_drift + std_drift).item()
        
        return {
            'drift_score': total_drift,
            'drift_detected': total_drift > threshold
        }
    
    def _detect_prediction_drift(self, current_predictions, threshold=0.05):
        """Detect drift in model predictions"""
        current_mean = current_predictions.mean(dim=0)
        
        # KL divergence
        kl_div = F.kl_div(
            torch.log(current_mean + 1e-8),
            self.reference_stats['prediction_mean'],
            reduction='sum'
        ).item()
        
        return {
            'kl_divergence': kl_div,
            'drift_detected': kl_div > threshold
        }
    
    def _check_alerts(self, feature_drift, prediction_drift, performance_degradation):
        """Check for alert conditions"""
        timestamp = datetime.now().isoformat()
        
        if feature_drift['drift_detected']:
            self.alerts.append({
                'timestamp': timestamp,
                'type': 'feature_drift',
                'severity': 'medium',
                'details': feature_drift
            })
        
        if prediction_drift['drift_detected']:
            self.alerts.append({
                'timestamp': timestamp,
                'type': 'prediction_drift', 
                'severity': 'high',
                'details': prediction_drift
            })
        
        if performance_degradation and performance_degradation['degradation_detected']:
            self.alerts.append({
                'timestamp': timestamp,
                'type': 'performance_degradation',
                'severity': 'critical',
                'details': performance_degradation
            })
```

## 10. Future Directions and Emerging Trends

### Next-Generation AI Architectures
```python
# Vision-Language Models for Industrial Applications
class VisionLanguageIndustrialModel:
    """Vision-language model for industrial applications"""
    
    def __init__(self):
        # Load pre-trained vision-language model
        from transformers import BlipProcessor, BlipForConditionalGeneration
        
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    
    def analyze_with_context(self, image, text_query):
        """Analyze image with natural language context"""
        
        # Process inputs
        inputs = self.processor(image, text_query, return_tensors="pt")
        
        # Generate response
        out = self.model.generate(**inputs, max_length=100)
        response = self.processor.decode(out[0], skip_special_tokens=True)
        
        return response
    
    def few_shot_classification(self, support_images, support_texts, query_image):
        """Few-shot classification using vision-language understanding"""
        
        similarities = []
        
        for support_img, support_text in zip(support_images, support_texts):
            # Compute similarity between query and support
            query_inputs = self.processor(query_image, return_tensors="pt")
            support_inputs = self.processor(support_img, return_tensors="pt")
            
            # Extract embeddings (simplified)
            query_embedding = self.model.vision_model(**query_inputs).last_hidden_state.mean(dim=1)
            support_embedding = self.model.vision_model(**support_inputs).last_hidden_state.mean(dim=1)
            
            similarity = F.cosine_similarity(query_embedding, support_embedding)
            similarities.append(similarity.item())
        
        # Return most similar class
        best_match_idx = torch.argmax(torch.tensor(similarities))
        return best_match_idx.item(), similarities

# Neurosymbolic AI for Interpretable Industrial AI
class NeurosymbolicIndustrialAI:
    """Neurosymbolic approach combining neural networks with symbolic reasoning"""
    
    def __init__(self, neural_model, knowledge_base):
        self.neural_model = neural_model
        self.knowledge_base = knowledge_base  # Rules and constraints
    
    def symbolic_constraints(self, prediction, image_metadata):
        """Apply symbolic constraints to neural predictions"""
        
        # Example: Physical constraints for semiconductor manufacturing
        constraints = {
            'size_constraints': {
                'min_defect_size': 0.1,  # micrometers
                'max_defect_size': 100.0
            },
            'location_constraints': {
                'edge_exclusion_zone': 0.05,  # 5% from edge
                'forbidden_regions': []  # Specific coordinates
            },
            'temporal_constraints': {
                'max_defects_per_wafer': 10,
                'consistency_check': True
            }
        }
        
        # Apply constraints
        constrained_prediction = self._apply_constraints(
            prediction, image_metadata, constraints
        )
        
        return constrained_prediction
    
    def _apply_constraints(self, prediction, metadata, constraints):
        """Apply symbolic constraints to modify predictions"""
        
        # Size constraints
        if metadata.get('defect_size'):
            size = metadata['defect_size']
            if (size < constraints['size_constraints']['min_defect_size'] or 
                size > constraints['size_constraints']['max_defect_size']):
                # Reduce confidence for size violations
                prediction *= 0.1
        
        # Location constraints
        if metadata.get('location'):
            x, y = metadata['location']
            image_size = metadata.get('image_size', (224, 224))
            
            # Check edge exclusion
            edge_zone = constraints['location_constraints']['edge_exclusion_zone']
            if (x < edge_zone * image_size[0] or 
                x > (1 - edge_zone) * image_size[0] or
                y < edge_zone * image_size[1] or 
                y > (1 - edge_zone) * image_size[1]):
                # Reduce confidence for edge violations
                prediction *= 0.5
        
        return prediction

# Federated Learning for Privacy-Preserving Industrial AI
class FederatedIndustrialLearning:
    """Federated learning for industrial applications"""
    
    def __init__(self, global_model):
        self.global_model = global_model
        self.client_models = {}
        self.aggregation_weights = {}
    
    def federated_training_round(self, client_data, num_local_epochs=5):
        """Execute one round of federated training"""
        
        # Distribute global model to clients
        for client_id, (train_loader, _) in client_data.items():
            # Create local model copy
            local_model = copy.deepcopy(self.global_model)
            
            # Local training
            local_model = self._local_training(
                local_model, train_loader, num_local_epochs
            )
            
            self.client_models[client_id] = local_model
            self.aggregation_weights[client_id] = len(train_loader.dataset)
        
        # Aggregate models using FedAvg
        self._federated_averaging()
        
        return self.global_model
    
    def _local_training(self, model, train_loader, epochs):
        """Local training on client data"""
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
        
        return model
    
    def _federated_averaging(self):
        """FedAvg aggregation strategy"""
        
        # Calculate total samples
        total_samples = sum(self.aggregation_weights.values())
        
        # Initialize aggregated parameters
        aggregated_params = {}
        
        # Get parameter names from global model
        param_names = [name for name, _ in self.global_model.named_parameters()]
        
        for param_name in param_names:
            aggregated_params[param_name] = torch.zeros_like(
                dict(self.global_model.named_parameters())[param_name]
            )
        
        # Weighted aggregation
        for client_id, client_model in self.client_models.items():
            weight = self.aggregation_weights[client_id] / total_samples
            
            for param_name, param in client_model.named_parameters():
                aggregated_params[param_name] += weight * param.data
        
        # Update global model
        for param_name, param in self.global_model.named_parameters():
            param.data = aggregated_params[param_name]
```

## Key Industry Insights and Recommendations

### 1. **Data Strategy Evolution**
- **Synthetic Data Generation**: Use GANs and diffusion models to augment small datasets
- **Active Learning**: Reduce labeling costs by 40-60% through strategic sample selection
- **Cross-domain Transfer**: Leverage models trained on related industrial domains

### 2. **Architecture Trends**
- **Vision Transformers**: Superior performance on small datasets with proper pre-training
- **Hybrid Architectures**: Combining CNNs and ViTs for optimal performance
- **Foundation Models**: Adapt large pre-trained models rather than training from scratch

### 3. **Deployment Strategies**
- **Edge-First Design**: Optimize for edge deployment from the beginning
- **Model Versioning**: Implement comprehensive model lifecycle management
- **Continuous Learning**: Deploy systems that adapt to new data patterns

### 4. **Regulatory Compliance**
- **Documentation**: Maintain comprehensive audit trails for regulatory approval
- **Bias Assessment**: Regular evaluation for fairness across different conditions
- **Interpretability**: Build explainable systems for critical industrial applications

### 5. **Future-Proofing**
- **Modular Design**: Build systems that can incorporate new techniques
- **Scalable Infrastructure**: Design for growth from small to large datasets
- **Cross-modal Learning**: Prepare for integration of multiple data types

This comprehensive framework represents the cutting edge of ML for industrial applications, specifically optimized for your small data constraints in semiconductor manufacturing.

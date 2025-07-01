import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import timm
import numpy as np
import cv2
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict
import logging
from pathlib import Path
import json
import pickle

class SemiconductorImageDataset(Dataset):
    """Custom dataset for semiconductor images with advanced augmentation"""
    
    def __init__(self, image_paths: List[str], labels: List[int], 
                 transform=None, is_training=True):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.is_training = is_training
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        label = self.labels[idx]
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        return image, label

class AdvancedDataAugmentation:
    """Semiconductor-specific data augmentation strategies"""
    
    @staticmethod
    def get_training_transforms(image_size=224):
        """Heavy augmentation for training with small datasets"""
        return A.Compose([
            # Geometric transformations
            A.Resize(image_size, image_size),
            A.RandomRotate90(p=0.8),
            A.Flip(p=0.8),
            A.ShiftScaleRotate(
                shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.8
            ),
            
            # Optical/microscopy specific augmentations
            A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=0.5),
            A.GridDistortion(num_steps=5, distort_limit=0.1, p=0.5),
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),
            
            # Lighting and contrast (important for semiconductor imaging)
            A.RandomBrightnessContrast(
                brightness_limit=0.2, contrast_limit=0.2, p=0.8
            ),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
            A.RandomGamma(gamma_limit=(80, 120), p=0.5),
            
            # Noise simulation (sensor noise, thermal noise)
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.3),
            
            # Blur effects (defocus, motion blur)
            A.OneOf([
                A.MotionBlur(blur_limit=3, p=1.0),
                A.MedianBlur(blur_limit=3, p=1.0),
                A.GaussianBlur(blur_limit=3, p=1.0),
            ], p=0.3),
            
            # Normalization
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2(),
        ])
    
    @staticmethod
    def get_validation_transforms(image_size=224):
        """Minimal augmentation for validation"""
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2(),
        ])

class TransferLearningModel(nn.Module):
    """Transfer learning model optimized for small datasets"""
    
    def __init__(self, model_name='efficientnet_b3', num_classes=2, 
                 pretrained=True, dropout_rate=0.3):
        super(TransferLearningModel, self).__init__()
        
        # Load pre-trained model
        self.backbone = timm.create_model(
            model_name, 
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            global_pool='avg'
        )
        
        # Get feature dimension
        self.feature_dim = self.backbone.num_features
        
        # Custom classifier with dropout for regularization
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize classifier weights"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        features = self.backbone(x)
        output = self.classifier(features)
        return output
    
    def extract_features(self, x):
        """Extract features for analysis"""
        return self.backbone(x)

class FewShotLearningSystem:
    """Few-shot learning system for semiconductor image analysis"""
    
    def __init__(self, model_name='efficientnet_b3', num_classes=2, 
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.num_classes = num_classes
        self.model = TransferLearningModel(
            model_name=model_name, 
            num_classes=num_classes
        ).to(device)
        
        # Training components
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.optimizer = None
        self.scheduler = None
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def setup_training(self, learning_rate=1e-4, weight_decay=1e-4):
        """Setup optimizer and scheduler"""
        # Use different learning rates for backbone and classifier
        backbone_params = []
        classifier_params = []
        
        for name, param in self.model.named_parameters():
            if 'classifier' in name:
                classifier_params.append(param)
            else:
                backbone_params.append(param)
        
        self.optimizer = optim.AdamW([
            {'params': backbone_params, 'lr': learning_rate * 0.1},  # Lower LR for backbone
            {'params': classifier_params, 'lr': learning_rate}       # Higher LR for classifier
        ], weight_decay=weight_decay)
        
        # Cosine annealing scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, targets)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total
        
        # Calculate detailed metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, all_preds, average='weighted'
        )
        
        return avg_loss, accuracy, precision, recall, f1
    
    def train_with_cross_validation(self, image_paths, labels, n_splits=5, 
                                   epochs=100, batch_size=16, early_stopping_patience=15):
        """Train with cross-validation for robust evaluation"""
        
        # Convert to numpy arrays
        image_paths = np.array(image_paths)
        labels = np.array(labels)
        
        # Stratified K-Fold
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        cv_results = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(image_paths, labels)):
            self.logger.info(f"Training Fold {fold + 1}/{n_splits}")
            
            # Split data
            train_paths, val_paths = image_paths[train_idx], image_paths[val_idx]
            train_labels, val_labels = labels[train_idx], labels[val_idx]
            
            # Create datasets
            train_dataset = SemiconductorImageDataset(
                train_paths, train_labels,
                transform=AdvancedDataAugmentation.get_training_transforms(),
                is_training=True
            )
            
            val_dataset = SemiconductorImageDataset(
                val_paths, val_labels,
                transform=AdvancedDataAugmentation.get_validation_transforms(),
                is_training=False
            )
            
            # Create data loaders
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, 
                shuffle=True, num_workers=4, pin_memory=True
            )
            
            val_loader = DataLoader(
                val_dataset, batch_size=batch_size, 
                shuffle=False, num_workers=4, pin_memory=True
            )
            
            # Reset model and optimizer
            self.model = TransferLearningModel(
                num_classes=self.num_classes
            ).to(self.device)
            self.setup_training()
            
            # Training loop with early stopping
            best_val_accuracy = 0
            patience_counter = 0
            
            for epoch in range(epochs):
                train_loss, train_acc = self.train_epoch(train_loader)
                val_loss, val_acc, val_precision, val_recall, val_f1 = self.validate(val_loader)
                
                self.scheduler.step()
                
                # Early stopping
                if val_acc > best_val_accuracy:
                    best_val_accuracy = val_acc
                    patience_counter = 0
                    # Save best model
                    torch.save(self.model.state_dict(), f'best_model_fold_{fold}.pth')
                else:
                    patience_counter += 1
                
                if patience_counter >= early_stopping_patience:
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break
                
                if epoch % 10 == 0:
                    self.logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, "
                                   f"Val Acc: {val_acc:.2f}%")
            
            # Store fold results
            cv_results.append({
                'fold': fold,
                'best_accuracy': best_val_accuracy,
                'precision': val_precision,
                'recall': val_recall,
                'f1': val_f1
            })
        
        return cv_results
    
    def predict_with_uncertainty(self, image_paths, model_paths, num_tta=5):
        """Predict with uncertainty estimation using multiple models and TTA"""
        
        predictions = []
        uncertainties = []
        
        for image_path in image_paths:
            # Load image
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            fold_predictions = []
            
            # Get predictions from all fold models
            for model_path in model_paths:
                self.model.load_state_dict(torch.load(model_path))
                self.model.eval()
                
                tta_predictions = []
                
                # Test Time Augmentation
                for _ in range(num_tta):
                    transform = AdvancedDataAugmentation.get_training_transforms()
                    augmented = transform(image=image)
                    input_tensor = augmented['image'].unsqueeze(0).to(self.device)
                    
                    with torch.no_grad():
                        output = self.model(input_tensor)
                        probabilities = torch.softmax(output, dim=1)
                        tta_predictions.append(probabilities.cpu().numpy())
                
                # Average TTA predictions
                avg_tta_pred = np.mean(tta_predictions, axis=0)
                fold_predictions.append(avg_tta_pred)
            
            # Average across all folds
            final_prediction = np.mean(fold_predictions, axis=0)
            uncertainty = np.std(fold_predictions, axis=0)
            
            predictions.append(final_prediction)
            uncertainties.append(uncertainty)
        
        return np.array(predictions), np.array(uncertainties)

class ActiveLearningStrategy:
    """Active learning for efficient data collection"""
    
    def __init__(self, model_system):
        self.model_system = model_system
    
    def uncertainty_sampling(self, unlabeled_paths, model_paths, n_samples=10):
        """Select most uncertain samples for labeling"""
        
        predictions, uncertainties = self.model_system.predict_with_uncertainty(
            unlabeled_paths, model_paths
        )
        
        # Calculate entropy-based uncertainty
        entropies = []
        for pred in predictions:
            entropy = -np.sum(pred * np.log(pred + 1e-8), axis=1)
            entropies.append(entropy.max())
        
        # Select top uncertain samples
        uncertain_indices = np.argsort(entropies)[-n_samples:]
        
        return [unlabeled_paths[i] for i in uncertain_indices], entropies

class SemiconductorAnalysisSystem:
    """Complete system for semiconductor image analysis"""
    
    def __init__(self, config_path=None):
        # Default configuration
        self.config = {
            'model_name': 'efficientnet_b3',
            'image_size': 224,
            'batch_size': 16,
            'learning_rate': 1e-4,
            'epochs': 100,
            'n_splits': 5,
            'early_stopping_patience': 15,
            'num_classes': 2,
            'class_names': ['defective', 'normal']
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                self.config.update(json.load(f))
        
        self.model_system = FewShotLearningSystem(
            model_name=self.config['model_name'],
            num_classes=self.config['num_classes']
        )
        
        self.active_learning = ActiveLearningStrategy(self.model_system)
    
    def train_system(self, data_dir, save_dir='./models'):
        """Train the complete system"""
        
        # Create save directory
        Path(save_dir).mkdir(exist_ok=True)
        
        # Load data (assuming organized in subdirectories by class)
        image_paths, labels = self._load_data_from_directory(data_dir)
        
        print(f"Loaded {len(image_paths)} images with {len(set(labels))} classes")
        print(f"Class distribution: {np.bincount(labels)}")
        
        # Train with cross-validation
        cv_results = self.model_system.train_with_cross_validation(
            image_paths, labels,
            n_splits=self.config['n_splits'],
            epochs=self.config['epochs'],
            batch_size=self.config['batch_size']
        )
        
        # Save results
        results_summary = {
            'cv_results': cv_results,
            'mean_accuracy': np.mean([r['best_accuracy'] for r in cv_results]),
            'std_accuracy': np.std([r['best_accuracy'] for r in cv_results]),
            'mean_f1': np.mean([r['f1'] for r in cv_results]),
            'config': self.config
        }
        
        with open(f"{save_dir}/training_results.json", 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        print(f"Training completed!")
        print(f"Mean CV Accuracy: {results_summary['mean_accuracy']:.2f}% ± {results_summary['std_accuracy']:.2f}%")
        print(f"Mean F1 Score: {results_summary['mean_f1']:.3f}")
        
        return results_summary
    
    def _load_data_from_directory(self, data_dir):
        """Load image paths and labels from directory structure"""
        data_path = Path(data_dir)
        image_paths = []
        labels = []
        
        for class_idx, class_dir in enumerate(sorted(data_path.iterdir())):
            if class_dir.is_dir():
                for img_path in class_dir.glob('*.jpg'):  # Add other extensions as needed
                    image_paths.append(str(img_path))
                    labels.append(class_idx)
                for img_path in class_dir.glob('*.png'):
                    image_paths.append(str(img_path))
                    labels.append(class_idx)
        
        return image_paths, labels
    
    def predict_batch(self, image_paths, model_dir='./models'):
        """Predict on a batch of images"""
        
        # Get all model paths
        model_paths = list(Path(model_dir).glob('best_model_fold_*.pth'))
        
        if not model_paths:
            raise ValueError(f"No trained models found in {model_dir}")
        
        predictions, uncertainties = self.model_system.predict_with_uncertainty(
            image_paths, model_paths
        )
        
        # Convert to class predictions
        predicted_classes = np.argmax(predictions, axis=2)
        confidence_scores = np.max(predictions, axis=2)
        
        results = []
        for i, img_path in enumerate(image_paths):
            results.append({
                'image_path': img_path,
                'predicted_class': int(predicted_classes[i].mean()),
                'confidence': float(confidence_scores[i].mean()),
                'uncertainty': float(uncertainties[i].max()),
                'class_probabilities': predictions[i].mean(axis=0).tolist()
            })
        
        return results
    
    def suggest_next_samples(self, unlabeled_dir, model_dir='./models', n_samples=10):
        """Suggest next samples for labeling using active learning"""
        
        # Get unlabeled image paths
        unlabeled_paths = []
        for img_path in Path(unlabeled_dir).glob('*.jpg'):
            unlabeled_paths.append(str(img_path))
        for img_path in Path(unlabeled_dir).glob('*.png'):
            unlabeled_paths.append(str(img_path))
        
        # Get model paths
        model_paths = list(Path(model_dir).glob('best_model_fold_*.pth'))
        
        if not model_paths:
            raise ValueError(f"No trained models found in {model_dir}")
        
        # Get most uncertain samples
        suggested_paths, uncertainties = self.active_learning.uncertainty_sampling(
            unlabeled_paths, model_paths, n_samples
        )
        
        return suggested_paths, uncertainties

# Example usage and configuration
if __name__ == "__main__":
    # Configuration for semiconductor defect detection
    config = {
        'model_name': 'efficientnet_b3',  # Good balance of accuracy and efficiency
        'image_size': 224,
        'batch_size': 16,  # Small batch size for limited data
        'learning_rate': 1e-4,
        'epochs': 100,
        'n_splits': 5,
        'early_stopping_patience': 15,
        'num_classes': 2,
        'class_names': ['defective', 'normal']
    }
    
    # Save configuration
    with open('config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Initialize system
    system = SemiconductorAnalysisSystem('config.json')
    
    # Example training (uncomment when you have data)
    # results = system.train_system('./data/train')
    
    # Example prediction (uncomment when you have trained models)
    # predictions = system.predict_batch(['./test_images/sample1.jpg', './test_images/sample2.jpg'])
    
    # Example active learning (uncomment when you have unlabeled data)
    # suggested_samples, uncertainties = system.suggest_next_samples('./data/unlabeled', n_samples=5)
    
    print("Semiconductor Image Analysis System initialized!")
    print("Key features:")
    print("- Transfer learning with EfficientNet")
    print("- Advanced data augmentation for semiconductor images")
    print("- Cross-validation for robust evaluation")
    print("- Uncertainty estimation for reliable predictions")
    print("- Active learning for efficient data collection")
    print("- Early stopping to prevent overfitting")

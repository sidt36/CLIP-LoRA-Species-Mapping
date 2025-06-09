#!/usr/bin/env python3
"""
Comprehensive Evaluation Suite for CLIP-LoRA Species Classification
Enhanced version with detailed metrics, visualizations, and error analysis
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, top_k_accuracy_score,
    roc_auc_score, precision_recall_curve, average_precision_score,
    balanced_accuracy_score
)
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Import your existing functions
from utils import *
from loralib.utils import mark_only_lora_as_trainable, apply_lora, get_lora_parameters, lora_state_dict, save_lora, load_lora


class CLIPLoRAEvaluator:
    """Comprehensive evaluator for CLIP-LoRA species classification"""
    
    def __init__(self, model, dataset, test_loader, output_dir='clip_lora_evaluation', 
                 device='cuda', top_k=5, save_prefix="eval"):
        """
        Args:
            model: CLIP model with LoRA
            dataset: Dataset object with classnames and templates
            test_loader: Test data loader
            output_dir: Directory to save evaluation results
            device: Device to run evaluation on
            top_k: Top-k accuracy to calculate
            save_prefix: Prefix for saved files
        """
        self.model = model.to(device)
        self.dataset = dataset
        self.test_loader = test_loader
        self.device = device
        self.top_k = top_k
        self.save_prefix = save_prefix
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Get class information
        self.classnames = dataset.classnames
        self.num_classes = len(self.classnames)
        self.template = dataset.template[0]
        
        # Create subdirectories
        self.dirs = {
            'metrics': self.output_dir / 'metrics',
            'confusion_matrices': self.output_dir / 'confusion_matrices',
            'error_analysis': self.output_dir / 'error_analysis',
            'visualizations': self.output_dir / 'visualizations',
            'per_class_results': self.output_dir / 'per_class_results',
            'predictions': self.output_dir / 'predictions'
        }
        
        for dir_path in self.dirs.values():
            dir_path.mkdir(exist_ok=True, parents=True)
        
        # Prepare text features
        self._prepare_text_features()
        
        # Storage for results
        self.results = {
            'predictions': [],
            'probabilities': [],
            'targets': [],
            'filenames': [],
            'confidences': [],
            'image_features': [],
            'top_k_predictions': []
        }
    
    def _prepare_text_features(self):
        """Prepare text features for all classes"""
        self.model.eval()
        texts = [self.template.format(classname.replace('_', ' ')) for classname in self.classnames]
        
        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                texts = clip.tokenize(texts).to(self.device)
                class_embeddings = self.model.encode_text(texts)
        
        self.text_features = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
    
    def evaluate(self):
        """Run complete evaluation"""
        print("Running comprehensive CLIP-LoRA evaluation...")
        
        # Collect predictions
        self._collect_predictions()
        
        # Calculate metrics
        overall_metrics = self._calculate_overall_metrics()
        per_class_metrics = self._calculate_per_class_metrics()
        
        # Generate confusion matrices
        self._generate_confusion_matrices()
        
        # Perform error analysis
        error_analysis = self._perform_error_analysis()
        
        # Create visualizations
        self._create_visualizations()
        
        # Save per-class results
        self._save_per_class_results(per_class_metrics)
        
        # Save detailed error analysis
        self._save_detailed_error_analysis(error_analysis)
        
        # Save comprehensive report
        self._save_report(overall_metrics, per_class_metrics, error_analysis)
        
        # Create feature analysis
        self._analyze_features()
        
        return overall_metrics, per_class_metrics
    
    def _collect_predictions(self):
        """Collect predictions from the model"""
        self.model.eval()
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Collecting predictions"):
                # Handle both dict and tuple batch formats
                if isinstance(batch, dict):
                    images = batch['image'].to(self.device)
                    targets = batch.get('label', batch.get('species_index', batch.get('target')))
                    if targets is None:
                        raise ValueError("Batch dict must contain 'label', 'species_index', or 'target'")
                    targets = targets.to(self.device)
                    filenames = batch.get('filename', [f"sample_{i}" for i in range(len(images))])
                else:
                    images, targets = batch
                    images, targets = images.to(self.device), targets.to(self.device)
                    filenames = [f"sample_{i}" for i in range(len(images))]
                
                # Get image features
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    image_features = self.model.encode_image(images)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                # Calculate similarities
                logits = image_features @ self.text_features.t()
                probabilities = torch.softmax(logits, dim=1)
                predictions = torch.argmax(logits, dim=1)
                confidences = torch.max(probabilities, dim=1)[0]
                
                # Get top-k predictions
                top_k_probs, top_k_preds = torch.topk(probabilities, min(self.top_k, self.num_classes), dim=1)
                
                # Store results
                self.results['predictions'].extend(predictions.cpu().numpy())
                self.results['probabilities'].extend(probabilities.cpu().numpy())
                self.results['targets'].extend(targets.cpu().numpy())
                self.results['confidences'].extend(confidences.cpu().numpy())
                self.results['filenames'].extend(filenames)
                self.results['image_features'].extend(image_features.cpu().numpy())
                self.results['top_k_predictions'].extend(top_k_preds.cpu().numpy())
        
        # Convert to numpy arrays
        for key in ['predictions', 'probabilities', 'targets', 'confidences', 'image_features', 'top_k_predictions']:
            self.results[key] = np.array(self.results[key])
        
        print(f"Evaluated {len(self.results['predictions'])} samples")
    
    def _calculate_overall_metrics(self):
        """Calculate overall classification metrics"""
        y_true = self.results['targets']
        y_pred = self.results['predictions']
        y_prob = self.results['probabilities']
        
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
        
        # F1 scores
        metrics['f1_micro'] = f1_score(y_true, y_pred, average='micro')
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro')
        metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted')
        
        # Precision and Recall
        metrics['precision_micro'] = precision_score(y_true, y_pred, average='micro')
        metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall_micro'] = recall_score(y_true, y_pred, average='micro')
        metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Top-k accuracy
        for k in [1, 3, 5]:
            if k <= self.num_classes:
                metrics[f'top_{k}_accuracy'] = self._calculate_top_k_accuracy(y_true, y_prob, k)
        
        # Confidence statistics
        metrics['mean_confidence'] = float(np.mean(self.results['confidences']))
        metrics['std_confidence'] = float(np.std(self.results['confidences']))
        metrics['min_confidence'] = float(np.min(self.results['confidences']))
        metrics['max_confidence'] = float(np.max(self.results['confidences']))
        
        # Per-class ROC-AUC (one-vs-rest)
        if self.num_classes > 2:
            try:
                from sklearn.preprocessing import label_binarize
                y_true_bin = label_binarize(y_true, classes=range(self.num_classes))
                metrics['roc_auc_macro'] = roc_auc_score(y_true_bin, y_prob, average='macro')
                metrics['roc_auc_weighted'] = roc_auc_score(y_true_bin, y_prob, average='weighted')
            except:
                metrics['roc_auc_macro'] = 0.0
                metrics['roc_auc_weighted'] = 0.0
        
        return metrics
    
    def _calculate_top_k_accuracy(self, y_true, y_prob, k):
        """Calculate top-k accuracy"""
        top_k_preds = np.argsort(y_prob, axis=1)[:, -k:]
        correct = 0
        for i, true_label in enumerate(y_true):
            if true_label in top_k_preds[i]:
                correct += 1
        return correct / len(y_true)
    
    def _calculate_per_class_metrics(self):
        """Calculate metrics for each class"""
        y_true = self.results['targets']
        y_pred = self.results['predictions']
        y_prob = self.results['probabilities']
        
        per_class_metrics = {}
        
        for class_idx in range(self.num_classes):
            class_name = self.classnames[class_idx]
            
            # Binary classification metrics for this class
            y_true_binary = (y_true == class_idx).astype(int)
            y_pred_binary = (y_pred == class_idx).astype(int)
            y_prob_binary = y_prob[:, class_idx]
            
            # Calculate metrics
            metrics = {
                'class_id': int(class_idx),
                'support': int(np.sum(y_true_binary)),
                'predicted_count': int(np.sum(y_pred_binary)),
                'precision': precision_score(y_true_binary, y_pred_binary, zero_division=0),
                'recall': recall_score(y_true_binary, y_pred_binary, zero_division=0),
                'f1_score': f1_score(y_true_binary, y_pred_binary, zero_division=0),
                'accuracy': accuracy_score(y_true_binary, y_pred_binary)
            }
            
            # ROC AUC and Average Precision if we have positive samples
            if metrics['support'] > 0 and len(np.unique(y_true_binary)) > 1:
                try:
                    metrics['roc_auc'] = roc_auc_score(y_true_binary, y_prob_binary)
                    metrics['average_precision'] = average_precision_score(y_true_binary, y_prob_binary)
                except:
                    metrics['roc_auc'] = 0.0
                    metrics['average_precision'] = 0.0
            else:
                metrics['roc_auc'] = 0.0
                metrics['average_precision'] = 0.0
            
            # Confusion matrix values
            cm = confusion_matrix(y_true_binary, y_pred_binary)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                metrics['true_positives'] = int(tp)
                metrics['false_positives'] = int(fp)
                metrics['true_negatives'] = int(tn)
                metrics['false_negatives'] = int(fn)
                metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            
            # Average confidence for this class
            class_mask = y_pred == class_idx
            if np.any(class_mask):
                metrics['avg_confidence'] = float(np.mean(self.results['confidences'][class_mask]))
            else:
                metrics['avg_confidence'] = 0.0
            
            per_class_metrics[class_name] = metrics
        
        return per_class_metrics
    
    def _generate_confusion_matrices(self):
        """Generate and save confusion matrices"""
        y_true = self.results['targets']
        y_pred = self.results['predictions']
        
        # Get unique classes that actually appear in the test set
        unique_true = np.unique(y_true)
        unique_pred = np.unique(y_pred)
        unique_classes = np.unique(np.concatenate([unique_true, unique_pred]))
        n_unique = len(unique_classes)
        
        # Map class indices to positions in confusion matrix
        class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
        
        # Calculate confusion matrix for classes that appear in test set
        cm = confusion_matrix(y_true, y_pred, labels=unique_classes)
        
        # Get class names for the classes that appear
        present_classnames = [self.classnames[i] for i in unique_classes]
        
        # Create normalized confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized)  # Handle division by zero
        
        # Determine figure size based on number of classes present
        fig_width = max(12, n_unique * 0.5)
        fig_height = max(10, n_unique * 0.4)
        
        # Plot normalized confusion matrix
        plt.figure(figsize=(fig_width, fig_height))
        
        # Use smaller font and annotations for large matrices
        annot = n_unique <= 30
        fmt = '.2f' if annot else ''
        
        sns.heatmap(cm_normalized, annot=annot, fmt=fmt, cmap='Blues',
                   xticklabels=present_classnames, yticklabels=present_classnames,
                   cbar_kws={'label': 'Normalized Frequency'})
        plt.title(f'Normalized Confusion Matrix ({n_unique}/{self.num_classes} classes present)')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks(rotation=45, ha='right', fontsize=8 if n_unique > 20 else 10)
        plt.yticks(rotation=0, fontsize=8 if n_unique > 20 else 10)
        plt.tight_layout()
        plt.savefig(self.dirs['confusion_matrices'] / f'{self.save_prefix}_normalized_confusion_matrix.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot raw counts confusion matrix for smaller datasets
        if n_unique <= 20:
            plt.figure(figsize=(fig_width, fig_height))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=present_classnames, yticklabels=present_classnames,
                       cbar_kws={'label': 'Count'})
            plt.title(f'Confusion Matrix - Raw Counts ({n_unique}/{self.num_classes} classes present)')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.savefig(self.dirs['confusion_matrices'] / f'{self.save_prefix}_raw_confusion_matrix.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        # Save confusion matrix data
        cm_df = pd.DataFrame(cm, index=present_classnames, columns=present_classnames)
        cm_df.to_csv(self.dirs['confusion_matrices'] / f'{self.save_prefix}_confusion_matrix.csv')
        
        cm_norm_df = pd.DataFrame(cm_normalized, index=present_classnames, columns=present_classnames)
        cm_norm_df.to_csv(self.dirs['confusion_matrices'] / f'{self.save_prefix}_confusion_matrix_normalized.csv')
        
        # Save a mapping of which classes were present
        class_presence = {
            'total_classes': self.num_classes,
            'classes_in_test': n_unique,
            'present_class_ids': unique_classes.tolist(),
            'present_class_names': present_classnames,
            'missing_class_ids': [i for i in range(self.num_classes) if i not in unique_classes],
            'missing_class_names': [self.classnames[i] for i in range(self.num_classes) if i not in unique_classes]
        }
        
        with open(self.dirs['confusion_matrices'] / f'{self.save_prefix}_class_presence.json', 'w') as f:
            json.dump(class_presence, f, indent=2)
    
    def _perform_error_analysis(self):
        """Analyze prediction errors"""
        y_true = self.results['targets']
        y_pred = self.results['predictions']
        confidences = self.results['confidences']
        
        error_analysis = {}
        
        # Find misclassified samples
        misclassified_mask = y_true != y_pred
        misclassified_indices = np.where(misclassified_mask)[0]
        
        error_analysis['total_errors'] = len(misclassified_indices)
        error_analysis['error_rate'] = len(misclassified_indices) / len(y_true)
        
        # Analyze confidence of errors
        if len(misclassified_indices) > 0:
            error_confidences = confidences[misclassified_indices]
            
            error_analysis['error_confidence_stats'] = {
                'mean': float(np.mean(error_confidences)),
                'std': float(np.std(error_confidences)),
                'median': float(np.median(error_confidences)),
                'min': float(np.min(error_confidences)),
                'max': float(np.max(error_confidences))
            }
        else:
            error_analysis['error_confidence_stats'] = {
                'mean': 0.0,
                'std': 0.0,
                'median': 0.0,
                'min': 0.0,
                'max': 0.0
            }
        
        # Handle correct predictions (may be empty if accuracy is 0)
        correct_confidences = confidences[~misclassified_mask]
        if len(correct_confidences) > 0:
            error_analysis['correct_confidence_stats'] = {
                'mean': float(np.mean(correct_confidences)),
                'std': float(np.std(correct_confidences)),
                'median': float(np.median(correct_confidences)),
                'min': float(np.min(correct_confidences)),
                'max': float(np.max(correct_confidences))
            }
        else:
            error_analysis['correct_confidence_stats'] = {
                'mean': 0.0,
                'std': 0.0,
                'median': 0.0,
                'min': 0.0,
                'max': 0.0
            }
        
        # Most confused class pairs
        confused_pairs = defaultdict(int)
        for i in misclassified_indices:
            true_class = y_true[i]
            pred_class = y_pred[i]
            confused_pairs[(true_class, pred_class)] += 1
        
        # Top confused pairs
        top_confused = sorted(confused_pairs.items(), key=lambda x: x[1], reverse=True)[:20]
        error_analysis['most_confused_pairs'] = [
            {
                'true_class': self.classnames[pair[0]],
                'predicted_class': self.classnames[pair[1]],
                'true_class_id': int(pair[0]),
                'predicted_class_id': int(pair[1]),
                'count': count
            }
            for (pair, count) in top_confused
        ]
        
        # High confidence errors
        if len(misclassified_indices) > 0:
            high_conf_threshold = np.percentile(confidences[misclassified_indices], 75)
            high_conf_errors = misclassified_indices[confidences[misclassified_indices] > high_conf_threshold]
            
            error_examples = []
            for idx in high_conf_errors[:30]:  # Top 30 high-confidence errors
                error_examples.append({
                    'sample_index': int(idx),
                    'filename': self.results['filenames'][idx],
                    'true_species': self.classnames[y_true[idx]],
                    'predicted_species': self.classnames[y_pred[idx]],
                    'true_class_id': int(y_true[idx]),
                    'predicted_class_id': int(y_pred[idx]),
                    'confidence': float(confidences[idx]),
                    'true_class_prob': float(self.results['probabilities'][idx][y_true[idx]]),
                    'top_k_predictions': [self.classnames[pred_idx] for pred_idx in self.results['top_k_predictions'][idx]]
                })
            
            error_analysis['high_confidence_errors'] = error_examples
        
        # Per-class error rates
        per_class_errors = {}
        for class_idx in range(self.num_classes):
            class_name = self.classnames[class_idx]
            class_mask = y_true == class_idx
            if np.any(class_mask):
                class_errors = np.sum(misclassified_mask[class_mask])
                class_total = np.sum(class_mask)
                
                per_class_errors[class_name] = {
                    'class_id': int(class_idx),
                    'total_samples': int(class_total),
                    'errors': int(class_errors),
                    'error_rate': float(class_errors / class_total) if class_total > 0 else 0.0,
                    'accuracy': float(1 - (class_errors / class_total)) if class_total > 0 else 0.0
                }
        
        error_analysis['per_class_error_rates'] = per_class_errors
        
        return error_analysis
    
    def _analyze_features(self):
        """Analyze feature space and similarities"""
        print("Analyzing feature space...")
        
        # Calculate average intra-class and inter-class similarities
        image_features = self.results['image_features']
        targets = self.results['targets']
        
        # Normalize features
        image_features = image_features / np.linalg.norm(image_features, axis=1, keepdims=True)
        
        intra_class_sims = []
        inter_class_sims = []
        
        for class_idx in range(min(self.num_classes, 20)):  # Limit to 20 classes for efficiency
            class_mask = targets == class_idx
            if np.sum(class_mask) < 2:
                continue
            
            class_features = image_features[class_mask]
            other_features = image_features[~class_mask]
            
            # Intra-class similarities
            if len(class_features) > 1:
                intra_sims = np.dot(class_features, class_features.T)
                mask = np.triu(np.ones_like(intra_sims), k=1).astype(bool)
                intra_class_sims.extend(intra_sims[mask])
            
            # Inter-class similarities (sample for efficiency)
            if len(other_features) > 0:
                n_samples = min(len(class_features), 100)
                sample_idx = np.random.choice(len(other_features), 
                                            min(n_samples, len(other_features)), 
                                            replace=False)
                inter_sims = np.dot(class_features, other_features[sample_idx].T)
                inter_class_sims.extend(inter_sims.flatten())
        
        # Plot similarity distributions
        plt.figure(figsize=(10, 6))
        plt.hist(intra_class_sims, bins=50, alpha=0.7, label='Intra-class', density=True)
        plt.hist(inter_class_sims, bins=50, alpha=0.7, label='Inter-class', density=True)
        plt.xlabel('Cosine Similarity')
        plt.ylabel('Density')
        plt.title('Feature Similarity Distributions')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.dirs['visualizations'] / f'{self.save_prefix}_feature_similarities.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save feature analysis
        feature_analysis = {
            'intra_class_similarity_mean': float(np.mean(intra_class_sims)) if intra_class_sims else 0.0,
            'intra_class_similarity_std': float(np.std(intra_class_sims)) if intra_class_sims else 0.0,
            'inter_class_similarity_mean': float(np.mean(inter_class_sims)) if inter_class_sims else 0.0,
            'inter_class_similarity_std': float(np.std(inter_class_sims)) if inter_class_sims else 0.0,
            'feature_separability': float(np.mean(intra_class_sims) - np.mean(inter_class_sims)) if intra_class_sims and inter_class_sims else 0.0
        }
        
        with open(self.dirs['visualizations'] / f'{self.save_prefix}_feature_analysis.json', 'w') as f:
            json.dump(feature_analysis, f, indent=2)
    
    def _save_detailed_error_analysis(self, error_analysis):
        """Save detailed error analysis files"""
        # Save main error analysis
        with open(self.dirs['error_analysis'] / f'{self.save_prefix}_error_analysis.json', 'w') as f:
            json.dump(error_analysis, f, indent=2)
        
        # Save high confidence errors
        if 'high_confidence_errors' in error_analysis:
            high_conf_df = pd.DataFrame(error_analysis['high_confidence_errors'])
            high_conf_df.to_csv(self.dirs['error_analysis'] / f'{self.save_prefix}_high_confidence_errors.csv', 
                               index=False)
        
        # Save confused pairs
        if 'most_confused_pairs' in error_analysis:
            confused_df = pd.DataFrame(error_analysis['most_confused_pairs'])
            confused_df.to_csv(self.dirs['error_analysis'] / f'{self.save_prefix}_most_confused_pairs.csv', 
                              index=False)
        
        # Save per-class error rates
        if 'per_class_error_rates' in error_analysis:
            error_rates_df = pd.DataFrame(error_analysis['per_class_error_rates']).T
            error_rates_df.to_csv(self.dirs['error_analysis'] / f'{self.save_prefix}_per_class_error_rates.csv')
    
    def _save_per_class_results(self, per_class_metrics):
        """Save detailed per-class results"""
        # Save overall per-class metrics
        per_class_df = pd.DataFrame(per_class_metrics).T
        per_class_df.to_csv(self.dirs['per_class_results'] / f'{self.save_prefix}_per_class_metrics.csv')
        
        # Create summary plots for top and bottom performers
        sorted_classes = sorted(per_class_metrics.items(), 
                               key=lambda x: x[1]['f1_score'], reverse=True)
        
        # Save top and bottom performers
        top_performers = sorted_classes[:10]
        bottom_performers = sorted_classes[-10:]
        
        performance_summary = {
            'top_10_classes': {name: {'f1_score': metrics['f1_score'], 
                                     'support': metrics['support']} 
                              for name, metrics in top_performers},
            'bottom_10_classes': {name: {'f1_score': metrics['f1_score'], 
                                        'support': metrics['support']} 
                                 for name, metrics in bottom_performers}
        }
        
        with open(self.dirs['per_class_results'] / f'{self.save_prefix}_performance_summary.json', 'w') as f:
            json.dump(performance_summary, f, indent=2)
    
    def _create_visualizations(self):
        """Create comprehensive visualizations"""
        # 1. Class distribution
        self._plot_class_distribution()
        
        # 2. Confidence distributions
        self._plot_confidence_distributions()
        
        # 3. Per-class performance
        self._plot_per_class_performance()
        
        # 4. Error analysis plots
        self._plot_error_analysis()
        
        # 5. Top-k accuracy curve
        self._plot_topk_accuracy()
        
        # 6. Calibration plot
        self._plot_calibration()
    
    def _plot_class_distribution(self):
        """Plot class distribution in test set"""
        y_true = self.results['targets']
        
        # Count samples per class
        unique_classes, counts = np.unique(y_true, return_counts=True)
        
        # Get class names for classes that appear in test set
        present_classnames = [self.classnames[i] for i in unique_classes]
        
        # Sort by frequency
        sorted_indices = np.argsort(counts)[::-1]
        
        plt.figure(figsize=(12, 8))
        bars = plt.bar(range(len(counts)), counts[sorted_indices])
        
        # Only show labels for top classes if many classes
        n_present = len(unique_classes)
        if n_present > 30:
            xtick_indices = list(range(0, len(counts), max(1, len(counts)//20)))
            plt.xticks(xtick_indices, 
                      [present_classnames[sorted_indices[i]] for i in xtick_indices], 
                      rotation=45, ha='right')
        else:
            plt.xticks(range(len(counts)), 
                      [present_classnames[sorted_indices[i]] for i in range(len(counts))], 
                      rotation=45, ha='right')
        
        plt.xlabel('Species')
        plt.ylabel('Number of Samples')
        plt.title(f'Test Set Class Distribution ({n_present}/{self.num_classes} classes present)')
        plt.yscale('log')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Color bars by frequency
        colors = plt.cm.viridis(np.linspace(0, 1, len(bars)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        plt.tight_layout()
        plt.savefig(self.dirs['visualizations'] / f'{self.save_prefix}_class_distribution.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save class distribution data
        class_dist_df = pd.DataFrame({
            'class_id': unique_classes[sorted_indices],
            'class_name': [present_classnames[i] for i in sorted_indices],
            'count': counts[sorted_indices]
        })
        class_dist_df.to_csv(self.dirs['visualizations'] / f'{self.save_prefix}_class_distribution.csv', 
                            index=False)
    
    def _plot_confidence_distributions(self):
        """Plot confidence distributions"""
        y_true = self.results['targets']
        y_pred = self.results['predictions']
        confidences = self.results['confidences']
        
        correct_mask = y_true == y_pred
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Overall confidence distribution
        correct_conf = confidences[correct_mask]
        incorrect_conf = confidences[~correct_mask]
        
        if len(correct_conf) > 0:
            axes[0].hist(correct_conf, bins=50, alpha=0.7, 
                        label=f'Correct (n={np.sum(correct_mask)})', color='green', density=True)
        if len(incorrect_conf) > 0:
            axes[0].hist(incorrect_conf, bins=50, alpha=0.7, 
                        label=f'Incorrect (n={np.sum(~correct_mask)})', color='red', density=True)
        
        axes[0].set_xlabel('Confidence Score')
        axes[0].set_ylabel('Density')
        axes[0].set_title('Confidence Distribution: Correct vs Incorrect')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Box plot by correctness
        data_to_plot = []
        labels = []
        if len(correct_conf) > 0:
            data_to_plot.append(correct_conf)
            labels.append('Correct')
        if len(incorrect_conf) > 0:
            data_to_plot.append(incorrect_conf)
            labels.append('Incorrect')
        
        if data_to_plot:
            axes[1].boxplot(data_to_plot, labels=labels)
            axes[1].set_ylabel('Confidence Score')
            axes[1].set_title('Confidence Distribution Comparison')
            axes[1].grid(True, alpha=0.3, axis='y')
        else:
            axes[1].text(0.5, 0.5, 'No data to display', ha='center', va='center')
            axes[1].set_title('Confidence Distribution Comparison')
        
        plt.tight_layout()
        plt.savefig(self.dirs['visualizations'] / f'{self.save_prefix}_confidence_distributions.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_per_class_performance(self):
        """Plot per-class performance metrics"""
        per_class_metrics = self._calculate_per_class_metrics()
        
        # Extract metrics
        class_names = list(per_class_metrics.keys())
        f1_scores = [metrics['f1_score'] for metrics in per_class_metrics.values()]
        supports = [metrics['support'] for metrics in per_class_metrics.values()]
        
        # Sort by F1 score
        sorted_indices = np.argsort(f1_scores)[::-1]
        
        # Select top and bottom classes for visualization
        n_show = min(30, len(class_names))
        show_indices = list(sorted_indices[:n_show//2]) + list(sorted_indices[-(n_show//2):])
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # F1 scores
        y_pos = np.arange(len(show_indices))
        colors = plt.cm.RdYlGn([f1_scores[i] for i in show_indices])
        
        ax1.barh(y_pos, [f1_scores[i] for i in show_indices], color=colors)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels([class_names[i] for i in show_indices], fontsize=8)
        ax1.set_xlabel('F1 Score')
        ax1.set_title(f'Per-Class F1 Scores (Top {n_show//2} and Bottom {n_show//2})')
        ax1.grid(axis='x', alpha=0.3)
        
        # Support distribution
        ax2.bar(range(len(supports)), sorted(supports, reverse=True))
        ax2.set_xlabel('Class (sorted by support)')
        ax2.set_ylabel('Number of samples')
        ax2.set_title('Per-Class Support Distribution')
        ax2.set_yscale('log')
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.dirs['visualizations'] / f'{self.save_prefix}_per_class_performance.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_error_analysis(self):
        """Plot error analysis visualizations"""
        error_analysis = self._perform_error_analysis()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Error rate by class (top errors)
        if 'per_class_error_rates' in error_analysis:
            error_data = error_analysis['per_class_error_rates']
            
            # Sort by error rate and get top 20
            sorted_classes = sorted(error_data.items(), 
                                  key=lambda x: x[1]['error_rate'], 
                                  reverse=True)[:20]
            
            class_names = [item[0] for item in sorted_classes]
            error_rates = [item[1]['error_rate'] for item in sorted_classes]
            supports = [item[1]['total_samples'] for item in sorted_classes]
            
            y_pos = np.arange(len(class_names))
            bars = axes[0, 0].barh(y_pos, error_rates)
            
            # Color by support
            colors = plt.cm.viridis(np.array(supports) / max(supports))
            for bar, color in zip(bars, colors):
                bar.set_color(color)
            
            axes[0, 0].set_yticks(y_pos)
            axes[0, 0].set_yticklabels(class_names, fontsize=8)
            axes[0, 0].set_xlabel('Error Rate')
            axes[0, 0].set_title('Top 20 Classes by Error Rate')
            axes[0, 0].grid(axis='x', alpha=0.3)
        
        # 2. Confidence comparison
        if 'error_confidence_stats' in error_analysis:
            categories = ['Errors', 'Correct']
            means = [error_analysis['error_confidence_stats']['mean'], 
                    error_analysis['correct_confidence_stats']['mean']]
            stds = [error_analysis['error_confidence_stats']['std'], 
                   error_analysis['correct_confidence_stats']['std']]
            
            axes[0, 1].bar(categories, means, yerr=stds, capsize=5, 
                          color=['red', 'green'], alpha=0.7)
            axes[0, 1].set_ylabel('Mean Confidence')
            axes[0, 1].set_title('Confidence: Errors vs Correct')
            axes[0, 1].grid(axis='y', alpha=0.3)
        
        # 3. Most confused pairs
        if 'most_confused_pairs' in error_analysis:
            confused_pairs = error_analysis['most_confused_pairs'][:10]
            pair_labels = [f"{pair['true_class'][:15]}\n→\n{pair['predicted_class'][:15]}" 
                          for pair in confused_pairs]
            counts = [pair['count'] for pair in confused_pairs]
            
            axes[1, 0].bar(range(len(pair_labels)), counts, color='coral')
            axes[1, 0].set_xticks(range(len(pair_labels)))
            axes[1, 0].set_xticklabels(pair_labels, rotation=0, ha='center', fontsize=8)
            axes[1, 0].set_ylabel('Number of Confusions')
            axes[1, 0].set_title('Top 10 Most Confused Class Pairs')
            axes[1, 0].grid(axis='y', alpha=0.3)
        
        # 4. Error distribution summary
        total_samples = len(self.results['predictions'])
        correct_samples = total_samples - error_analysis['total_errors']
        
        labels = ['Correct', 'Errors']
        sizes = [correct_samples, error_analysis['total_errors']]
        colors = ['lightgreen', 'lightcoral']
        explode = (0.1, 0)
        
        axes[1, 1].pie(sizes, explode=explode, labels=labels, colors=colors, 
                      autopct='%1.1f%%', shadow=True, startangle=90)
        axes[1, 1].set_title('Overall Classification Results')
        
        plt.tight_layout()
        plt.savefig(self.dirs['visualizations'] / f'{self.save_prefix}_error_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_topk_accuracy(self):
        """Plot top-k accuracy curve"""
        y_true = self.results['targets']
        y_prob = self.results['probabilities']
        
        k_values = range(1, min(11, self.num_classes + 1))
        topk_accs = []
        
        for k in k_values:
            acc = self._calculate_top_k_accuracy(y_true, y_prob, k)
            topk_accs.append(acc)
        
        plt.figure(figsize=(8, 6))
        plt.plot(k_values, topk_accs, 'o-', linewidth=2, markersize=8)
        plt.xlabel('k')
        plt.ylabel('Top-k Accuracy')
        plt.title('Top-k Accuracy Curve')
        plt.grid(True, alpha=0.3)
        plt.xticks(k_values)
        
        # Add value labels
        for k, acc in zip(k_values, topk_accs):
            plt.annotate(f'{acc:.3f}', (k, acc), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(self.dirs['visualizations'] / f'{self.save_prefix}_topk_accuracy.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_calibration(self):
        """Plot confidence calibration"""
        y_true = self.results['targets']
        y_pred = self.results['predictions']
        confidences = self.results['confidences']
        
        # Bin predictions by confidence
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []
        
        for i in range(n_bins):
            mask = (confidences >= bin_boundaries[i]) & (confidences < bin_boundaries[i + 1])
            if i == n_bins - 1:  # Include 1.0 in the last bin
                mask = (confidences >= bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
            
            if np.sum(mask) > 0:
                bin_accuracy = np.mean((y_true == y_pred)[mask])
                bin_confidence = np.mean(confidences[mask])
                bin_accuracies.append(bin_accuracy)
                bin_confidences.append(bin_confidence)
                bin_counts.append(np.sum(mask))
        
        # Expected Calibration Error (ECE)
        ece = 0
        for acc, conf, count in zip(bin_accuracies, bin_confidences, bin_counts):
            ece += (count / len(confidences)) * abs(acc - conf)
        
        # Plot
        plt.figure(figsize=(8, 8))
        plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
        plt.plot(bin_confidences, bin_accuracies, 'o-', linewidth=2, markersize=8, 
                label=f'Model (ECE={ece:.3f})')
        
        plt.xlabel('Mean Confidence')
        plt.ylabel('Accuracy')
        plt.title('Confidence Calibration Plot')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.dirs['visualizations'] / f'{self.save_prefix}_calibration.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _save_report(self, overall_metrics, per_class_metrics, error_analysis):
        """Save comprehensive evaluation report"""
        report = {
            'evaluation_date': datetime.now().isoformat(),
            'dataset_info': {
                'num_samples': len(self.results['predictions']),
                'num_classes': self.num_classes,
                'class_names': self.classnames
            },
            'overall_metrics': overall_metrics,
            'per_class_metrics': per_class_metrics,
            'error_analysis': error_analysis
        }
        
        # Save JSON report
        with open(self.output_dir / f'{self.save_prefix}_evaluation_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        # Save predictions
        predictions_df = pd.DataFrame({
            'filename': self.results['filenames'],
            'true_class': [self.classnames[i] for i in self.results['targets']],
            'predicted_class': [self.classnames[i] for i in self.results['predictions']],
            'true_class_id': self.results['targets'],
            'predicted_class_id': self.results['predictions'],
            'confidence': self.results['confidences'],
            'correct': self.results['targets'] == self.results['predictions'],
            'true_class_prob': [self.results['probabilities'][i, self.results['targets'][i]] 
                               for i in range(len(self.results['targets']))]
        })
        
        # Add top-k predictions
        for k in range(min(self.top_k, self.num_classes)):
            predictions_df[f'top_{k+1}_pred'] = [self.classnames[self.results['top_k_predictions'][i, k]] 
                                                 for i in range(len(self.results['predictions']))]
        
        predictions_df.to_csv(self.dirs['predictions'] / f'{self.save_prefix}_predictions.csv', index=False)
        
        # Create markdown report
        self._create_markdown_report(overall_metrics, per_class_metrics, error_analysis)
    
    def _create_markdown_report(self, overall_metrics, per_class_metrics, error_analysis):
        """Create a human-readable markdown report"""
        report_lines = [
            f"# CLIP-LoRA Species Classification Evaluation Report",
            f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"\n## Dataset Summary",
            f"- Total test samples: {len(self.results['predictions'])}",
            f"- Number of species classes: {self.num_classes}",
            f"\n## Overall Performance Metrics",
            f"- **Accuracy**: {overall_metrics['accuracy']:.4f}",
            f"- **Balanced Accuracy**: {overall_metrics['balanced_accuracy']:.4f}",
            f"- **F1 Score (Macro)**: {overall_metrics['f1_macro']:.4f}",
            f"- **F1 Score (Weighted)**: {overall_metrics['f1_weighted']:.4f}",
            f"- **Precision (Macro)**: {overall_metrics['precision_macro']:.4f}",
            f"- **Recall (Macro)**: {overall_metrics['recall_macro']:.4f}",
        ]
        
        # Add top-k accuracies
        for k in [1, 3, 5]:
            if f'top_{k}_accuracy' in overall_metrics:
                report_lines.append(f"- **Top-{k} Accuracy**: {overall_metrics[f'top_{k}_accuracy']:.4f}")
        
        # Confidence statistics
        report_lines.extend([
            f"\n### Confidence Statistics",
            f"- Mean: {overall_metrics['mean_confidence']:.4f}",
            f"- Std: {overall_metrics['std_confidence']:.4f}",
            f"- Range: [{overall_metrics['min_confidence']:.4f}, {overall_metrics['max_confidence']:.4f}]",
        ])
        
        # Top performing classes
        sorted_classes = sorted(per_class_metrics.items(), 
                               key=lambda x: x[1]['f1_score'], reverse=True)
        
        report_lines.extend([
            "\n## Top 10 Performing Species",
            "\n| Species | F1 Score | Precision | Recall | Support |",
            "|---------|----------|-----------|---------|---------|"
        ])
        
        for class_name, metrics in sorted_classes[:10]:
            report_lines.append(
                f"| {class_name} | {metrics['f1_score']:.3f} | {metrics['precision']:.3f} | "
                f"{metrics['recall']:.3f} | {metrics['support']} |"
            )
        
        # Bottom performing classes with support > 0
        bottom_classes = [c for c in sorted_classes if c[1]['support'] > 0][-10:]
        if bottom_classes:
            report_lines.extend([
                "\n## Bottom 10 Performing Species (with samples)",
                "\n| Species | F1 Score | Precision | Recall | Support |",
                "|---------|----------|-----------|---------|---------|"
            ])
            
            for class_name, metrics in bottom_classes:
                report_lines.append(
                    f"| {class_name} | {metrics['f1_score']:.3f} | {metrics['precision']:.3f} | "
                    f"{metrics['recall']:.3f} | {metrics['support']} |"
                )
        
        # Error analysis summary
        report_lines.extend([
            "\n## Error Analysis Summary",
            f"- Total errors: {error_analysis['total_errors']}",
            f"- Error rate: {error_analysis['error_rate']:.4f}",
        ])
        
        if 'error_confidence_stats' in error_analysis:
            report_lines.extend([
                f"- Mean confidence on errors: {error_analysis['error_confidence_stats']['mean']:.4f}",
                f"- Mean confidence on correct: {error_analysis['correct_confidence_stats']['mean']:.4f}",
            ])
        
        # Most confused pairs
        if 'most_confused_pairs' in error_analysis:
            report_lines.extend([
                "\n### Top 10 Most Confused Species Pairs",
                "\n| True Species | Predicted Species | Count |",
                "|--------------|-------------------|-------|"
            ])
            
            for pair in error_analysis['most_confused_pairs'][:10]:
                report_lines.append(
                    f"| {pair['true_class']} | {pair['predicted_class']} | {pair['count']} |"
                )
        
        # Save markdown report
        with open(self.output_dir / f'{self.save_prefix}_evaluation_report.md', 'w') as f:
            f.write('\n'.join(report_lines))


def evaluate_lora_comprehensive(args, clip_model, test_loader, dataset, save_path=None, 
                               save_prefix="eval", top_k=5):
    """
    Comprehensive evaluation function for CLIP-LoRA
    
    Args:
        args: Arguments object
        clip_model: CLIP model with LoRA
        test_loader: Test data loader
        dataset: Dataset object
        save_path: Path to save results
        save_prefix: Prefix for saved files
        top_k: Top-k accuracy to calculate
    
    Returns:
        float: Accuracy value
    """
    # Determine output directory
    if save_path is None:
        save_path = getattr(args, "save_path", None)
    
    if save_path is not None:
        backbone = args.backbone.replace('/', '').replace('-', '').lower()
        output_dir = f'{save_path}/{backbone}/{args.dataset}/{args.shots}shots/seed{args.seed}'
    else:
        output_dir = "clip_lora_evaluation"
    
    # Create evaluator
    evaluator = CLIPLoRAEvaluator(
        model=clip_model,
        dataset=dataset,
        test_loader=test_loader,
        output_dir=output_dir,
        device=args.device if hasattr(args, 'device') else 'cuda',
        top_k=top_k,
        save_prefix=save_prefix
    )
    
    # Run evaluation
    overall_metrics, per_class_metrics = evaluator.evaluate()
    
    # Print summary
    print(f"\nEvaluation complete! Results saved to: {output_dir}")
    print(f"\nOverall Metrics:")
    print(f"  - Accuracy: {overall_metrics['accuracy']:.4f}")
    print(f"  - Balanced Accuracy: {overall_metrics['balanced_accuracy']:.4f}")
    print(f"  - F1 (Macro): {overall_metrics['f1_macro']:.4f}")
    print(f"  - F1 (Weighted): {overall_metrics['f1_weighted']:.4f}")
    for k in [1, 3, 5]:
        if f'top_{k}_accuracy' in overall_metrics:
            print(f"  - Top-{k} Accuracy: {overall_metrics[f'top_{k}_accuracy']:.4f}")
    
    return overall_metrics['accuracy']
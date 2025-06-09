import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils import *

from loralib.utils import mark_only_lora_as_trainable, apply_lora, get_lora_parameters, lora_state_dict, save_lora, load_lora
from loralib import layers as lora_layers
from sklearn.metrics import classification_report, precision_recall_fscore_support, accuracy_score
import numpy as np
import json

def evaluate_lora(args, clip_model, loader, dataset):
    
    clip_model.eval()
    with torch.no_grad():
        template = dataset.template[0] 
        texts = [template.format(classname.replace('_', ' ')) for classname in dataset.classnames]
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            texts = clip.tokenize(texts).to(device)
            class_embeddings = clip_model.encode_text(texts)
        text_features = class_embeddings/class_embeddings.norm(dim=-1, keepdim=True)

    acc = 0.
    tot_samples = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for i, (images, target) in enumerate(loader):
            images, target = images.to(device), target.to(device)
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                image_features = clip_model.encode_image(images)
            image_features = image_features/image_features.norm(dim=-1, keepdim=True)
            cosine_similarity = image_features @ text_features.t()
            predictions = cosine_similarity.argmax(dim=1)
            
            # Calculate accuracy
            acc += cls_acc(cosine_similarity, target) * len(cosine_similarity)
            tot_samples += len(cosine_similarity)
            
            # Store predictions and targets for classification report
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    # Calculate overall accuracy
    acc /= tot_samples
    
    # Calculate aggregate metrics
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        all_targets, all_predictions, average='macro', zero_division=0
    )
    
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        all_targets, all_predictions, average='weighted', zero_division=0
    )
    
    # Calculate per-class metrics
    class_names = dataset.classnames
    per_class_precision, per_class_recall, per_class_f1, per_class_support = precision_recall_fscore_support(
        all_targets, all_predictions, labels=range(len(class_names)), zero_division=0
    )
    
    # Create per-species performance dictionary
    per_species_performance = {}
    for i, classname in enumerate(class_names):
        per_species_performance[classname] = {
            "precision": float(per_class_precision[i]),
            "recall": float(per_class_recall[i]),
            "f1_score": float(per_class_f1[i]),
            "support": int(per_class_support[i])
        }
    
    # Find top 5 and bottom 5 performing species based on F1 score
    species_f1_scores = [(classname, per_class_f1[i]) for i, classname in enumerate(class_names)]
    species_f1_scores.sort(key=lambda x: x[1], reverse=True)
    
    top5_species = species_f1_scores[:5]
    bottom5_species = species_f1_scores[-5:]
    
    top5_dict = {species: float(score) for species, score in top5_species}
    bottom5_dict = {species: float(score) for species, score in bottom5_species}
    
    # Save metrics to a JSON file
    metrics = {
        "accuracy": float(acc),
        "macro_precision": float(precision_macro),
        "macro_recall": float(recall_macro),
        "macro_f1": float(f1_macro),
        "weighted_precision": float(precision_weighted),
        "weighted_recall": float(recall_weighted),
        "weighted_f1": float(f1_weighted),
        "per_species_performance": per_species_performance,
        "top5_species_by_f1": top5_dict,
        "bottom5_species_by_f1": bottom5_dict
    }
    
    with open("/app/CLIP-LoRA-Species-Mapping/eval_results.json", "w") as f:
        json.dump(metrics, f, indent=4)

    # Print metrics
    print(f"\nEvaluation Metrics:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro Precision: {precision_macro:.4f}")
    print(f"Macro Recall: {recall_macro:.4f}")
    print(f"Macro F1: {f1_macro:.4f}")
    print(f"Weighted Precision: {precision_weighted:.4f}")
    print(f"Weighted Recall: {recall_weighted:.4f}")
    print(f"Weighted F1: {f1_weighted:.4f}")
    
    # Print top 5 and bottom 5 species by F1 score
    print("\nTop 5 species by F1 score:")
    for species, score in top5_species:
        print(f"{species}: {score:.4f}")
    
    print("\nBottom 5 species by F1 score:")
    for species, score in bottom5_species:
        print(f"{species}: {score:.4f}")
    
    # Print classification report for first few and last few classes (to avoid long output)
    if len(class_names) > 10:
        # For datasets with many classes, print a subset
        report = classification_report(
            all_targets, 
            all_predictions, 
            target_names=class_names[:5] + ["..."] + class_names[-5:], 
            labels=list(range(5)) + list(range(len(class_names)-5, len(class_names))),
            digits=4
        )
    else:
        # For datasets with few classes, print all
        report = classification_report(
            all_targets, 
            all_predictions, 
            target_names=class_names, 
            digits=4
        )
    
    print("\nPer-class Metrics (sample):")
    print(report)
    
    return acc

def evaluate_lora_full(args, clip_model, loader, dataset, save_path=None, save_prefix="eval", top_k=5):
    """
    Comprehensive evaluation for CLIP-LoRA species classification.
    Saves metrics, per-class results, error analysis, and confusion matrices in the same folder as the weights (.pth).
    """
    import os
    import numpy as np
    import pandas as pd
    from pathlib import Path
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score,
        confusion_matrix, classification_report, top_k_accuracy_score
    )
    import json

    # --- Determine the correct output directory for saving ---
    # Use the same logic as save_lora in loralib/utils.py
    if save_path is None:
        save_path = getattr(args, "save_path", None)
    if save_path is not None:
        backbone = args.backbone.replace('/', '').replace('-', '').lower()
        save_dir = f'{save_path}/{backbone}/{args.dataset}/{args.shots}shots/seed{args.seed}'
        output_dir = Path(save_dir)
    else:
        output_dir = Path(".")
    output_dir.mkdir(exist_ok=True, parents=True)

    # Prepare for evaluation
    clip_model.eval()
    device = next(clip_model.parameters()).device
    classnames = dataset.classnames
    template = dataset.template[0]
    texts = [template.format(classname.replace('_', ' ')) for classname in classnames]
    with torch.no_grad(), torch.amp.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.float16):
        texts = clip.tokenize(texts).to(device)
        class_embeddings = clip_model.encode_text(texts)
    text_features = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)

    all_targets = []
    all_predictions = []
    all_probs = []
    all_filenames = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            # Support both dict and tuple batch
            if isinstance(batch, dict):
                images = batch['image'].to(device)
                targets = batch.get('label', batch.get('species_index', None))
                if targets is None:
                    raise ValueError("Batch dict must contain 'label' or 'species_index'.")
                targets = targets.to(device)
                filenames = batch.get('filename', [""] * len(images))
            else:
                images, targets = batch
                images, targets = images.to(device), targets.to(device)
                filenames = [""] * len(images)

            with torch.amp.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.float16):
                image_features = clip_model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            logits = image_features @ text_features.t()
            probs = logits.softmax(dim=1)
            preds = logits.argmax(dim=1)

            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_filenames.extend(filenames)

    all_targets = np.array(all_targets)
    all_predictions = np.array(all_predictions)
    all_probs = np.array(all_probs)
    all_filenames = np.array(all_filenames)

    # Metrics
    metrics = {}
    metrics['accuracy'] = accuracy_score(all_targets, all_predictions)
    metrics['f1_macro'] = f1_score(all_targets, all_predictions, average='macro')
    metrics['f1_weighted'] = f1_score(all_targets, all_predictions, average='weighted')
    metrics['precision_macro'] = precision_score(all_targets, all_predictions, average='macro', zero_division=0)
    metrics['recall_macro'] = recall_score(all_targets, all_predictions, average='macro', zero_division=0)
    metrics['precision_weighted'] = precision_score(all_targets, all_predictions, average='weighted', zero_division=0)
    metrics['recall_weighted'] = recall_score(all_targets, all_predictions, average='weighted', zero_division=0)
    metrics['balanced_accuracy'] = recall_score(all_targets, all_predictions, average='macro', zero_division=0)
    if top_k > 1:
        try:
            metrics[f'top_{top_k}_accuracy'] = top_k_accuracy_score(all_targets, all_probs, k=top_k)
        except Exception:
            metrics[f'top_{top_k}_accuracy'] = 0.0

    # Per-class metrics
    per_class_report = classification_report(
        all_targets, 
        all_predictions, 
        labels=list(range(len(classnames))),
        target_names=classnames, 
        output_dict=True, 
        zero_division=0
    )
    per_class_df = pd.DataFrame(per_class_report).T
    per_class_df.to_csv(output_dir / f"{save_prefix}_per_class_metrics.csv")

    # Confusion matrix
    cm = confusion_matrix(all_targets, all_predictions)
    cm_df = pd.DataFrame(cm, index=classnames, columns=classnames)
    cm_df.to_csv(output_dir / f"{save_prefix}_confusion_matrix.csv")

    # Save main metrics
    with open(output_dir / f"{save_prefix}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Save predictions
    pred_df = pd.DataFrame({
        "filename": all_filenames,
        "true_label": [classnames[i] for i in all_targets],
        "pred_label": [classnames[i] for i in all_predictions],
        "true_label_id": all_targets,
        "pred_label_id": all_predictions,
        "correct": all_targets == all_predictions,
        "confidence": all_probs.max(axis=1)
    })
    pred_df.to_csv(output_dir / f"{save_prefix}_predictions.csv", index=False)

    # Error analysis: misclassified samples
    misclassified = pred_df[~pred_df["correct"]]
    misclassified.to_csv(output_dir / f"{save_prefix}_misclassified.csv", index=False)

    # Print summary
    print(f"\nEvaluation complete. Results saved to {output_dir}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['f1_macro']:.4f}")
    print(f"Weighted F1: {metrics['f1_weighted']:.4f}")
    if f'top_{top_k}_accuracy' in metrics:
        print(f"Top-{top_k} Accuracy: {metrics[f'top_{top_k}_accuracy']:.4f}")

    # Return metrics for further use
    return metrics['accuracy']


def run_lora(args, clip_model, logit_scale, dataset, train_loader, val_loader, test_loader):
    
    VALIDATION = True  # Enable validation by default
    
    # Textual features
    print("\nGetting textual features as CLIP's classifier.")
    textual_features = clip_classifier(dataset.classnames, dataset.template, clip_model)

    # Pre-load val features
    print("\nLoading visual features and labels from val set.")
    val_features, val_labels = pre_load_features(clip_model, val_loader)

    # Pre-load test features
    print("\nLoading visual features and labels from test set.")
    test_features, test_labels = pre_load_features(clip_model, test_loader)
    
    test_features = test_features.to(device)
    test_labels = test_labels.to(device)
 
    # Zero-shot CLIP
    clip_logits = logit_scale * test_features @ textual_features
    zs_acc = cls_acc(clip_logits, test_labels)
    print("\n**** Zero-shot CLIP's test accuracy: {:.2f}. ****\n".format(zs_acc))
    
    test_features = test_features.cpu()
    test_labels = test_labels.cpu()
    
    
    list_lora_layers = apply_lora(args, clip_model)
    clip_model = clip_model.to(device) 
    
    if args.eval_only:
        load_lora(args, list_lora_layers)
        acc_test = evaluate_lora(args, clip_model, test_loader, dataset)
        print("**** Test accuracy: {:.2f}. ****\n".format(acc_test))
        return

    mark_only_lora_as_trainable(clip_model)
    
    # Use n_iters as the number of epochs for full dataset mode
    num_epochs = args.n_iters
    
    # Calculate total iterations based on epochs for scheduler (epochs * batches per epoch)
    total_steps = num_epochs * len(train_loader)
    
    optimizer = torch.optim.AdamW(get_lora_parameters(clip_model), weight_decay=1e-2, betas=(0.9, 0.999), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps, eta_min=1e-6)
    
    best_acc_val, best_acc_test = 0., 0.
    best_epoch_val = 0
    
    # training LoRA
    scaler = torch.cuda.amp.GradScaler()
    
    for epoch in range(num_epochs):
        clip_model.train()
        acc_train = 0
        tot_samples = 0
        loss_epoch = 0.
        
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        if args.encoder == 'vision': 
            text_features = textual_features.t().half()
            
        # Use tqdm for progress bar during training
        for i, (images, target) in enumerate(tqdm(train_loader, desc=f"Training epoch {epoch+1}")):
            
            template = dataset.template[0]
            texts = [template.format(classname.replace('_', ' ')) for classname in dataset.classnames]
            images, target = images.to(device), target.to(device)
            
            if args.encoder == 'text' or args.encoder == 'both':
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    texts = clip.tokenize(texts).to(device)
                    class_embeddings = clip_model.encode_text(texts)
                text_features = class_embeddings/class_embeddings.norm(dim=-1, keepdim=True)
                
            if args.encoder == 'vision' or args.encoder == 'both':
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    image_features = clip_model.encode_image(images)
            else:
                with torch.no_grad():
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                        image_features = clip_model.encode_image(images)
                        
            image_features = image_features/image_features.norm(dim=-1, keepdim=True)
            
            cosine_similarity = logit_scale * image_features @ text_features.t()
            loss = F.cross_entropy(cosine_similarity, target)
            acc_train += cls_acc(cosine_similarity, target) * target.shape[0]
            loss_epoch += loss.item() * target.shape[0]
            tot_samples += target.shape[0]
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
        # Epoch complete - report metrics
        acc_train /= tot_samples
        loss_epoch /= tot_samples
        current_lr = scheduler.get_last_lr()[0]
        print('Epoch: {}/{}, LR: {:.6f}, Train Acc: {:.4f}, Loss: {:.4f}'.format(
            epoch+1, num_epochs, current_lr, acc_train, loss_epoch))
        
        # Validation after each epoch
        if VALIDATION:
            clip_model.eval()
            acc_val = evaluate_lora(args, clip_model, val_loader, dataset)
            print("**** Epoch {}/{} - Validation accuracy: {:.2f}. ****".format(epoch+1, num_epochs, acc_val))
            
            # Save best model based on validation accuracy
            if acc_val > best_acc_val:
                best_acc_val = acc_val
                best_epoch_val = epoch + 1
                
                # Evaluate on test set
                acc_test = evaluate_lora(args, clip_model, test_loader, dataset)
                best_acc_test = acc_test
                print("**** New best model! Test accuracy: {:.2f}. ****".format(acc_test))
                
                # Save best model
                if args.save_path is not None:
                    save_path = f"{args.save_path}_best"
                    save_lora(args, list_lora_layers, clip_model)
                    print(f"Best model saved to {save_path}")

    
    # Final evaluation on test set
    clip_model.eval()
    final_acc_test = evaluate_lora_full(args, clip_model, test_loader, dataset)
    print("\n**** Training complete ****")
    print("**** Best validation accuracy: {:.2f} (epoch {}) ****".format(best_acc_val, best_epoch_val))
    print("**** Best model test accuracy: {:.2f} ****".format(best_acc_test))
    print("**** Final model test accuracy: {:.2f} ****".format(final_acc_test))
    
    # Save final model
    if args.save_path is not None:
        save_lora(args, list_lora_layers, clip_model)
        print(f"Final model saved to {args.save_path}")
    
    return
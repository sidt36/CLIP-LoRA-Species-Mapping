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
    
    # Print detailed metrics
    # Save metrics to a JSON file
    metrics = {
        "accuracy": acc,
        "macro_precision": precision_macro,
        "macro_recall": recall_macro,
        "macro_f1": f1_macro,
        "weighted_precision": precision_weighted,
        "weighted_recall": recall_weighted,
        "weighted_f1": f1_weighted
    }
    with open("/app/CLIP-LoRA-Species-Mapping.json", "w") as f:
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
    
    # Print classification report for first few and last few classes (to avoid long output)
    class_names = dataset.classnames
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
    final_acc_test = evaluate_lora(args, clip_model, test_loader, dataset)
    print("\n**** Training complete ****")
    print("**** Best validation accuracy: {:.2f} (epoch {}) ****".format(best_acc_val, best_epoch_val))
    print("**** Best model test accuracy: {:.2f} ****".format(best_acc_test))
    print("**** Final model test accuracy: {:.2f} ****".format(final_acc_test))
    
    # Save final model
    if args.save_path is not None:
        save_lora(args, list_lora_layers, clip_model)
        print(f"Final model saved to {args.save_path}")
    
    return
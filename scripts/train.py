#!/usr/bin/env python3
"""Training script for interpretable computer vision models."""

import hydra
from omegaconf import DictConfig
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import wandb
from pathlib import Path

from interpretable_cv_xai.models import create_model
from interpretable_cv_xai.data import CIFAR10DataModule
from interpretable_cv_xai.utils import set_seed, get_device, save_checkpoint, load_checkpoint


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train model for one epoch.
    
    Args:
        model: PyTorch model.
        dataloader: Training data loader.
        optimizer: Optimizer.
        criterion: Loss function.
        device: Device to run on.
        
    Returns:
        Average loss and accuracy for the epoch.
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(tqdm(dataloader, desc="Training")):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy


def validate_epoch(model, dataloader, criterion, device):
    """Validate model for one epoch.
    
    Args:
        model: PyTorch model.
        dataloader: Validation data loader.
        criterion: Loss function.
        device: Device to run on.
        
    Returns:
        Average loss and accuracy for the epoch.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in tqdm(dataloader, desc="Validation"):
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training function."""
    
    # Set random seed for reproducibility
    set_seed(cfg.seed)
    
    # Get device
    if cfg.device == "auto":
        device = get_device()
    else:
        device = torch.device(cfg.device)
    
    print(f"Using device: {device}")
    
    # Initialize logging
    if cfg.logging.use_wandb:
        wandb.init(
            project=cfg.logging.wandb_project,
            name=cfg.experiment_name,
            config=cfg
        )
    
    # Create data module
    data_module = hydra.utils.instantiate(cfg.data)
    
    # Create model
    model = hydra.utils.instantiate(cfg.model)
    model = model.to(device)
    
    print(f"Model: {cfg.model.name}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Create optimizer and scheduler
    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay
    )
    
    if cfg.training.scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.training.epochs
        )
    else:
        scheduler = None
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    best_val_acc = 0.0
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    print(f"\nStarting training for {cfg.training.epochs} epochs...")
    
    for epoch in range(cfg.training.epochs):
        print(f"\nEpoch {epoch + 1}/{cfg.training.epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, data_module.train_dataloader(), optimizer, criterion, device
        )
        
        # Validate
        val_loss, val_acc = validate_epoch(
            model, data_module.val_dataloader(), criterion, device
        )
        
        # Update scheduler
        if scheduler is not None:
            scheduler.step()
        
        # Store metrics
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        # Log metrics
        if cfg.logging.use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "learning_rate": optimizer.param_groups[0]['lr']
            })
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = Path("checkpoints") / f"{cfg.experiment_name}_best.pth"
            checkpoint_path.parent.mkdir(exist_ok=True)
            save_checkpoint(
                model, optimizer, epoch, val_loss, str(checkpoint_path),
                val_accuracy=val_acc
            )
            print(f"New best model saved with validation accuracy: {val_acc:.2f}%")
    
    # Final evaluation on test set
    print("\nEvaluating on test set...")
    test_loss, test_acc = validate_epoch(
        model, data_module.test_dataloader(), criterion, device
    )
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
    
    # Save final model
    final_checkpoint_path = Path("checkpoints") / f"{cfg.experiment_name}_final.pth"
    save_checkpoint(
        model, optimizer, cfg.training.epochs - 1, test_loss, str(final_checkpoint_path),
        test_accuracy=test_acc
    )
    
    # Save training history
    history = {
        "train_losses": train_losses,
        "train_accuracies": train_accuracies,
        "val_losses": val_losses,
        "val_accuracies": val_accuracies,
        "test_accuracy": test_acc,
        "best_val_accuracy": best_val_acc
    }
    
    import json
    history_path = Path("checkpoints") / f"{cfg.experiment_name}_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\nTraining completed!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Test accuracy: {test_acc:.2f}%")
    print(f"Models saved to: checkpoints/")
    
    if cfg.logging.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()

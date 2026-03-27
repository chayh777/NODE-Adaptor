import torch
import numpy as np
from typing import Dict, List


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Calculate top-1 accuracy."""
    preds = torch.argmax(logits, dim=-1)
    correct = (preds == labels).sum().item()
    total = labels.size(0)
    return correct / total


def top_k_accuracy(logits: torch.Tensor, labels: torch.Tensor, k: int = 5) -> float:
    """Calculate top-k accuracy."""
    _, top_k_preds = torch.topk(logits, k, dim=-1)
    labels_expanded = labels.view(-1, 1).expand_as(top_k_preds)
    correct = (top_k_preds == labels_expanded).any(dim=-1).sum().item()
    total = labels.size(0)
    return correct / total


def calculate_task_accuracy(logits: torch.Tensor, labels: torch.Tensor, num_ways: int) -> Dict[str, float]:
    """Calculate per-task metrics."""
    task_acc = accuracy(logits, labels)
    task_acc_top5 = top_k_accuracy(logits, labels, k=min(5, num_ways))
    
    return {
        'accuracy': task_acc,
        'accuracy_top5': task_acc_top5
    }


def evaluate_model(
    model: torch.nn.Module,
    clip_model,
    dataloader: torch.utils.data.DataLoader,
    class_names: List[str],
    prompts: List[str],
    device: str = 'cuda',
    num_eval_tasks: int = 1000
) -> Dict[str, float]:
    """Evaluate model on few-shot tasks."""
    model.eval()
    
    all_accuracies = []
    
    with torch.no_grad():
        for task_idx, task in enumerate(dataloader):
            if task_idx >= num_eval_tasks:
                break
            
            support_images = task['support_images'].to(device)
            query_images = task['query_images'].to(device)
            query_labels = task['query_labels'].to(device)
            
            selected_classes = task['selected_classes']
            selected_class_names = [class_names[i] for i in selected_classes]
            
            logits = model(
                clip_model=clip_model,
                support_images=support_images,
                query_images=query_images,
                class_names=selected_class_names,
                prompts=prompts
            )
            
            task_acc = accuracy(logits, query_labels)
            all_accuracies.append(task_acc)
    
    mean_accuracy = np.mean(all_accuracies)
    std_accuracy = np.std(all_accuracies)
    
    return {
        'mean_accuracy': mean_accuracy,
        'std_accuracy': std_accuracy,
        'num_tasks': len(all_accuracies)
    }


def compute_confusion_matrix(logits: torch.Tensor, labels: torch.Tensor, num_classes: int) -> np.ndarray:
    """Compute confusion matrix."""
    preds = torch.argmax(logits, dim=-1)
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    
    for pred, label in zip(preds, labels):
        cm[label.item()][pred.item()] += 1
    
    return cm


def per_class_accuracy(logits: torch.Tensor, labels: torch.Tensor, num_classes: int) -> Dict[int, float]:
    """Compute per-class accuracy."""
    preds = torch.argmax(logits, dim=-1)
    class_correct = torch.zeros(num_classes)
    class_total = torch.zeros(num_classes)
    
    for pred, label in zip(preds, labels):
        class_total[label.item()] += 1
        if pred == label:
            class_correct[label.item()] += 1
    
    per_class_acc = {}
    for i in range(num_classes):
        if class_total[i] > 0:
            per_class_acc[i] = (class_correct[i] / class_total[i]).item()
        else:
            per_class_acc[i] = 0.0
    
    return per_class_acc

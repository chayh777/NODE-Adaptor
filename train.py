import os
import sys
import argparse
import yaml
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import clip

from models import NODEAdapter
from data.prompts import CIFAR100_CLASSES, CIFAR100_PROMPTS, get_cifar100_dataset
from data_loader import get_clip_transforms, create_fewshot_dataloader
from utils import accuracy


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description='NODE-Adapter Training')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')
    parser.add_argument('--num_ways', type=int, default=5,
                        help='Number of ways (classes) per task')
    parser.add_argument('--num_shots', type=int, default=5,
                        help='Number of shots (support images) per class')
    parser.add_argument('--num_query', type=int, default=15,
                        help='Number of query images per class')
    parser.add_argument('--num_tasks', type=int, default=100,
                        help='Number of tasks per epoch')
    parser.add_argument('--num_eval_tasks', type=int, default=1000,
                        help='Number of tasks for evaluation')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                        help='Weight decay')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--data_root', type=str, default='./data',
                        help='Data root directory')
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def train_one_epoch(
    model: NODEAdapter,
    clip_model,
    dataloader,
    class_names: list,
    prompts: list,
    optimizer: optim.Optimizer,
    device: str = 'cuda',
    num_ways: int = 5,
    num_shots: int = 5
) -> dict:
    """Train for one epoch."""
    model.train()
    
    total_loss = 0
    total_acc = 0
    num_tasks = 0
    
    pbar = tqdm(dataloader, desc='Training')
    
    for task in pbar:
        support_images = task['support_images'].to(device)
        support_labels = task['support_labels'].to(device)
        query_images = task['query_images'].to(device)
        query_labels = task['query_labels'].to(device)
        
        selected_classes = task['selected_classes']
        selected_class_names = [class_names[i] for i in selected_classes]
        
        task_num_ways = task.get('num_ways', num_ways)
        task_num_shots = task.get('num_shots', num_shots)
        
        logits = model(
            clip_model=clip_model,
            support_images=support_images,
            query_images=query_images,
            class_names=selected_class_names,
            prompts=prompts,
            num_ways=task_num_ways,
            num_shots=task_num_shots
        )
        
        loss = nn.CrossEntropyLoss()(logits, query_labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        preds = torch.argmax(logits, dim=-1)
        acc = (preds == query_labels).float().mean().item()
        
        total_loss += loss.item()
        total_acc += acc
        num_tasks += 1
        
        pbar.set_postfix({
            'loss': f'{total_loss / num_tasks:.4f}',
            'acc': f'{total_acc / num_tasks:.4f}'
        })
    
    return {
        'loss': total_loss / num_tasks,
        'accuracy': total_acc / num_tasks
    }


def evaluate(
    model: NODEAdapter,
    clip_model,
    dataloader,
    class_names: list,
    prompts: list,
    device: str = 'cuda',
    num_eval_tasks: int = 1000
) -> dict:
    """Evaluate model."""
    model.eval()
    
    all_accuracies = []
    
    with torch.no_grad():
        for task_idx, task in enumerate(tqdm(dataloader, desc='Evaluating')):
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
            
            preds = torch.argmax(logits, dim=-1)
            acc = (preds == query_labels).float().mean().item()
            all_accuracies.append(acc)
    
    mean_acc = np.mean(all_accuracies)
    std_acc = np.std(all_accuracies)
    
    return {
        'mean_accuracy': mean_acc,
        'std_accuracy': std_acc
    }


def main():
    args = parse_args()
    
    config = load_config(args.config)
    
    set_seed(args.seed)
    
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        print(f'GPU: {torch.cuda.get_device_name(0)}')
        print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB')
    else:
        torch.set_num_threads(4)
    
    print('Loading CLIP model...')
    clip_model, preprocess = clip.load(config['model']['clip_model'], device=device)
    clip_model.eval()
    
    print('Loading CIFAR-100 dataset...')
    train_dataset = get_cifar100_dataset(root=args.data_root, train=True, transform=preprocess)
    test_dataset = get_cifar100_dataset(root=args.data_root, train=False, transform=preprocess)
    
    print(f'Train dataset: {len(train_dataset)} images')
    print(f'Test dataset: {len(test_dataset)} images')
    
    class_names = CIFAR100_CLASSES
    prompts = CIFAR100_PROMPTS
    
    print(f'Creating NODE-Adapter...')
    model = NODEAdapter(
        num_classes=args.num_ways,
        embed_dim=config['model']['embed_dim'],
        num_heads=config['model']['num_heads'],
        ode_steps=config['model']['ode_steps'],
        temperature=config['model']['temperature']
    ).to(device)
    
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    print(f'\n=== {args.num_ways}-way {args.num_shots}-shot Training ===')
    
    for epoch in range(args.epochs):
        print(f'\nEpoch {epoch + 1}/{args.epochs}')
        
        train_dataloader = create_fewshot_dataloader(
            dataset=train_dataset,
            num_ways=args.num_ways,
            num_shots=args.num_shots,
            num_query=args.num_query,
            num_tasks=args.num_tasks,
            batch_size=config['training']['batch_size']
        )
        
        train_metrics = train_one_epoch(
            model=model,
            clip_model=clip_model,
            dataloader=train_dataloader,
            class_names=class_names,
            prompts=prompts,
            optimizer=optimizer,
            device=device,
            num_ways=args.num_ways,
            num_shots=args.num_shots
        )
        
        print(f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.4f}")
    
    print('\n=== Evaluation ===')
    
    test_dataloader = create_fewshot_dataloader(
        dataset=test_dataset,
        num_ways=args.num_ways,
        num_shots=args.num_shots,
        num_query=args.num_query,
        num_tasks=args.num_eval_tasks,
        batch_size=1
    )
    
    eval_metrics = evaluate(
        model=model,
        clip_model=clip_model,
        dataloader=test_dataloader,
        class_names=class_names,
        prompts=prompts,
        device=device,
        num_eval_tasks=args.num_eval_tasks
    )
    
    print(f"\nEvaluation Results:")
    print(f"Mean Accuracy: {eval_metrics['mean_accuracy']:.4f} ± {eval_metrics['std_accuracy']:.4f}")


if __name__ == '__main__':
    main()

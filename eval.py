import os
import argparse
import yaml
import random
import numpy as np
import torch
from tqdm import tqdm
import clip

from models import (
    NODEAdapter, 
    CoOp, CoOpSimple,
    TipAdapterF, TipAdapterFSimple, 
    ProGrad, ProGradSimple,
    APE, APESimple
)
from data.prompts import CIFAR100_CLASSES, CIFAR100_PROMPTS, get_cifar100_dataset
from data_loader import get_clip_transforms, create_fewshot_dataloader
from utils import accuracy
from utils.visualization import ExperimentLogger, Visualizer, compute_confidence


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description='NODE-Adapter Evaluation')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')
    parser.add_argument('--model', type=str, default='node_adapter',
                        choices=['node_adapter', 'coop', 'tip_adapter', 'prograd', 'ape', 'clip'],
                        help='Model to evaluate')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to saved model')
    parser.add_argument('--num_ways', type=int, default=20,
                        help='Number of ways (classes) per task')
    parser.add_argument('--num_shots', type=int, default=5,
                        help='Number of shots (support images) per class')
    parser.add_argument('--num_query', type=int, default=15,
                        help='Number of query images per class')
    parser.add_argument('--num_eval_tasks', type=int, default=100,
                        help='Number of tasks for evaluation')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--data_root', type=str, default='./data',
                        help='Data root directory')
    parser.add_argument('--train', action='store_true',
                        help='Evaluate on train set')
    parser.add_argument('--multi_shot', action='store_true',
                        help='Run multi-shot experiments (1/2/5/16)')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Experiment name for logging')
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_model(model_name: str, num_classes: int, config: dict, device: str):
    """
    Load model by name.
    
    Args:
        model_name: name of the model
        num_classes: number of classes
        config: configuration dict
        device: device to load model to
        
    Returns:
        model: the loaded model
    """
    if model_name == 'node_adapter':
        model = NODEAdapter(
            num_classes=num_classes,
            embed_dim=config['model']['embed_dim'],
            num_heads=config['model']['num_heads'],
            ode_steps=config['model']['ode_steps'],
            temperature=config['model']['temperature']
        ).to(device)
        
    elif model_name == 'coop':
        model = CoOpSimple(
            num_classes=num_classes,
            embed_dim=config['model']['embed_dim'],
            temperature=config['model']['temperature']
        ).to(device)
        
    elif model_name == 'tip_adapter':
        model = TipAdapterFSimple(
            num_classes=num_classes,
            embed_dim=config['model']['embed_dim'],
            alpha=1.0,
            beta=2.0,
            temperature=config['model']['temperature']
        ).to(device)
        
    elif model_name == 'prograd':
        model = ProGradSimple(
            num_classes=num_classes,
            embed_dim=config['model']['embed_dim'],
            temperature=config['model']['temperature']
        ).to(device)
        
    elif model_name == 'ape':
        model = APESimple(
            num_classes=num_classes,
            embed_dim=config['model']['embed_dim'],
            num_experts=4,
            temperature=config['model']['temperature']
        ).to(device)
        
    elif model_name == 'clip':
        model = None
        
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model


def evaluate(
    model,
    clip_model,
    dataloader,
    class_names: list,
    prompts: list,
    device: str = 'cuda',
    num_eval_tasks: int = 1000,
    collect_confidence: bool = False,
    model_name: str = 'node_adapter'
) -> dict:
    """Evaluate model on few-shot tasks."""
    if model is not None:
        model.eval()
    
    all_accuracies = []
    all_confidences = []
    
    with torch.no_grad():
        for task_idx, task in enumerate(tqdm(dataloader, desc='Evaluating')):
            if task_idx >= num_eval_tasks:
                break
            
            support_images = task['support_images'].to(device)
            query_images = task['query_images'].to(device)
            query_labels = task['query_labels'].to(device)
            
            selected_classes = task['selected_classes']
            selected_class_names = [class_names[i] for i in selected_classes]
            
            if model_name == 'clip':
                logits = evaluate_clip_zero_shot(
                    clip_model, query_images, selected_class_names, prompts, device
                )
            else:
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
            
            if collect_confidence:
                confidences = compute_confidence(logits)
                all_confidences.extend(confidences.tolist())
    
    mean_acc = np.mean(all_accuracies)
    std_acc = np.std(all_accuracies)
    
    result = {
        'mean_accuracy': float(mean_acc),
        'std_accuracy': float(std_acc),
        'num_tasks': len(all_accuracies)
    }
    
    if collect_confidence and all_confidences:
        result['mean_confidence'] = float(np.mean(all_confidences))
        result['confidence_distribution'] = all_confidences
    
    return result


def evaluate_clip_zero_shot(clip_model, query_images, class_names, prompts, device):
    """Evaluate using CLIP zero-shot classification."""
    text_features_list = []
    for class_name in class_names:
        class_prompts = [p.format(class_name) for p in prompts]
        text_tokens = clip.tokenize(class_prompts).to(device)
        text_features = clip_model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        text_features_list.append(text_features.mean(dim=0))
    
    text_features = torch.stack(text_features_list, dim=0)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    image_features = clip_model.encode_image(query_images)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    
    logits = image_features @ text_features.T / 0.07
    
    return logits


def run_single_shot_experiment(
    model,
    clip_model,
    dataset,
    class_names: list,
    prompts: list,
    device: str,
    num_ways: int,
    num_shots: int,
    num_eval_tasks: int,
    num_query: int,
    logger: ExperimentLogger,
    collect_confidence: bool = False,
    model_name: str = 'node_adapter'
) -> dict:
    """Run experiment for a single shot setting."""
    print(f"\n{'='*50}")
    print(f"Running {num_ways}-way {num_shots}-shot experiment with {model_name}")
    print(f"{'='*50}")
    
    dataloader = create_fewshot_dataloader(
        dataset=dataset,
        num_ways=num_ways,
        num_shots=num_shots,
        num_query=num_query,
        num_tasks=num_eval_tasks,
        batch_size=1
    )
    
    metrics = evaluate(
        model=model,
        clip_model=clip_model,
        dataloader=dataloader,
        class_names=class_names,
        prompts=prompts,
        device=device,
        num_eval_tasks=num_eval_tasks,
        collect_confidence=collect_confidence,
        model_name=model_name
    )
    
    print(f"\n{num_ways}-way {num_shots}-shot Results:")
    print(f"  Mean Accuracy: {metrics['mean_accuracy']:.4f} ± {metrics['std_accuracy']:.4f}")
    if 'mean_confidence' in metrics:
        print(f"  Mean Confidence: {metrics['mean_confidence']:.4f}")
    
    logger.log_per_shot_result(num_shots, metrics)
    
    return metrics


def main():
    args = parse_args()
    
    config = load_config(args.config)
    
    set_seed(args.seed)
    
    experiment_name = args.experiment_name or f"{args.model}_{args.num_ways}way_{args.num_shots}shot"
    logger = ExperimentLogger(experiment_name=experiment_name)
    logger.log_config({
        'model': args.model,
        'num_ways': args.num_ways,
        'num_shots': args.num_shots,
        'num_query': args.num_query,
        'num_eval_tasks': args.num_eval_tasks,
        'ode_steps': config['model']['ode_steps'],
        'num_heads': config['model']['num_heads'],
        'temperature': config['model']['temperature'],
        'clip_model': config['model']['clip_model']
    })
    
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
    if args.train:
        dataset = get_cifar100_dataset(root=args.data_root, train=True, transform=preprocess)
    else:
        dataset = get_cifar100_dataset(root=args.data_root, train=False, transform=preprocess)
    print(f'Dataset: {len(dataset)} images')
    
    class_names = CIFAR100_CLASSES
    prompts = CIFAR100_PROMPTS
    
    print(f'Creating {args.model} model...')
    model = load_model(args.model, args.num_ways, config, device)
    
    if args.model_path and os.path.exists(args.model_path) and model is not None:
        print(f'Loading model from {args.model_path}...')
        model.load_state_dict(torch.load(args.model_path))
    
    if args.multi_shot:
        shot_values = [1, 2, 5, 16]
        print(f"\n{'#'*60}")
        print(f"# Running Multi-Shot Experiments ({args.num_ways}-way) with {args.model}")
        print(f"{'#'*60}")
        
        for shot in shot_values:
            run_single_shot_experiment(
                model=model,
                clip_model=clip_model,
                dataset=dataset,
                class_names=class_names,
                prompts=prompts,
                device=device,
                num_ways=args.num_ways,
                num_shots=shot,
                num_eval_tasks=args.num_eval_tasks,
                num_query=args.num_query,
                logger=logger,
                collect_confidence=(shot == 5),
                model_name=args.model
            )
        
        visualizer = Visualizer(logger)
        visualizer.save_all_plots()
        
    else:
        dataloader = create_fewshot_dataloader(
            dataset=dataset,
            num_ways=args.num_ways,
            num_shots=args.num_shots,
            num_query=args.num_query,
            num_tasks=args.num_eval_tasks,
            batch_size=1
        )
        
        eval_metrics = evaluate(
            model=model,
            clip_model=clip_model,
            dataloader=dataloader,
            class_names=class_names,
            prompts=prompts,
            device=device,
            num_eval_tasks=args.num_eval_tasks,
            collect_confidence=True,
            model_name=args.model
        )
        
        print(f"\n{'='*50}")
        print(f"Evaluation Results ({args.num_ways}-way {args.num_shots}-shot)")
        print(f"{'='*50}")
        print(f"Mean Accuracy: {eval_metrics['mean_accuracy']:.4f} ± {eval_metrics['std_accuracy']:.4f}")
        print(f"Number of tasks: {eval_metrics['num_tasks']}")
        if 'mean_confidence' in eval_metrics:
            print(f"Mean Confidence: {eval_metrics['mean_confidence']:.4f}")
        
        logger.log_test_result(eval_metrics)
        
        visualizer = Visualizer(logger)
        
        if 'confidence_distribution' in eval_metrics:
            visualizer.plot_confidence_distribution(
                eval_metrics['confidence_distribution']
            )
        
        visualizer.generate_summary_report()
    
    logger.save_results()
    print(f"\nExperiment completed! Results saved to ./results/")


if __name__ == '__main__':
    main()

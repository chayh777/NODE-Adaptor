import os
import json
import datetime
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
from pathlib import Path

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12


class ExperimentLogger:
    """Experiment logger for storing metrics and results."""
    
    def __init__(self, save_dir: str = "./results", experiment_name: str = None):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        if experiment_name is None:
            experiment_name = f"exp_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.experiment_name = experiment_name
        
        self.plots_dir = self.save_dir / "plots" / experiment_name
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
        self.data_dir = self.save_dir / "data"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.experiment_data = {
            'experiment_name': experiment_name,
            'timestamp': datetime.datetime.now().isoformat(),
            'config': {},
            'train_metrics': [],
            'test_metrics': [],
            'per_shot_results': {},
            'task_results': []
        }
    
    def log_config(self, config: dict):
        """Log experiment configuration."""
        self.experiment_data['config'] = config
    
    def log_train_epoch(self, epoch: int, metrics: dict):
        """Log training epoch metrics."""
        self.experiment_data['train_metrics'].append({
            'epoch': epoch,
            **metrics
        })
    
    def log_test_result(self, metrics: dict):
        """Log test/evaluation metrics."""
        self.experiment_data['test_metrics'].append(metrics)
    
    def log_per_shot_result(self, shot: int, metrics: dict):
        """Log per-shot results."""
        key = f"{shot}shot"
        if key not in self.experiment_data['per_shot_results']:
            self.experiment_data['per_shot_results'][key] = []
        self.experiment_data['per_shot_results'][key].append(metrics)
    
    def log_task_result(self, task_result: dict):
        """Log single task result."""
        self.experiment_data['task_results'].append(task_result)
    
    def save_results(self):
        """Save experiment data to JSON."""
        filepath = self.data_dir / f"{self.experiment_name}.json"
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.experiment_data, f, indent=2, ensure_ascii=False)
        print(f"Results saved to: {filepath}")
        return filepath


class Visualizer:
    """Visualization generator for experiment results."""
    
    def __init__(self, logger: ExperimentLogger):
        self.logger = logger
        self.save_dir = logger.plots_dir
    
    def plot_training_curve(self, train_metrics: List[dict], save_name: str = "training_curve.png"):
        """Plot training loss and accuracy curves."""
        if not train_metrics:
            print("No training metrics to plot")
            return
        
        epochs = [m['epoch'] for m in train_metrics]
        losses = [m.get('loss', 0) for m in train_metrics]
        accuracies = [m.get('accuracy', 0) for m in train_metrics]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        ax1.plot(epochs, losses, 'b-o', linewidth=2, markersize=6)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss')
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(epochs, accuracies, 'g-o', linewidth=2, markersize=6)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training Accuracy')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Training curve saved to: {save_path}")
    
    def plot_accuracy_comparison(self, per_shot_results: Dict[str, List[dict]], 
                                   save_name: str = "accuracy_comparison.png"):
        """Plot accuracy comparison across different shots."""
        shots = sorted(per_shot_results.keys(), key=lambda x: int(x.replace('shot', '')))
        
        means = []
        stds = []
        labels = []
        
        for shot in shots:
            results = per_shot_results[shot]
            if results:
                accuracies = [r.get('mean_accuracy', r.get('accuracy', 0)) for r in results]
                means.append(np.mean(accuracies))
                stds.append(np.std(accuracies))
                labels.append(shot.replace('shot', '-shot'))
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(shots))
        bars = ax.bar(x, means, yerr=stds, capsize=5, 
                     color=['#3498db', '#2ecc71', '#e74c3c', '#9b59b6'],
                     edgecolor='black', linewidth=1.2, alpha=0.8)
        
        ax.set_xlabel('Few-shot Setting')
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy vs Number of Shots (20-way)')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 1.1)
        
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{mean:.2%}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Accuracy comparison saved to: {save_path}")
    
    def plot_confusion_matrix(self, confusion_matrix: np.ndarray, labels: List[str],
                             save_name: str = "confusion_matrix.png"):
        """Plot confusion matrix heatmap."""
        fig, ax = plt.subplots(figsize=(12, 10))
        
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=labels, yticklabels=labels, ax=ax,
                   cbar_kws={'label': 'Count'})
        
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title('Confusion Matrix (20-way Classification)')
        
        plt.tight_layout()
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Confusion matrix saved to: {save_path}")
    
    def plot_confidence_distribution(self, confidences: List[float], 
                                    save_name: str = "confidence_distribution.png"):
        """Plot prediction confidence distribution."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.hist(confidences, bins=20, color='#3498db', edgecolor='black', alpha=0.7)
        ax.axvline(np.mean(confidences), color='red', linestyle='--', 
                  linewidth=2, label=f'Mean: {np.mean(confidences):.3f}')
        ax.axvline(np.median(confidences), color='green', linestyle='--',
                  linewidth=2, label=f'Median: {np.median(confidences):.3f}')
        
        ax.set_xlabel('Prediction Confidence')
        ax.set_ylabel('Frequency')
        ax.set_title('Prediction Confidence Distribution')
        ax.legend()
        
        plt.tight_layout()
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Confidence distribution saved to: {save_path}")
    
    def plot_class_accuracy(self, per_class_acc: Dict[int, float], num_classes: int,
                           save_name: str = "class_accuracy.png"):
        """Plot per-class accuracy."""
        classes = list(range(num_classes))
        accuracies = [per_class_acc.get(i, 0) for i in classes]
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        colors = ['#2ecc71' if acc >= 0.8 else '#e74c3c' if acc < 0.5 else '#f39c12' 
                 for acc in accuracies]
        
        ax.bar(classes, accuracies, color=colors, edgecolor='black', alpha=0.8)
        ax.axhline(np.mean(accuracies), color='blue', linestyle='--', 
                  linewidth=2, label=f'Mean: {np.mean(accuracies):.2%}')
        
        ax.set_xlabel('Class ID')
        ax.set_ylabel('Accuracy')
        ax.set_title('Per-Class Accuracy')
        ax.set_ylim(0, 1.1)
        ax.legend()
        
        plt.tight_layout()
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Class accuracy saved to: {save_path}")
    
    def plot_ode_trajectory_2d(self, trajectories: torch.Tensor, class_names: List[str],
                               save_name: str = "ode_trajectory.png"):
        """Plot 2D projection of ODE trajectory using PCA."""
        try:
            from sklearn.decomposition import PCA
            
            trajectories_np = trajectories.cpu().numpy()
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            colors = plt.cm.tab20(np.linspace(0, 1, len(class_names)))
            
            for i, (traj, name, color) in enumerate(zip(trajectories_np, class_names, colors)):
                if traj.shape[0] > 0:
                    if traj.shape[1] > 2:
                        pca = PCA(n_components=2)
                        traj_2d = pca.fit_transform(traj)
                    else:
                        traj_2d = traj
                    
                    ax.plot(traj_2d[:, 0], traj_2d[:, 1], 
                           'o-', color=color, label=name, linewidth=2, markersize=4)
                    ax.scatter(traj_2d[0, 0], traj_2d[0, 1], 
                              marker='s', s=100, color=color, edgecolor='black')
                    ax.scatter(traj_2d[-1, 0], traj_2d[-1, 1], 
                              marker='*', s=200, color=color, edgecolor='black')
            
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.set_title('ODE Prototype Trajectory (2D Projection)')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            
            plt.tight_layout()
            save_path = self.save_dir / save_name
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"ODE trajectory saved to: {save_path}")
        except Exception as e:
            print(f"Failed to plot ODE trajectory: {e}")
    
    def generate_summary_report(self, save_name: str = "experiment_report.md"):
        """Generate markdown summary report."""
        data = self.logger.experiment_data
        
        report = f"""# NODE-Adapter 实验报告

## 实验信息
- **实验名称**: {data['experiment_name']}
- **时间**: {data['timestamp']}

## 实验配置
"""
        
        config = data.get('config', {})
        for key, value in config.items():
            report += f"- **{key}**: {value}\n"
        
        report += """
## 实验结果

### 准确率汇总
"""
        
        per_shot = data.get('per_shot_results', {})
        for shot, results in sorted(per_shot.items()):
            if results:
                accs = [r.get('mean_accuracy', r.get('accuracy', 0)) for r in results]
                report += f"- **{shot}**: {np.mean(accs):.2%} ± {np.std(accs):.2%}\n"
        
        if data.get('test_metrics'):
            test = data['test_metrics'][0]
            report += f"""
### 测试结果
- **测试准确率**: {test.get('mean_accuracy', 0):.2%} ± {test.get('std_accuracy', 0):.2%}
- **任务数**: {test.get('num_tasks', 0)}
"""
        
        report += """
## 可视化结果

"""
        
        report += """- 训练曲线: `training_curve.png`
- 准确率对比: `accuracy_comparison.png`
- 混淆矩阵: `confusion_matrix.png`
- 置信度分布: `confidence_distribution.png`

## 结论

本实验验证了 NODE-Adapter 在 CIFAR-100 数据集上的少样本分类性能。
"""
        
        save_path = self.save_dir / save_name
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"Report saved to: {save_path}")
        return save_path
    
    def save_all_plots(self):
        """Generate all standard plots."""
        data = self.logger.experiment_data
        
        if data.get('train_metrics'):
            self.plot_training_curve(data['train_metrics'])
        
        if data.get('per_shot_results'):
            self.plot_accuracy_comparison(data['per_shot_results'])
        
        self.generate_summary_report()


def compute_confidence(logits: torch.Tensor) -> np.ndarray:
    """Compute prediction confidence from logits."""
    probs = torch.softmax(logits, dim=-1)
    confidences = probs.max(dim=-1)[0].cpu().numpy()
    return confidences

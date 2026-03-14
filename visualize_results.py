import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import warnings
warnings.filterwarnings('ignore')

import config
import h5py
from tqdm import tqdm
from typing import Any, Dict, Optional
from seed_utils import HierarchicalSeedManager

class ResultsVisualizer:
    def __init__(self, seed_manager=None) -> None:
        self.results: Optional[Dict[str, Any]] = None
        self.seed_manager = seed_manager or HierarchicalSeedManager(config.RANDOM_SEED)
        self.bifurcation_rng = self.seed_manager.numpy_rng("visualization.bifurcation.subsample")
        self.perturbation_rng = self.seed_manager.numpy_rng("visualization.perturbation.subsample")
        self.setup_plotting_style()
        
    def setup_plotting_style(self):
        plt.style.use('default')
        plt.rcParams.update({
            'font.size': 12,
            'axes.labelsize': 14,
            'axes.titlesize': 20,
            'xtick.labelsize': 11,
            'ytick.labelsize': 11,
            'legend.fontsize': 10,
            'figure.titlesize': 20,
            'lines.linewidth': 2,
            'axes.linewidth': 1.5,
            'figure.facecolor': 'white',
            'axes.facecolor': 'white'
        })
    
    def load_results(self):
        h5_path = os.path.join(config.RESULTS_PATH, 'chaos_analysis_results.h5')
        summary_path = os.path.join(config.RESULTS_PATH, 'analysis_summary.json')
        history_path = os.path.join(config.RESULTS_PATH, 'training_history.json')

        print(f"Loading results from h5 file from: {h5_path}")
        if os.path.exists(h5_path):
            try:
                with h5py.File(h5_path, 'r') as f:
                    self.results = {}
                    keep_as_array = {
                        'analyzed_epochs', 'mean_final_perturbed_distances', 'bifurcation_data', 
                        'perturbed_distances'
                    }
                    for key in f.keys():
                        if key in keep_as_array:
                            self.results[key] = f[key][:]  # 保持为 numpy array
                        else:
                            self.results[key] = f[key][:].tolist()  # 其他转成 list
                print(f"Loaded results from h5 file for {len(self.results['epochs'])} epochs")
                return True
            except Exception as e:
                print(f"Could not load h5: {e}")
        
        # Try summary
        if os.path.exists(summary_path):
            try:
                with open(summary_path, 'r') as f:
                    data = json.load(f)
                self.results = data.get('training_curves', {})
                print(f"Loaded results from summary file")
                return True
            except Exception as e:
                print(f"Could not load summary: {e}")
        
        # Fallback to training history
        if os.path.exists(history_path):
            try:
                with open(history_path, 'r') as f:
                    self.results = json.load(f)
                print(f"Loaded training history as fallback")
                return True
            except Exception as e:
                print(f"Could not load history: {e}")
        
        return False
    
    def plot_training_curves(self, save_path=None):
        if not self.results:
            print("No results loaded")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        epochs = self.results.get('epochs', [])
        test_loss = self.results.get('test_loss', [])
        test_accuracy = self.results.get('test_accuracy', [])
        train_loss = self.results.get('train_loss', [])
        train_accuracy = self.results.get('train_accuracy', [])
        
        if not epochs or not test_loss:
            print("No training data available")
            return
        
        # Plot 1: Loss curves
        ax1.plot(epochs, train_loss, color='blue', linewidth=1, label='Train Loss', alpha=0.7)
        ax1.plot(epochs, test_loss, color='brown', linewidth=1, label='Test Loss')
        
        # Mark global optimum
        min_loss_idx = np.argmin(test_loss)
        global_opt_epoch = epochs[min_loss_idx]
        global_opt_loss = test_loss[min_loss_idx]
        
        ax1.plot(global_opt_epoch, global_opt_loss, 'ro', markersize=10, 
                label=f'Global Optimum (Epoch {global_opt_epoch})')
        
        ax1.set_xlabel('Training Epochs', fontsize=16)
        ax1.set_ylabel('Loss', fontsize=16)
        ax1.set_title('Training Curves - Loss', fontsize=20)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: Accuracy curves
        if test_accuracy and train_accuracy:
            ax2.plot(epochs, train_accuracy, color='blue', linewidth=1, 
                    label='Train Accuracy', alpha=0.7)
            ax2.plot(epochs, test_accuracy, color='brown', linewidth=1, 
                    label='Test Accuracy')
            ax2.set_xlabel('Training Epochs', fontsize=16)
            ax2.set_ylabel('Accuracy (%)', fontsize=16)
            ax2.set_title('Training Curves - Accuracy', fontsize=20)
            ax2.grid(True, alpha=0.3)
            ax2.legend()
        else:
            ax2.text(0.5, 0.5, 'Test accuracy data not available', 
                    transform=ax2.transAxes, ha='center', va='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        
        plt.show()
    
    def plot_test_loss_with_ftle(self, *, save_path: Optional[str] = None) -> None:
        epochs = np.asarray(self.results.get("epochs", []))
        test_loss = np.asarray(self.results.get("test_loss", []), dtype=np.float32)
        analyzed_epochs = np.asarray(self.results.get("analyzed_epochs", []))
        ftle_mean = np.asarray(self.results.get("ftle_mean", []), dtype=np.float32)

        fig, ax1 = plt.subplots(figsize=(12, 5))

        ax1.plot(epochs, test_loss, color="brown", linewidth=1.2, label="Test Loss")
        
        min_loss_idx = np.argmin(test_loss)
        global_opt_epoch = epochs[min_loss_idx]
        ax1.axvline(x=global_opt_epoch, color='r', linestyle='--',
                label=f"Global Optimum (Epoch {global_opt_epoch})")
        
        seed = getattr(config, "RANDOM_SEED", None)
        lr = getattr(config, "LEARNING_RATE", None)
        ax1.set_xlabel(f"Epochs (seed={seed}, lr={lr})", fontsize=16)
        ax1.set_ylabel("Test Loss", color="brown", fontsize=16)
        ax1.tick_params(axis="y", labelcolor="brown")
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, int(np.max(epochs)))

        ax2 = ax1.twinx()
        ftle_color = "green"
        ax2.plot(analyzed_epochs, ftle_mean, color=ftle_color, linewidth=1.2, label="Mean FTLE")
        ax2.axhline(0.0, color="gray", linestyle="--", linewidth=1, alpha=0.6)
        ax2.set_ylabel("FTLE", color=ftle_color, fontsize=16)
        ax2.tick_params(axis="y", labelcolor=ftle_color)
        
        ymin, ymax = ax2.get_ylim()
        ax2.set_ylim(ymin, ymax)
        ax2.axhspan(0, ymax, alpha=0.05, color="red")   # FTLE > 0
        ax2.axhspan(ymin, 0, alpha=0.05, color="blue")  # FTLE < 0

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right", framealpha=0.6)

        ax1.set_title("Test Loss vs. FTLE (Benettin)", fontsize=18)
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches="tight")
            print(f"Saved plot to {save_path}")

        plt.show()
        
    def plot_mean_final_pt_dists_bifurcation_diagram_with_animation(self, video_path=None,
                                                                subsample_epochs=1,
                                                                subsample_samples=100):
        # ==== 1. 准备数据 ====
        if not self.results or 'bifurcation_data' not in self.results:
            print("No bifurcation data available.")
            return

        epochs = self.results['epochs']
        test_loss = self.results['test_loss']
        analyzed_epochs = self.results['analyzed_epochs']
        mean_final_perturbed_distances = self.results['mean_final_perturbed_distances']
        bifurcation_data = self.results['bifurcation_data']

        # ==== 2. 创建画布和静态元素 ====
        fig, ax = plt.subplots(figsize=(16, 8))

        # 测试损失
        ax.plot(epochs, test_loss, color='brown', linewidth=1, label='Test Loss')
        ax.set_ylabel('Test Loss', color='brown', fontsize=16)
        ax.tick_params(axis='y', labelcolor='brown')

        # 全局最优
        min_loss_idx = np.argmin(test_loss)
        global_opt_epoch = epochs[min_loss_idx]
        ax.axvline(x=global_opt_epoch, color='r', linestyle='--',
                label=f"Global Optimum (Epoch {global_opt_epoch})")

        # 第二轴
        ax_twin = ax.twinx()
        ax_twin.set_ylabel('Mean Final Perturbed Distance / Reduced Sum Bifurcation', color='green', fontsize=16) 
        ax_twin.tick_params(axis='y', labelcolor='green')  # 第二轴刻度颜色
        
        # 渐进距离
        valid_mask = np.isfinite(mean_final_perturbed_distances)
        
        if np.any(valid_mask):
            ax_twin.plot(analyzed_epochs[valid_mask], mean_final_perturbed_distances[valid_mask],
                        color='green', linewidth=1, label='Mean Final Perturbed Distance')

        # ==== 3. 分岔数据降采样 ====
        epoch_indices = np.arange(0, analyzed_epochs.size, subsample_epochs)
        selected_epochs = analyzed_epochs[epoch_indices]
        selected_bifurcation = bifurcation_data[epoch_indices, :, :]

        sample_indices = self.bifurcation_rng.choice(
            selected_bifurcation.shape[1],
            size=min(subsample_samples, selected_bifurcation.shape[1]),
            replace=False,
        )
        selected_bifurcation = selected_bifurcation[:, sample_indices, :]

        n_timesteps = selected_bifurcation.shape[2]

        # ==== 4. 设置坐标轴范围 ====
        # 展平
        a_flat = mean_final_perturbed_distances.reshape(-1)
        b_flat = bifurcation_data.reshape(-1)

        # 合并并去掉 NaN/Inf
        all_vals = np.concatenate([a_flat, b_flat])
        finite_vals = all_vals[np.isfinite(all_vals)]
        y_min, y_max = np.min(finite_vals), np.max(finite_vals)
        ax_twin.set_ylim(y_min, y_max)
        ax.set_xlim(selected_epochs.min(), selected_epochs.max())

        # ==== 5. 添加标题、网格、图例（在动画前完成） ====
        ax.set_xlabel('Epochs', fontsize=16)
        ax.set_title('Test Loss + Mean Final Perturbed Distance + Bifurcation Map', fontsize=20)
        ax.grid(True, alpha=0.3)

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax_twin.get_legend_handles_labels()
        ax.legend(
            lines1 + lines2,          # 所有曲线对象
            labels1 + labels2,        # 对应标签
            loc='lower right',        # 固定在右上角
            framealpha=0.5            # 图例背景透明度 0~1
        )

        # ==== 6. 初始化散点和文本 ====
        scatter = ax_twin.scatter([], [], s=0.3, alpha=0.5, color='#1f77b4', label='Bifurcation Map')
        timestep_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                                fontsize=16, verticalalignment='top',
                                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        # ==== 7. 动画函数 ====
        def init():
            scatter.set_offsets(np.empty((0, 2)))
            timestep_text.set_text('')
            return [scatter, timestep_text]

        def animate(frame):
            vals = selected_bifurcation[:, :, frame].flatten()
            xs = np.repeat(selected_epochs, selected_bifurcation.shape[1])
            scatter.set_offsets(np.column_stack((xs, vals)))
            timestep_text.set_text(f'Timestep: {frame}')
            return [scatter, timestep_text]

        # ==== 8. 创建动画 ====
        print("DEBUG selected_bifurcation shape:", selected_bifurcation.shape)
        print("DEBUG n_timesteps:", n_timesteps)

        pbar = tqdm(total=n_timesteps, desc="Generating frames")

        def animate_with_pbar(frame):
            out = animate(frame)   # 调用你原来的 animate
            pbar.update(1)
            return out

        frames = list(range(n_timesteps))
        anim = FuncAnimation(
            fig, animate_with_pbar, init_func=init,
            frames=frames, interval=50,
            blit=True, repeat=False
        )

        # ==== 9. 保存视频 ====
        if video_path:
            writer = FFMpegWriter(fps=20)
            anim.save(video_path, writer=writer, dpi=100)
            print(f"Animation saved to {video_path}")

        plt.tight_layout()
        plt.show()
    
    def create_perturbed_distance_animation(self, save_path=None, subsample_epochs=10, subsample_samples=50):
        """
        Create scatter plot animation showing perturbation distances across epochs for each timestep
        
        Args:
            save_path: Path to save the animation (mov format)
            subsample_epochs: Sample every N epochs to reduce animation length
            subsample_samples: Sample N samples to reduce scatter plot density
        """
        if not self.results or 'perturbed_distances' not in self.results:
            print("No perturbed_distances data available. Please ensure 'perturbed_distances' is saved in results.")
            return
        
        
        perturbed_distances = np.array(self.results['perturbed_distances'])  # Shape: (n_epochs, n_samples, n_timesteps)
        analyzed_epochs = self.results.get('analyzed_epochs', list(range(1, perturbed_distances.shape[0] + 1)))
        
        # Subsample epochs and samples
        epoch_indices = list(range(0, len(analyzed_epochs), subsample_epochs))
        selected_epochs = [analyzed_epochs[i] for i in epoch_indices]
        selected_perturbed_distances = perturbed_distances[epoch_indices, :, :]
        
        sample_indices = self.perturbation_rng.choice(
            perturbed_distances.shape[1],
            size=min(subsample_samples, perturbed_distances.shape[1]),
            replace=False,
        )
        selected_perturbed_distances = selected_perturbed_distances[:, sample_indices, :]
        
        n_timesteps = selected_perturbed_distances.shape[2]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(16, 8))
        
        # Set axis limits
        vmin = selected_perturbed_distances.min()
        vmax = selected_perturbed_distances.max()

        ax.set_xlim(min(selected_epochs), max(selected_epochs))
        ax.set_ylim(vmin, vmax)
        
        ax.set_xlabel('Epoch', fontsize=16)
        ax.set_ylabel('Log Perturbation Distance', fontsize=16)
        ax.set_title('Perturbation Distance Evolution', fontsize=20)
        ax.grid(True, alpha=0.3)
        
        # 初始化绘图元素
        scatter = ax.scatter([], [], s=0.5, alpha=0.5, color='green', label='Individual Samples')  # 散点：单个样本
        mean_line, = ax.plot([], [], color='red', linewidth=1, label='Sample Mean') 
        timestep_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                            fontsize=16, verticalalignment='top')
        ax.legend(loc='upper right') 
        
        def init():
            """Initialize animation"""
            scatter.set_offsets(np.empty((0, 2)))
            mean_line.set_data([], [])
            timestep_text.set_text('')
            return [scatter, timestep_text]
        
        def animate(frame):
            """Update animation for each frame"""
            timestep = frame
            
            # Get data for current timestep
            epoch_data = np.array(selected_epochs)
            distance_data = selected_perturbed_distances[:, :, timestep].flatten()
            
            # Update scatter plot
            scatter.set_offsets(np.column_stack((epoch_data.repeat(len(sample_indices)), 
                                              distance_data)))
            
            # 对每个epoch，计算该epoch下所有样本在当前时间步的均值（axis=1：按样本维度求平均）
            sample_mean_per_epoch = np.nanmean(selected_perturbed_distances[:, :, timestep], axis=1)
            mean_line.set_data(selected_epochs, sample_mean_per_epoch)
        
            # Update timestep text
            timestep_text.set_text(f'Timestep: {timestep}')
            
            return [scatter, timestep_text]
        
        # ==== 8. 创建动画 ====

        pbar = tqdm(total=n_timesteps, desc="Generating frames")

        def animate_with_pbar(frame):
            out = animate(frame)   # 调用你原来的 animate
            pbar.update(1)
            return out

        frames = list(range(n_timesteps))
        anim = FuncAnimation(
            fig, animate_with_pbar, init_func=init,
            frames=frames, interval=2,
            blit=True, repeat=False
        )
    
        # Save animation
        if save_path:
            writer = FFMpegWriter(fps=30)
            anim.save(save_path, writer=writer, dpi=100)
            print(f"Animation saved to {save_path}")
        
        plt.show()
        
        return anim
    

    def generate_all_plots(self):
        """Generate all plots"""
        
        if not self.load_results():
            print("Cannot load results. Please run analysis first.")
            return
        
        figures_dir = os.path.join(config.RESULTS_PATH, 'figures')
        os.makedirs(figures_dir, exist_ok=True)
        
        # Generate plots
        print("Generating plots...")
        plot_name = 'training_curves'
        try:
            save_path = os.path.join(figures_dir, f"{plot_name}.png") if config.SAVE_FIGURES else None
            self.plot_training_curves(save_path=save_path)
            print(f"✓ Generated {plot_name} plot")
        except Exception as e:
            print(f"✗ Failed to generate {plot_name} plot: {str(e)}")

        plot_name = "test_loss_with_ftle"
        if self.results and all(
            key in self.results for key in ["epochs", "test_loss", "analyzed_epochs", "ftle_mean"]
        ):
            try:
                save_path = os.path.join(figures_dir, f"{plot_name}.png") if config.SAVE_FIGURES else None
                self.plot_test_loss_with_ftle(save_path=save_path)
                print(f"✓ Generated {plot_name} plot")
            except Exception as e:
                print(f"✗ Failed to generate {plot_name} plot: {str(e)}")
        
        # Generate combined animation: Test Loss + Mean Final Perturbed Distance + Bifurcation Map
        if self.results and all(key in self.results for key in ['bifurcation_data', 'mean_final_perturbed_distances', 'test_loss']):
            print("\nGenerating combined animation (Test Loss + Mean Final Perturbed Distance + Bifurcation Map)...")
            try:
                # 1. 定义视频保存路径（仅当需要保存时生成路径）
                combined_anim_path = os.path.join(figures_dir, 'combined_bifurcation_animation.mov') if config.SAVE_FIGURES else None
                
                # 2. 调用动画函数（核心：只传 video_path，不传 save_path，避免生成静态图）
                self.plot_mean_final_pt_dists_bifurcation_diagram_with_animation(
                    video_path=combined_anim_path,  # 只保存视频，不保存静态图
                    subsample_epochs=1,             # 按需求调整：每隔1个epoch取1个（不丢数据）
                    subsample_samples=config.NUM_TEST_SAMPLES      # 按需求调整
                )
                
                print("✓ Generated combined animation (Test Loss + Mean Final Perturbed Distance + Bifurcation Map)")
            except Exception as e:
                print(f"✗ Error generating combined animation: {str(e)}")

        print(f"\nAnimations saved to {figures_dir}")

        # Generate animations if perturbed_distance data is available
        if self.results and 'perturbed_distances' in self.results:
            print("\nGenerating perturbed_distance animations...")
            
            try:
                anim_path = os.path.join(figures_dir, 'perturbed_distance_animation.mov') if config.SAVE_FIGURES else None
                self.create_perturbed_distance_animation(
                    save_path=anim_path, 
                    subsample_epochs=1, 
                    subsample_samples=config.NUM_TEST_SAMPLES)
                print("✓ Generated perturbed_distance animation")
            except Exception as e:
                print(f"✗ Error generating perturbed_distance animation: {str(e)}")
        
        print(f"\nPlots saved to {figures_dir}")


__all__ = ["ResultsVisualizer"]

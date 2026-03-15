import os
import numpy as np
import torch
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
        self.h5_path: Optional[str] = None
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
            'legend.fontsize': 12,
            'figure.titlesize': 20,
            'lines.linewidth': 2,
            'axes.linewidth': 1.5,
            'figure.facecolor': 'white',
            'axes.facecolor': 'white'
        })
    
    def load_results(self):
        h5_path = os.path.join(config.RESULTS_PATH, 'chaos_analysis_results.h5')
        self.h5_path = h5_path

        allowed_h5_keys = {
            'epochs',
            'train_loss',
            'test_loss',
            'train_accuracy',
            'test_accuracy',
            'grad_norms',
            'analyzed_epochs',
            'bifurcation_data',
            'sample_indices',
            'ftle_mean',
            'ftle_per_sample',
        }
        def _load_h5(path):
            if not os.path.exists(path):
                return None

            with h5py.File(path, 'r') as f:
                data = {}
                for key in f.keys():
                    if key not in allowed_h5_keys:
                        continue
                    ds = f[key]
                    if key == 'bifurcation_data':
                        # Defer heavy bifurcation_data loading to animation methods.
                        data['bifurcation_data_shape'] = tuple(ds.shape)
                        continue
                    data[key] = ds[:].tolist()
            return data

        try:
            data = _load_h5(h5_path)
        except Exception as e:
            print(f"[load_results] Failed to load h5: {e}", flush=True)
            return False

        if data is None:
            print(f"[load_results] H5 file not found: {h5_path}", flush=True)
            return False

        self.results = data
        print("[load_results] Loaded h5 data.", flush=True)
        return True
    
    def plot_training_curves(self, save_path=None):
        print("Generating training_curves plot...")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        epochs = self.results.get('epochs', [])
        test_loss = self.results.get('test_loss', [])
        test_accuracy = self.results.get('test_accuracy', [])
        train_loss = self.results.get('train_loss', [])
        train_accuracy = self.results.get('train_accuracy', [])
        
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
            print(f"✓ training_curves saved to {save_path}")
        
        plt.show()
    
    def plot_test_loss_with_ftle(self, *, save_path: Optional[str] = None) -> None:
        print("Generating test_loss_with_ftle plot...")
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
            print(f"✓ test_loss_with_ftle saved to {save_path}")

        plt.show()

    def plot_test_loss_bifurcation_animation_gpu(
        self,
        video_path: str,
        subsample_epochs: int = 1,
        subsample_samples: int = 100,
        fps: int = 20,
        dpi: int = 100,
        device: Optional[str] = None,
        point_alpha: float = 0.5,
        point_color: tuple = (31, 119, 180),
        codec: Optional[str] = None,
        frame_step: int = 2,
    ) -> None:
        print("Generating test-loss vs bifurcation GPU animation...", flush=True)
        if not self.results:
            raise ValueError("No bifurcation data available.")
        if not video_path:
            raise ValueError("video_path must be provided.")

        print("[gpu_anim] Loading rendering dependencies...", flush=True)
        try:
            import cv2
        except ImportError as exc:
            raise ImportError("opencv-python is required for GPU animation rendering.") from exc

        try:
            import imageio.v2 as imageio
        except ImportError as exc:
            raise ImportError("imageio is required for video writing.") from exc

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_device = torch.device(device)

        if codec is None:
            codec = "h264_videotoolbox" if os.uname().sysname == "Darwin" else "libx264"

        print("[gpu_anim] Preparing metadata and selected epochs...", flush=True)
        epochs = np.asarray(self.results['epochs'])
        test_loss = np.asarray(self.results['test_loss'], dtype=np.float32)
        analyzed_epochs = np.asarray(self.results['analyzed_epochs'])

        bifurcation_data = self.results.get('bifurcation_data', None)
        use_h5 = bifurcation_data is None and self.h5_path and os.path.exists(self.h5_path)
        if not use_h5 and bifurcation_data is None:
            raise ValueError("No bifurcation data available.")

        if use_h5:
            print(f"[gpu_anim] Opening H5 dataset: {self.h5_path}", flush=True)
            with h5py.File(self.h5_path, 'r') as h5_file:
                h5_ds = h5_file['bifurcation_data']
                shape = tuple(int(v) for v in h5_ds.shape)
                est_gib = (np.prod(shape) * h5_ds.dtype.itemsize) / (1024 ** 3)
                print(
                    f"[gpu_anim] Loading full bifurcation_data to RAM. shape={shape}, dtype={h5_ds.dtype}, est={est_gib:.2f} GiB",
                    flush=True,
                )
                bifurcation_data = h5_ds[:]
        else:
            print("[gpu_anim] Using bifurcation data from memory.", flush=True)
            bifurcation_data = np.asarray(bifurcation_data)

        bifurcation_data = np.asarray(bifurcation_data)
        if bifurcation_data.ndim != 3:
            raise ValueError(f"bifurcation_data must be 3D [epochs, samples, timesteps], got {bifurcation_data.shape}")
        if not bifurcation_data.flags["C_CONTIGUOUS"]:
            bifurcation_data = np.ascontiguousarray(bifurcation_data)

        n_available_samples = int(bifurcation_data.shape[1])
        n_timesteps = int(bifurcation_data.shape[2])

        data_on_gpu = False
        bifurcation_tensor = None
        if torch_device.type == "cuda":
            try:
                print("[gpu_anim] Transferring full bifurcation_data to GPU VRAM...", flush=True)
                bifurcation_tensor = torch.from_numpy(bifurcation_data).to(torch_device)
                data_on_gpu = True
                print("[gpu_anim] Full tensor is now resident on GPU.", flush=True)
            except RuntimeError as exc:
                print(f"[gpu_anim] GPU preload failed, fallback to CPU-resident data: {exc}", flush=True)

        epoch_indices = np.arange(0, analyzed_epochs.size, max(1, int(subsample_epochs)))

        selected_sample_count = min(int(subsample_samples), n_available_samples)

        selected_epochs = analyzed_epochs[epoch_indices]
        sample_indices = np.sort(self.bifurcation_rng.choice(
            n_available_samples,
            size=selected_sample_count,
            replace=False,
        ))
        frame_step = max(1, int(frame_step))
        frame_indices = np.arange(0, n_timesteps, frame_step, dtype=np.int32)
        print(
            f"[gpu_anim] Data ready: epochs={len(selected_epochs)}, samples={selected_sample_count}, timesteps={n_timesteps}, rendered_frames={len(frame_indices)}, frame_step={frame_step}",
            flush=True,
        )

        use_full_epochs = (
            len(epoch_indices) == bifurcation_data.shape[0]
            and int(epoch_indices[0]) == 0
            and np.all(np.diff(epoch_indices) == 1)
        )
        use_full_samples = (
            len(sample_indices) == n_available_samples
            and int(sample_indices[0]) == 0
            and np.all(np.diff(sample_indices) == 1)
        )
        epoch_indices_t = torch.from_numpy(epoch_indices.astype(np.int64)).to(torch_device) if data_on_gpu else None
        sample_indices_t = torch.from_numpy(sample_indices.astype(np.int64)).to(torch_device) if data_on_gpu else None

        def _frame_vals(frame: int) -> torch.Tensor:
            if data_on_gpu:
                frame_slice = bifurcation_tensor[:, :, int(frame)]
                if not use_full_epochs:
                    frame_slice = frame_slice.index_select(0, epoch_indices_t)
                if not use_full_samples:
                    frame_slice = frame_slice.index_select(1, sample_indices_t)
                return frame_slice.reshape(-1)

            vals_np = bifurcation_data[np.ix_(epoch_indices, sample_indices, [frame])][:, :, 0].reshape(-1)
            return torch.from_numpy(vals_np).to(torch_device, non_blocking=(torch_device.type == "cuda"))

        print("[gpu_anim] Building base figure...", flush=True)
        fig, ax = plt.subplots(figsize=(16, 8), dpi=dpi)
        ax.plot(epochs, test_loss, color='brown', linewidth=1, label='Test Loss')
        ax.set_ylabel('Test Loss', color='brown', fontsize=16)
        ax.tick_params(axis='y', labelcolor='brown')

        min_loss_idx = int(np.argmin(test_loss))
        global_opt_epoch = epochs[min_loss_idx]
        ax.axvline(x=global_opt_epoch, color='r', linestyle='--', label=f"Global Optimum (Epoch {global_opt_epoch})")

        ax_twin = ax.twinx()
        ax_twin.set_ylabel('Reduced Sum Bifurcation', color='green', fontsize=16)
        ax_twin.tick_params(axis='y', labelcolor='green')

        print("[gpu_anim] Probing value range for y-axis scaling...", flush=True)
        probe_count = 64
        probe_frames = np.linspace(0, n_timesteps - 1, min(probe_count, n_timesteps), dtype=int)
        bif_min = np.inf
        bif_max = -np.inf
        for frame in probe_frames:
            vals_t = _frame_vals(int(frame))
            finite_mask = torch.isfinite(vals_t)
            if not torch.any(finite_mask):
                continue
            finite_vals = vals_t[finite_mask]
            bif_min = min(bif_min, float(torch.min(finite_vals).item()))
            bif_max = max(bif_max, float(torch.max(finite_vals).item()))
        if not np.isfinite(bif_min) or not np.isfinite(bif_max):
            bif_min, bif_max = -1.0, 1.0
        ax_twin.set_ylim(bif_min, bif_max)
        ax.set_xlim(float(np.min(selected_epochs)), float(np.max(selected_epochs)))
        ax.set_xlabel('Epochs', fontsize=16)
        ax.set_title('Test Loss + Bifurcation Map', fontsize=20)
        ax.grid(True, alpha=0.3)

        ax_twin.scatter([], [], s=0.3, alpha=point_alpha, color=np.array(point_color) / 255.0, label='Bifurcation Map')
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax_twin.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='lower right', framealpha=0.5)

        print("[gpu_anim] Precomputing canvas and pixel mapping...", flush=True)
        plt.tight_layout()
        fig.canvas.draw()

        background_rgba = np.asarray(fig.canvas.buffer_rgba()).copy()
        height, width = background_rgba.shape[:2]

        x0_data = float(np.min(selected_epochs))
        x1_data = float(np.max(selected_epochs))
        y0_data, y1_data = ax_twin.get_ylim()
        px0, py0 = ax_twin.transData.transform((x0_data, y0_data))
        px1, py1 = ax_twin.transData.transform((x1_data, y1_data))

        x_slope = (px1 - px0) / (x1_data - x0_data + 1e-12)
        x_bias = px0 - x_slope * x0_data
        y_slope = (py1 - py0) / (y1_data - y0_data + 1e-12)
        y_bias = py0 - y_slope * y0_data

        x_data = np.repeat(selected_epochs, selected_sample_count).astype(np.float32)
        x_pix = torch.from_numpy(np.rint(x_data * x_slope + x_bias).astype(np.int64)).to(torch_device)
        x_pix = torch.clamp(x_pix, 0, width - 1)

        point_color_np = np.asarray(point_color, dtype=np.float32).reshape(1, 3)
        one_minus_alpha = float(max(0.0, min(1.0, 1.0 - point_alpha)))

        try:
            print(f"[gpu_anim] Initializing video writer with codec='{codec}'...", flush=True)
            writer = imageio.get_writer(
                video_path,
                fps=fps,
                codec=codec,
                format='FFMPEG',
                pixelformat='yuv420p',
            )
        except Exception:
            print("[gpu_anim] Requested codec unavailable, fallback to 'libx264'.", flush=True)
            writer = imageio.get_writer(
                video_path,
                fps=fps,
                codec='libx264',
                format='FFMPEG',
                pixelformat='yuv420p',
            )

        print("[gpu_anim] Starting frame generation...", flush=True)
        try:
            for frame in tqdm(frame_indices, desc="Generating GPU frames"):
                vals_t = _frame_vals(int(frame))

                y_disp = vals_t * y_slope + y_bias
                y_pix = torch.round((height - 1) - y_disp).to(torch.int64)

                valid = (y_pix >= 0) & (y_pix < height)
                if torch.any(valid):
                    xv = x_pix[valid]
                    yv = y_pix[valid]
                    flat_idx = yv * width + xv
                    counts = torch.bincount(flat_idx, minlength=height * width)
                    nz = torch.nonzero(counts, as_tuple=False).squeeze(1)

                    frame_rgb = background_rgba[:, :, :3].copy()
                    if nz.numel() > 0:
                        alpha_eff = 1.0 - torch.pow(one_minus_alpha, counts[nz].to(torch.float32))
                        alpha_eff = torch.clamp(alpha_eff, 0.0, 1.0)

                        ys = (nz // width).cpu().numpy()
                        xs = (nz % width).cpu().numpy()
                        a = alpha_eff.cpu().numpy().reshape(-1, 1)

                        base = frame_rgb[ys, xs, :].astype(np.float32)
                        blended = (1.0 - a) * base + a * point_color_np
                        frame_rgb[ys, xs, :] = blended.astype(np.uint8)
                else:
                    frame_rgb = background_rgba[:, :, :3].copy()

                label = f"Timestep: {int(frame)}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.75
                text_thickness = 1
                
                cv2.putText(
                    frame_rgb, label, (24, 36),
                    font, font_scale, (0, 0, 0), text_thickness, cv2.LINE_AA,
                )
                writer.append_data(frame_rgb)
        finally:
            writer.close()
            plt.close(fig)

        print(f"✓ GPU animation saved to {video_path}", flush=True)

__all__ = ["ResultsVisualizer"]

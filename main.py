"""
Main experiment runner - orchestrates the complete pipeline
"""

import os
import sys
import argparse
import time
from datetime import datetime
from typing import Optional
import torch
import config
from train import LSTMTrainer
from analysis_runner import AnalysisRunner
from visualize_results import ResultsVisualizer
from seed_utils import HierarchicalSeedManager

print("="*80)
print("Chaos Analysis Experiment - RNN (LSTM/GRU/simple RNN)")
print("="*80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Device: {torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')}")
print(f"Random seed: {config.RANDOM_SEED}")
print(f"Learning Rate: {config.LEARNING_RATE}")


print(f"Data path: {config.DATA_PATH}")
print(f"Results path: {config.RESULTS_PATH}")
print(f"Checkpoints path: {config.CHECKPOINT_PATH}")

class ExperimentRunner:
    def __init__(self):
        self.start_time = time.time()
        self.seed_manager = HierarchicalSeedManager(config.RANDOM_SEED)
        self.setup_experiment()
        
    def setup_experiment(self):
        for path in [config.DATA_PATH, config.RESULTS_PATH, config.CHECKPOINT_PATH]:
            os.makedirs(path, exist_ok=True)
        
        print(f"Setting global random seed {config.RANDOM_SEED} for reproducibility")
        self.seed_manager.apply_global_seed()
        
    def run_training(self, max_epochs=None):
        print("\nSTEP 1: TRAINING RNN MODEL")
        print("-" * 40)
        
        if max_epochs is None:
            max_epochs = config.MAX_EPOCHS
        
        try:
            trainer = LSTMTrainer(seed_manager=self.seed_manager)
            vocab_size = trainer.load_data()
            trainer.initialize_model(vocab_size)
            
            print(f"Starting RNN training for {max_epochs} epochs (cell_type={config.RNN_CELL_TYPE})...")
            print()
            
            history = trainer.train(max_epochs)
            
            print(f"✓ RNN training completed successfully!")
            print(f"✓ Best epoch: {trainer.best_epoch}")
            print(f"✓ Best test loss: {trainer.best_test_loss:.4f}")
            print()
            
            return True
            
        except Exception as e:
            print(f"✗ RNN training failed: {str(e)}")
            return False
    
    def run_chaos_analysis(
        self,
        max_analysis_epochs=config.MAX_EPOCHS,
        *,
        epochs_to_check=None,
        start_epoch: int = 1,
        end_epoch: Optional[int] = None,
        interval: int = 1,
    ):
        print("STEP 2: CHAOS ANALYSIS")
        print("-" * 40)
        
        try:
            analyzer = AnalysisRunner(seed_manager=self.seed_manager)
            analyzer.load_data_and_model()
            
            if not analyzer.load_training_history():
                print("✗ No training history found. Please run training first.")
                return False
            
            if end_epoch is None:
                end_epoch = max_analysis_epochs
            print(f"Starting chaos analysis (max_analysis_epochs={max_analysis_epochs})...")
            print()
            
            epochs, *_ = analyzer.analyze_chaos_dynamics(
                start_epoch=start_epoch,
                end_epoch=end_epoch,
                interval=interval,
                epochs_to_check=epochs_to_check,
            )
            
            analyzer.save_results()
            
            print("✓ Chaos analysis completed successfully!")
            print(f"✓ Analyzed {len(epochs)} epochs")
            print("✓ Results saved for manual inspection")
            print()
            
            return True
            
        except Exception as e:
            print(f"✗ Chaos analysis failed: {str(e)}")
            return False
    
    def run_visualization(self, visualization_epochs=config.MAX_EPOCHS):
        print("STEP 3: GENERATING VISUALIZATIONS")
        print("-" * 40)

        visualizer = ResultsVisualizer(seed_manager=self.seed_manager)
        if not visualizer.load_results():
            print("✗ Cannot load results. Please run analysis first.")
            return False

        figures_dir = os.path.join(config.RESULTS_PATH, 'figures')
        os.makedirs(figures_dir, exist_ok=True)

        training_curves_path = os.path.join(figures_dir, 'training_curves.png')
        test_loss_ftle_path = os.path.join(figures_dir, 'test_loss_with_ftle.png')
        combined_anim_gpu_path = os.path.join(figures_dir, 'combined_bifurcation_animation_gpu.mov')

        visualizer.plot_training_curves(save_path=training_curves_path, max_epoch=visualization_epochs)
        visualizer.plot_test_loss_with_ftle(save_path=test_loss_ftle_path, max_epoch=visualization_epochs)
        visualizer.plot_test_loss_bifurcation_animation_gpu(
            video_path=combined_anim_gpu_path,
            subsample_epochs=1,
            subsample_samples=config.NUM_TEST_SAMPLES,
            max_epoch=visualization_epochs,
        )

        print(f"✓ Visualizations completed. Files saved to {figures_dir}")
        print()
        return True
    
    def print_summary(self):
        """Print experiment summary"""
        
        total_time = time.time() - self.start_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        
        print("="*80)
        print("EXPERIMENT SUMMARY")
        print("="*80)
        print(f"Total runtime: {hours}h {minutes}m")
        print(f"Results directory: {config.RESULTS_PATH}")
        print(f"Checkpoints directory: {config.CHECKPOINT_PATH}")
        print()
        
        print("="*80)

def main():
    parser = argparse.ArgumentParser(description='Run RNN Chaos Analysis Experiment')
    parser.add_argument('-te', type=int, default=config.MAX_EPOCHS,
                       help='short for training epochs, number of training epochs')
    parser.add_argument('-ae', type=int, default=config.MAX_EPOCHS,
                       help='short for analysis epochs, number of epochs to analyze')
    parser.add_argument('--epochs_to_check', type=str, default=None,
                       help='Comma-separated explicit epochs to analyze, e.g. "11,50,100,300,600,900"')
    parser.add_argument('--epoch_range', type=str, default=None,
                       help='Epoch range to analyze (inclusive), e.g. "600-1000"')
    parser.add_argument('--analysis_interval', type=int, default=1,
                       help='Stride for range-based analysis (default 1)')
    parser.add_argument('-ve', type=int, default=config.MAX_EPOCHS,
                       help='short for visualization epochs, only visualize first N epochs')
    parser.add_argument('-st', action='store_true',
                       help='Skip training if results exist')
    parser.add_argument('-sa', action='store_true', 
                       help='Skip analysis if results exist')
    parser.add_argument('-sv', action='store_true',
                       help='Skip visualization generation')
    parser.add_argument('-qt', action='store_true',
                       help='Quick test run (50 train epochs, 20 analysis epochs)')
    
    args = parser.parse_args()
    if args.ve is not None and args.ve <= 0:
        parser.error("-ve must be a positive integer")
    
    # Quick test mode
    if args.qt:
        args.te = 50
        args.ae = 20
        print("QUICK TEST MODE: Reduced epochs for fast testing")
        print()
    
    runner = ExperimentRunner()
    
    success = True
    
    # Step 1: Training
    if not args.st:
        success = success and runner.run_training(args.te)
        if not success:
            print("Training failed. Stopping experiment.")
            return
    
    # Step 2: Chaos Analysis
    if not args.sa:
        epochs_to_check = None
        start_epoch = 1
        end_epoch = args.ae
        if args.epochs_to_check:
            try:
                epochs_to_check = [int(x.strip()) for x in args.epochs_to_check.split(",") if x.strip()]
            except Exception:
                parser.error('--epochs_to_check must be a comma-separated list of integers, e.g. "11,50,100"')
        if args.epoch_range:
            s = args.epoch_range.strip()
            sep = "-" if "-" in s else (":" if ":" in s else None)
            if sep is None:
                parser.error('--epoch_range must look like "600-1000" (or "600:1000")')
            try:
                a, b = s.split(sep, 1)
                start_epoch = int(a.strip())
                end_epoch = int(b.strip())
            except Exception:
                parser.error('--epoch_range must look like "600-1000" (or "600:1000")')

        success = success and runner.run_chaos_analysis(
            max_analysis_epochs=args.ae,
            epochs_to_check=epochs_to_check,
            start_epoch=start_epoch,
            end_epoch=end_epoch,
            interval=args.analysis_interval,
        )
        if not success:
            print("Analysis failed. Stopping experiment.")
            return
    
    # Step 3: Visualization
    if not args.sv:
        success = success and runner.run_visualization(args.ve)
    
    # Print summary
    runner.print_summary()
    
    if success:
        print("✓ Experiment completed successfully!")
    else:
        print("⚠ Experiment completed with some issues.")

if __name__ == "__main__":
    main()

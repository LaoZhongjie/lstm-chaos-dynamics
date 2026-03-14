"""
Main experiment runner - orchestrates the complete pipeline
"""

import os
import sys
import argparse
import time
from datetime import datetime
import torch
import config
from train import LSTMTrainer
from analysis_runner import AnalysisRunner
from visualize_results import ResultsVisualizer
from seed_utils import HierarchicalSeedManager

print("="*80)
print("Chaos Analysis Experiment - LSTM")
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
        print("Setting up experiment...")
        for path in [config.DATA_PATH, config.RESULTS_PATH, config.CHECKPOINT_PATH]:
            os.makedirs(path, exist_ok=True)
        
        print(f"Setting global random seed {config.RANDOM_SEED} for reproducibility")
        self.seed_manager.apply_global_seed()
        
    def run_training(self, max_epochs=None):
        print("\nSTEP 1: TRAINING LSTM MODEL")
        print("-" * 40)
        
        if max_epochs is None:
            max_epochs = config.MAX_EPOCHS
        
        try:
            trainer = LSTMTrainer(seed_manager=self.seed_manager)
            vocab_size = trainer.load_data()
            trainer.initialize_model(
                vocab_size,
                # pretrained_checkpoint='checkpoints/pretrained.pt'
            )
            
            print(f"Starting LSTM training for {max_epochs} epochs...")
            print("This will take several hours depending on your hardware.")
            print()
            
            history = trainer.train(max_epochs)
            
            print(f"✓ LSTM training completed successfully!")
            print(f"✓ Best epoch: {trainer.best_epoch}")
            print(f"✓ Best test loss: {trainer.best_test_loss:.4f}")
            print()
            
            return True
            
        except Exception as e:
            print(f"✗ RNN training failed: {str(e)}")
            return False
    
    def run_chaos_analysis(self, max_analysis_epochs=config.MAX_EPOCHS):
        print("STEP 2: CHAOS ANALYSIS")
        print("-" * 40)
        
        try:
            analyzer = AnalysisRunner(seed_manager=self.seed_manager)
            analyzer.load_data_and_model()
            
            if not analyzer.load_training_history():
                print("✗ No training history found. Please run training first.")
                return False
            
            print(f"Starting chaos analysis for first {max_analysis_epochs} epochs...")
            print()
            
            epochs, *_ = analyzer.analyze_chaos_dynamics(
                start_epoch=config.START_EPOCH, 
                end_epoch=max_analysis_epochs, 
                interval=1
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
    
    def run_visualization(self):
        """Generate all visualization plots"""
        
        print("STEP 3: GENERATING VISUALIZATIONS")
        print("-" * 40)
        
        try:
            visualizer = ResultsVisualizer(seed_manager=self.seed_manager)
            visualizer.generate_all_plots()
            
            print("✓ All visualizations generated successfully!")
            print(f"✓ Plots saved to {os.path.join(config.RESULTS_PATH, 'figures')}")
            print()
            
            return True
            
        except Exception as e:
            print(f"✗ Visualization failed: {str(e)}")
            return False
    
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
        
        # Check what was completed
        files_to_check = [
            ('training_history.json', 'Training completed'),
            ('chaos_analysis_results.h5', 'Chaos analysis completed'),
            ('figures/training_curves.png', 'Visualizations generated')
        ]
        
        for filename, description in files_to_check:
            filepath = os.path.join(config.RESULTS_PATH, filename)
            status = "✓" if os.path.exists(filepath) else "✗"
            print(f"{status} {description}")
        
        print("="*80)

def main():
    parser = argparse.ArgumentParser(description='Run RNN Chaos Analysis Experiment')
    parser.add_argument('-te', type=int, default=config.MAX_EPOCHS,
                       help='short for training epochs, number of training epochs')
    parser.add_argument('-ae', type=int, default=config.MAX_EPOCHS,
                       help='short for analysis epochs, number of epochs to analyze')
    parser.add_argument('-st', action='store_true',
                       help='Skip training if results exist')
    parser.add_argument('-sa', action='store_true', 
                       help='Skip analysis if results exist')
    parser.add_argument('-sv', action='store_true',
                       help='Skip visualization generation')
    parser.add_argument('-qt', action='store_true',
                       help='Quick test run (50 train epochs, 20 analysis epochs)')
    
    args = parser.parse_args()
    
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
        success = success and runner.run_chaos_analysis(args.ae)
        if not success:
            print("Analysis failed. Stopping experiment.")
            return
    
    # Step 3: Visualization
    if not args.sv:
        success = success and runner.run_visualization()
    
    # Print summary
    runner.print_summary()
    
    if success:
        print("✓ Experiment completed successfully!")
    else:
        print("⚠ Experiment completed with some issues.")

if __name__ == "__main__":
    main()

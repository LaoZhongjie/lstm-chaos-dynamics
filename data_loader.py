"""
Data loading and preprocessing for IMDB Movie Review Dataset
Handles downloading, tokenization, and batch creation
"""

import os
import re
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from collections import Counter
from tqdm import tqdm
import config
from seed_utils import HierarchicalSeedManager

class IMDBDataset(Dataset):
    """Custom Dataset class for IMDB movie reviews"""
    
    def __init__(self, texts, labels, vocab, sequence_length=500):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.sequence_length = sequence_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Convert text to indices
        indices = self.text_to_indices(text)
        length = min(len(indices), self.sequence_length)
        
        # Pad or truncate to fixed length
        if len(indices) > self.sequence_length:
            indices = indices[:self.sequence_length]
        else:
            indices = indices + [0] * (self.sequence_length - len(indices))
            
        return (
            torch.tensor(indices, dtype=torch.long),
            torch.tensor(label, dtype=torch.float),
            torch.tensor(length, dtype=torch.long)
        )
    
    def text_to_indices(self, text):
        """Convert text to sequence of vocabulary indices"""
        # Simple tokenization
        words = text.lower().split()
        # Clean words (remove punctuation, keep only alphanumeric)
        words = [re.sub(r'[^a-zA-Z0-9]', '', word) for word in words]
        words = [word for word in words if word]  # Remove empty strings
        
        indices = []
        for word in words:
            if word in self.vocab:
                indices.append(self.vocab[word])
            else:
                indices.append(self.vocab['<UNK>'])  # Unknown token
        
        return indices

class IMDBDataLoader:
    """Data loader for IMDB dataset with preprocessing"""
    
    def __init__(self, seed_manager=None):
        self.vocab = None
        self.vocab_size = 0
        self.embedding_matrix = None
        self.seed_manager = seed_manager or HierarchicalSeedManager(config.RANDOM_SEED)
        
    def download_data(self):
        """Download IMDB dataset if not already present"""
        if not os.path.exists(config.DATA_PATH):
            os.makedirs(config.DATA_PATH)
            
        # Check if data already exists
        if os.path.exists(os.path.join(config.DATA_PATH, 'IMDB_Dataset.csv')):
            print("IMDB dataset already exists!")
            return
            
        print("Please download the IMDB dataset manually from:")
        print("https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")
        print("Or use the Hugging Face datasets library:")
        print("from datasets import load_dataset")
        print("dataset = load_dataset('imdb')")
        
        # Alternative: Load from Hugging Face datasets
        try:
            from datasets import load_dataset
            dataset = load_dataset('imdb')
            
            # Convert to pandas DataFrame
            train_df = pd.DataFrame({
                'review': dataset['train']['text'],
                'sentiment': ['positive' if label == 1 else 'negative' for label in dataset['train']['label']]
            })
            
            test_df = pd.DataFrame({
                'review': dataset['test']['text'],
                'sentiment': ['positive' if label == 1 else 'negative' for label in dataset['test']['label']]
            })
            
            # Combine and save
            full_df = pd.concat([train_df, test_df], ignore_index=True)
            full_df.to_csv(os.path.join(config.DATA_PATH, 'IMDB_Dataset.csv'), index=False)
            print("Dataset downloaded and saved successfully!")
            
        except ImportError:
            print("Please install the datasets library: pip install datasets")
            raise
    
    def build_vocabulary(self, texts, max_vocab_size=config.MAX_VOCAB_SIZE):
        """Build vocabulary from text data"""
        word_counts = Counter()
        
        for text in tqdm(texts, desc="Building vocabulary"):
            words = text.lower().split()
            words = [re.sub(r'[^a-zA-Z0-9]', '', word) for word in words]
            words = [word for word in words if word]
            word_counts.update(words)
        
        # Get most common words
        most_common = sorted(word_counts.items(), key=lambda x: (-x[1], x[0]))[:max_vocab_size - 2]

        # Create vocabulary mapping
        self.vocab = {'<PAD>': 0, '<UNK>': 1}
        for word, _ in most_common:
            self.vocab[word] = len(self.vocab)
        
        vocab_path = os.path.join(config.DATA_PATH, 'vocab.npy')
        np.save(vocab_path, self.vocab)
        print(f"Vocab saved to: {vocab_path}")
        
        self.vocab_size = len(self.vocab)
        print(f"Vocabulary size: {self.vocab_size}")
        
    def load_data(self):
        # Download data if needed
        self.download_data()
        
        df = pd.read_csv(os.path.join(config.DATA_PATH, 'IMDB_Dataset.csv'))
        
        # Check if pre-trained vocabulary exists
        vocab_path = os.path.join(config.DATA_PATH, 'vocab.npy')
        if os.path.exists(vocab_path):
            self.vocab = np.load(vocab_path, allow_pickle=True).item()
            self.vocab_size = len(self.vocab)
            print(f"Loaded pre-trained vocabulary: {self.vocab_size} words")
        else:
            # Load data and build vocabulary
            self.build_vocabulary(df['review'].values)
        
        # Convert sentiment to binary labels
        df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})
        
        # If vocabulary wasn't loaded, build it now
        if self.vocab is None:
            self.build_vocabulary(df['review'].values)
        
        # Split data
        split_seed = self.seed_manager.module_seed('data.split')
        X_train, X_test, y_train, y_test = train_test_split(
            df['review'].values,
            df['label'].values,
            test_size=config.TEST_RATIO,
            random_state=split_seed,
            stratify=df['label'].values
        )
        
        print(f"Training samples: {len(X_train)}")
        print(f"Testing samples: {len(X_test)}")
        
        return X_train, X_test, y_train, y_test
    
    def create_data_loaders(self, X_train, X_test, y_train, y_test):
        """Create PyTorch data loaders"""
        train_generator = self.seed_manager.torch_generator('dataloader.shuffle.train')
        test_generator = self.seed_manager.torch_generator('dataloader.shuffle.test')
    
        train_dataset = IMDBDataset(X_train, y_train, self.vocab, config.SEQUENCE_LENGTH)
        test_dataset = IMDBDataset(X_test, y_test, self.vocab, config.SEQUENCE_LENGTH)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=4,
            pin_memory=True if config.DEVICE == 'cuda' else False,
            generator=train_generator
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            pin_memory=True if config.DEVICE == 'cuda' else False,
            generator=test_generator
        )
        
        return train_loader, test_loader, test_dataset

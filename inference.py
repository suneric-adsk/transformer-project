from pathlib import Path
import torch 
import torch.nn as nn
from config import get_config, get_weights_file_path
from train import get_model, get_dataset, run_validation

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device {device}')
    
    config = get_config()
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_dataset(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

    model_filename = get_weights_file_path(config, f"29") 
    state = torch.load(model_filename)
    model.load_state_dict(state['model_state_dict'])

    run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: print(msg), None, num_examples=10)
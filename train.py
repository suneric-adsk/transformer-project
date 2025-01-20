import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from config import get_weights_file_path, get_config
from dataset import BilingualDataset, causal_mask
from model import build_transformer

from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
import warnings

# return a generator that yields all the sentences of target language in the dataset
def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]

# get or build the tokenizer for the given language
def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]')) # use WordLevel tokenizer
        tokenizer.pre_tokenizer = Whitespace()
        trianer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trianer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_dataset(config):
    lang_src, lang_tgt = config['lang_src'], config['lang_tgt']
    if (lang_src == 'en' and lang_tgt == 'zh') or (lang_src == 'zh' and lang_tgt == 'en'):   
        ds_raw = load_dataset("wmt19", "zh-en", split="train[1000:3000]")  # Use only 5% of the training data
    else:
        ds_raw = load_dataset('opus_books', f'{lang_src}-{lang_tgt}', split='train[:20%]') # Use only 20% of the training data
    
    # Build tokenizers
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, lang_src)
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, lang_tgt)

    # split dataset to random train and validation sets
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, lang_src, lang_tgt, config['seq_len'])
    valid_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, lang_src, lang_tgt, config['seq_len'])

    # (Optional) Find the maximum length of the source and target sentences
    max_len_src = 0
    max_len_tgt = 0
    for item in ds_raw:
        # print(item)
        src_ids = tokenizer_src.encode(item['translation'][lang_src]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][lang_tgt]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))
    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_tgt}')
    
    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(valid_ds, batch_size=1, shuffle=True) # validate one sentence at a time 

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

# Validation with greedy decoding
def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    encoder_output = model.encode(source, source_mask)
    decoder_input = torch.empty(1,1).fill_(sos_idx).type_as(source).to(device) # initialize the decoder input with SOS token
    while True:
        if decoder_input.size(1) == max_len:
            break

        # Build mask for the target sequence
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
        # Calculate the decoder output
        output = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)
        prob = model.project(output[:,-1])
        _, next_word = torch.max(prob, dim=1) # greedy decoding
        decoder_input = torch.cat([decoder_input, torch.empty(1,1).type_as(source).fill_(next_word.item()).to(device)], dim=1)

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)

def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_state, writer, num_examples=5):
    model.eval()
    count = 0
    console_width = 80
    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch['encoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)

            assert encoder_input.size(0) == 1, "batch size must be 1 for validation"    

            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

            source_text = batch['src_text'][0]
            target_text = batch['tgt_text'][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            print_msg('-'*console_width)
            print_msg(f'SOURCE: {source_text}')
            print_msg(f'TARGET: {target_text}')
            print_msg(f'PREDICTED: {model_out_text}')

            if count == num_examples:
                break
    return

# The training loop
def train_model(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device {device}')

    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    # Tensorboard
    writer = SummaryWriter(config['experiment_name'])

    # dataset for translation
    seq_len, d_model = config['seq_len'], config['d_model']
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_dataset(config)
    src_vocab_size, tgt_vocab_size = tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()

    # transformer model
    model = build_transformer(src_vocab_size, tgt_vocab_size, seq_len, seq_len, d_model).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr = config['lr'], eps=1e-9)
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    # Preload model
    initial_epoch = 0
    global_step = 0
    if config['preload']:
        model_filename = get_weights_file_path(config, config['preload'])
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']

    # training loop
    for epoch in range(initial_epoch, config['num_epochs']):
        batch_iterator = tqdm(train_dataloader, desc=f'Processing epoch {epoch}')
        for batch in batch_iterator:
            model.train()
            
            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)
            label = batch['label'].to(device)

            # transformer networks
            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
            proj_output = model.project(decoder_output) # (batch_size, seq_len, vocab_size)

            # computer loss
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))

            batch_iterator.set_postfix({f'loss': f"{loss.item():6.3f}"})
            writer.add_scalar('training loss', loss.item(), global_step)
            writer.flush()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
        
        run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, seq_len, device, lambda msg: batch_iterator.write(msg), global_step, writer)
        
        model_filename = get_weights_file_path(config, f'{epoch:02d}')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    config = get_config()
    train_model(config)
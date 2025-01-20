import torch
import torch.nn as nn
from torch.utils.data import Dataset

# load dataset and use the english to chinese translation

def causal_mask(size):
    # value above the diagonal should be masked out
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask == 0

class BilingualDataset(Dataset):
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, lang_src, lang_tgt, seq_len):
        super(BilingualDataset, self).__init__()

        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.lang_src = lang_src
        self.lang_tgt = lang_tgt
        self.seq_len = seq_len

        self.sos_token = torch.tensor([tokenizer_src.token_to_id('[SOS]')], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_src.token_to_id('[EOS]')], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_src.token_to_id('[PAD]')], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        src_tgt_pair = self.ds[index]
        src_text = src_tgt_pair['translation'][self.lang_src]
        tgt_text = src_tgt_pair['translation'][self.lang_tgt]
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2 # -2 for SOS and EOS tokens
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1 # -1 for EOS token

        # padding tokens should not be negative
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError('Sequence is too long')
        
        # add SOS and EOS to the source text
        encoder_input = torch.cat(
            [
                self.sos_token, 
                torch.tensor(enc_input_tokens, dtype=torch.int64), 
                self.eos_token, 
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64)
            ]
        )

        # add SOS tokens the target text
        decoder_input = torch.cat(
            [   
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64), 
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
            ]
        )

        # add EOS token to the label text (expected output from the decoder) 
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64), 
                self.eos_token, 
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
            ]
        )

        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input, # (seq_len)
            "decoder_input": decoder_input, # (seq_len)
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1, 1, seq_len) 
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), # (1, seq_len) & (1, seq_len, seq_len)
            "label": label, # (seq_len)
            "src_text": src_text,
            "tgt_text": tgt_text
        }
    



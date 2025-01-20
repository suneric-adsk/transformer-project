# This is the implementation of Transformer using PyTorch
# The goal is to gain a deep understanding of the Transformer model, mapping the theory to the code.
# Some of the reference are
# [1] https://nlp.seas.harvard.edu/2018/04/03/attention.html
# [2] https://www.bing.com/videos/riverview/relatedvideo?q=How+to+prepare+data+for+Transformer+training&mid=33F83BD072C84C53BE9633F83BD072C84C53BE96&mcid=C4CF52FAD6234B108C2DD5923F982E91&FORM=VIRE

# Importing the required libraries
import torch
import torch.nn as nn
import math

# Input Embedding
# Converting original sentences to embeddings (position of each words in the vacabulary)
class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super(InputEmbeddings, self).__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model) # embedding layer

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model) # scale the embeddings by sqrt(d_model) (as per the paper)

# Positional Encoding
# Adding the positional encoding to the input embeddings
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        
        # create positional encoding matrix (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model) # initialize the positional encoding matrix
        # create a vector of shape (seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) 
        # use log scale for learning stablity
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) 
        # apply the sin to even positions and cos to odd positions
        pe[:, 0::2] = torch.sin(position * div_term) # even indices
        pe[:, 1::2] = torch.cos(position * div_term) # odd indices
        pe = pe.unsqueeze(0) # add a new dimension to pe for accomadating batch -> (1, seq_len, d_model)
        self.register_buffer('pe', pe) # keep inside the model but not as learned parameters
    
    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) # make the particular tensor not learnable
        return self.dropout(x)

# Layer Normalization
# Can be replaced use nn.LayerNorm() in PyTorch
class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super(LayerNormalization, self).__init__()
        self.eps = eps # epsilon value for numerical stability
        self.alpha = nn.Parameter(torch.ones(1)) # learnable parameters weight vector
        self.bias = nn.Parameter(torch.zeros(1)) # bias vector

    def forward(self, x):
        # calculate mean and standard deviation, and normalize
        mean = x.mean(dim=-1, keepdim=True) # (batch, seq_len, 1)
        std = x.std(dim=-1, keepdim=True) # (batch, seq_len, 1)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias # (batch, seq_len, d_model)
    
# Feed Forward Block
# 
class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super(FeedForwardBlock, self).__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # W1 and b1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # W2 and b2

    def forward(self, x):
        # apply first linear transformation
        # (batch, seq_len, d_model) -> (batch, seq_len, d_ff) -> (batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x)))) 

# Multi-Head Attention Block
# 
class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super(MultiHeadAttentionBlock, self).__init__()
        self.d_model = d_model # The dimension of the model
        self.n_heads = n_heads # The number of heads
        assert d_model % n_heads == 0, "d_model should be divisible by n_heads"

        self.d_k = d_model // n_heads # The dimension of the key and value in each head

        self.w_q = nn.Linear(d_model, d_model) # Wq
        self.w_k = nn.Linear(d_model, d_model) # Wk
        self.w_v = nn.Linear(d_model, d_model) # Wv
        
        self.w_o = nn.Linear(d_model, d_model) # Wo
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def scaled_dot_product_attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        # (batch, n_heads, seq_len, d_k) x (batch, n_heads, d_k, seq_len) -> (batch, n_heads, seq_len, seq_len)
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        
        # apply mask if provided (useful for preventing attention to certain parts like padding)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        # apply softmax to get the attention probabilities (batch, n_heads, seq_len, seq_len)
        attn_probs = attn_scores.softmax(dim=-1) 
        
        # apply dropout if provided
        if dropout is not None:
            attn_probs = dropout(attn_probs)
        
        # multiply by value to get the final attention output
        # (batch, n_heads, seq_len, seq_len) x (batch, n_heads, seq_len, d_k) -> (batch, n_heads, seq_len, d_k)
        return torch.matmul(attn_probs, value), attn_probs 

    def forward(self, q, k, v, mask):
        # apply linear transformation
        query = self.w_q(q) # (batch, seq_len, d_model) -> (batch, seq_len, d_model)
        key = self.w_k(k) # (batch, seq_len, d_model) -> (batch, seq_len, d_model)
        value = self.w_v(v) # (batch, seq_len, d_model) -> (batch, seq_len, d_model)

        # split heads, reshaping the input to have n_heads for multi-head attention
        # (batch, seq_len, d_model) -> (batch, seq_len, n_heads, d_k) -> (batch, n_heads, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.n_heads, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.n_heads, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.n_heads, self.d_k).transpose(1, 2)

        # perform scaled dot-product attention
        x, self.attn_probs = MultiHeadAttentionBlock.scaled_dot_product_attention(query, key, value, mask, self.dropout)

        # combine the heads
        # (batch, n_heads, seq_len, d_k) -> (batch, seq_len, n_heads, d_k) -> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.n_heads*self.d_k) 

        # apply output linear trasnformation
        # (batch, seq_len, d_model) -> (batch, seq_len, d_model)
        return self.w_o(x)

# Residual Connection
#
class ResidualConnection(nn.Module):
    def __init__(self, dropout: float):
        super(ResidualConnection, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()
        
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

# Encoder block and Encoder
# 
class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float):
        super(EncoderBlock, self).__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection_1 = ResidualConnection(dropout)
        self.residual_connection_2 = ResidualConnection(dropout)

    # apply the src mask to exclude the [pad] words
    def forward(self, x, src_mask):
        x = self.residual_connection_1(x, lambda x: self.self_attention_block(x, x, x, src_mask)) # self attention block of the encoder
        x = self.residual_connection_2(x, self.feed_forward_block)
        return x

class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super(Encoder, self).__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

# Decoder block and Decoder
# 
class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, 
                 feed_forward_block: FeedForwardBlock, dropout: float):
        super(DecoderBlock, self).__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection_1 = ResidualConnection(dropout)
        self.residual_connection_2 = ResidualConnection(dropout)
        self.residual_connection_3 = ResidualConnection(dropout)

    # two masks are used, one for the target and one for the source (e.g. translation task)
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connection_1(x, lambda x: self.self_attention_block(x, x, x, tgt_mask)) # self attention block of the decoder
        x = self.residual_connection_2(x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connection_3(x, self.feed_forward_block)
        return x  
    
class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super(Decoder, self).__init__()
        self.layers = layers
        self.norm = LayerNormalization()
    
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)
    
# Projection Layer
# the linear layer before output, projecting the embeddings to the vocabulary size
class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super(ProjectionLayer, self).__init__()
        self.proj = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        # (batch, seq_len, d_model) -> (batch, seq_len, vocab_size)
        # apply log softmax for numerical stability, instead of softmax
        return torch.log_softmax(self.proj(x), dim = -1)

# Transformer
#
class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, 
                 src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, proj_layer: ProjectionLayer):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.proj_layer = proj_layer
    
    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x):
        return self.proj_layer(x)

# Build the Transformer
def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int = 512, 
                      h: int = 8, N: int = 6, dropout: float = 0.1, d_ff: int = 2048):
    # create the embedding layers 
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # create the positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)
    
    # create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_blcok = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_blcok)
    
    # create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)
    
    # create the encoder and the decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))
    
    # create the projection layer
    proj_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # create the transformer model
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, proj_layer)

    # Initialize the weights
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer


from typing import Dict, List, Optional, Union, Tuple, Callable
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from base_bert import BertPreTrainedModel
from utils import *


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
    def __init__(self, config):
        super().__init__()

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # initialize the linear transformation layers for key, value, query
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        # this dropout is applied to normalized attention scores following the original implementation of transformer
        # although it is a bit unusual, we empirically observe that it yields better performance
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        # initialize the linear transformation layers for key, value, query
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        # this dropout is applied to normalized attention scores following the original implementation of transformer
        # although it is a bit unusual, we empirically observe that it yields better performance
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transform(self, x, linear_layer):
        # the corresponding linear_layer of k, v, q are used to project the hidden_state (x)
        bs, seq_len = x.shape[:2]
        proj = linear_layer(x)
        # next, we need to produce multiple heads for the proj
        # this is done by spliting the hidden state to self.num_attention_heads, each of size self.attention_head_size
        proj = proj.view(
            bs, seq_len, self.num_attention_heads, self.attention_head_size
        )
        # by proper transpose, we have proj of [bs, num_attention_heads, seq_len, attention_head_size]
        proj = proj.transpose(1, 2)
        return proj

    def attention(self, key, query, value, attention_mask):
        # each attention is calculated following eq (1) of https://arxiv.org/pdf/1706.03762.pdf
        # attention scores are calculated by multiply query and key
        # and get back a score matrix S of [bs, num_attention_heads, seq_len, seq_len]
        # S[*, i, j, k] represents the (unnormalized)attention score between the j-th and k-th token, given by i-th attention head
        # before normalizing the scores, use the attention mask to mask out the padding token scores
        # Note again: in the attention_mask non-padding tokens with 0 and padding tokens with a large negative number
    def attention(self, key, query, value, attention_mask):
        # each attention is calculated following eq (1) of https://arxiv.org/pdf/1706.03762.pdf
        # attention scores are calculated by multiply query and key
        # and get back a score matrix S of [bs, num_attention_heads, seq_len, seq_len]
        # S[*, i, j, k] represents the (unnormalized)attention score between the j-th and k-th token, given by i-th attention head
        # before normalizing the scores, use the attention mask to mask out the padding token scores
        # Note again: in the attention_mask non-padding tokens with 0 and padding tokens with a large negative number

        # normalize the scores
        # multiply the attention scores to the value and get back V'
        # next, we need to concat multi-heads and recover the original shape [bs, seq_len, num_attention_heads * attention_head_size = hidden_size]
        # normalize the scores
        # multiply the attention scores to the value and get back V'
        # next, we need to concat multi-heads and recover the original shape [bs, seq_len, num_attention_heads * attention_head_size = hidden_size]

        ### TODO
        # 获取键值张量的维度大小
        d_k = key.size(-1)

        # 计算注意力分数矩阵
        # 使用矩阵乘法计算 query 和 key 的点积，除以 sqrt(d_k) 进行缩放
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        # 应用注意力掩码，将填充位置的分数设置为一个非常小的值
        attention_scores += attention_mask

        # 对注意力分数应用 softmax 函数，获得归一化的注意力权重
        attention_weights = F.softmax(attention_scores, dim=-1)

        # 使用注意力权重加权 value，得到注意力输出
        attention_output = torch.matmul(
            attention_weights, value
        )  # [bs, num_attention_heads, seq_len, attention_head_size]

        # 转置注意力输出的张量，以便拼接多头输出
        attention_output = attention_output.transpose(1, 2)

        # 获取批次大小和序列长度
        batch_size, seq_len = attention_output.size(0), attention_output.size(1)

        # 重塑注意力输出张量，恢复到原始形状
        attention_output = attention_output.reshape(
            batch_size, seq_len, self.num_attention_heads * self.attention_head_size
        )

        return attention_output

    def forward(self, hidden_states, attention_mask):
        """
        hidden_states: [bs, seq_len, hidden_state]
        attention_mask: [bs, 1, 1, seq_len]
        output: [bs, seq_len, hidden_state]
        """
        # first, we have to generate the key, value, query for each token for multi-head attention w/ transform (more details inside the function)
        # of *_layers are of [bs, num_attention_heads, seq_len, attention_head_size]
        key_layer = self.transform(hidden_states, self.key)
        value_layer = self.transform(hidden_states, self.value)
        query_layer = self.transform(hidden_states, self.query)
        # calculate the multi-head attention
        attn_value = self.attention(key_layer, query_layer, value_layer, attention_mask)
        return attn_value


class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # multi-head attention
        self.self_attention = BertSelfAttention(config)
        # add-norm
        self.attention_dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.attention_layer_norm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.attention_dropout = nn.Dropout(config.hidden_dropout_prob)
        # feed forward
        self.interm_dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.interm_af = F.gelu
        # another add-norm
        self.out_dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.out_layer_norm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.out_dropout = nn.Dropout(config.hidden_dropout_prob)

    def add_norm(self, input, output, dense_layer, dropout, ln_layer):
        """
        this function is applied after the multi-head attention layer or the feed forward layer
        input: the input of the previous layer
        output: the output of the previous layer
        dense_layer: used to transform the output
        dropout: the dropout to be applied 
        ln_layer: the layer norm to be applied
        """
        # Hint: Remember that BERT applies to the output of each sub-layer, before it is added to the sub-layer input and normalized
        ### TODO
        # 对前一层的输出应用全连接层
        transformed_output = dense_layer(output)

        # 对全连接层的输出应用 dropout
        dropout_output = dropout(transformed_output)

        # 将处理后的输出与原始输入相加
        residual_output = input + dropout_output

        # 对相加后的结果应用层归一化
        add_norm_output = ln_layer(residual_output)

        return add_norm_output

    def forward(self, hidden_states, attention_mask):
        """
        hidden_states: either from the embedding layer (first bert layer) or from the previous bert layer
        as shown in the left of Figure 1 of https://arxiv.org/pdf/1706.03762.pdf
        each block consists of
        1. a multi-head attention layer (BertSelfAttention)
        2. a add-norm that takes the input and output of the multi-head attention layer
        3. a feed forward layer
        4. a add-norm that takes the input and output of the feed forward layer
        """
        ### TODO
        # attention
        output = self.self_attention.forward(hidden_states, attention_mask)
        # add and norm
        output = self.add_norm(
            hidden_states,
            output,
            self.attention_dense,
            self.attention_dropout,
            self.attention_layer_norm,
        )
        # feed forward
        input = output
        output = self.interm_dense(input)
        output = self.interm_af(output)
        # add and norm
        output = self.add_norm(
            input, output, self.out_dense, self.out_dropout, self.out_layer_norm
        )

    def forward(self, hidden_states, attention_mask):
        """
        hidden_states: either from the embedding layer (first bert layer) or from the previous bert layer
        as shown in the left of Figure 1 of https://arxiv.org/pdf/1706.03762.pdf 
        each block consists of 
        1. a multi-head attention layer (BertSelfAttention)
        2. a add-norm that takes the input and output of the multi-head attention layer
        3. a feed forward layer
        4. a add-norm that takes the input and output of the feed forward layer
        """
        ### TODO
        # 1. 多头自注意力层
        attention_output = self.self_attention.forward(hidden_states, attention_mask)

        # 2. 加法和归一化（Add & Norm）
        normed_attention_output = self.add_norm(
            hidden_states,
            attention_output,
            self.attention_dense,
            self.attention_dropout,
            self.attention_layer_norm,
        )

        # 3. 前馈层
        feed_forward_input = normed_attention_output
        feed_forward_output = self.interm_dense(feed_forward_input)
        feed_forward_output = self.interm_af(feed_forward_output)

        # 4. 加法和归一化（Add & Norm）
        final_output = self.add_norm(
            feed_forward_input,
            feed_forward_output,
            self.out_dense,
            self.out_dropout,
            self.out_layer_norm,
        )

        return final_output


class BertModel(BertPreTrainedModel):
    """
    the bert model returns the final embeddings for each token in a sentence
    it consists
    1. embedding (used in self.embed)
    2. a stack of n bert layers (used in self.encode)
    3. a linear transformation layer for [CLS] token (used in self.forward, as given)
    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        # embedding
        self.word_embedding = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.pos_embedding = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.tk_type_embedding = nn.Embedding(
            config.type_vocab_size, config.hidden_size
        )
        self.embed_layer_norm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.embed_dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is a constant, register to buffer
        position_ids = torch.arange(config.max_position_embeddings).unsqueeze(0)
        self.register_buffer("position_ids", position_ids)

        # bert encoder
        self.bert_layers = nn.ModuleList(
            [BertLayer(config) for _ in range(config.num_hidden_layers)]
        )

        # for [CLS] token
        self.pooler_dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.pooler_af = nn.Tanh()

        self.init_weights()

    def embed(self, input_ids):
        input_shape = input_ids.size()
        seq_length = input_shape[1]

        # Get word embedding from self.word_embedding into input_embeds.
        ### TODO
        inputs_embeds = self.word_embedding(input_ids)

        # Get position index and position embedding from self.pos_embedding into pos_embeds.
        pos_ids = self.position_ids[:, :seq_length]

        ### TODO
        pos_embeds = self.pos_embedding(pos_ids)

        # Get token type ids, since we are not consider token type, just a placeholder.
        tk_type_ids = torch.zeros(
            input_shape, dtype=torch.long, device=input_ids.device
        )
        tk_type_embeds = self.tk_type_embedding(tk_type_ids)

        # Add three embeddings together; then apply embed_layer_norm and dropout and return.
        ### TODO
        # 将三个嵌入相加
        combined_embeds = inputs_embeds + pos_embeds + tk_type_embeds
        # 应用层归一化和 dropout
        normalized_embeds = self.embed_layer_norm(combined_embeds)
        final_embeds = self.embed_dropout(normalized_embeds)
        return final_embeds

    def encode(self, hidden_states, attention_mask):
        """
        the bert model returns the final embeddings for each token in a sentence
        it consists
        1. embedding (used in self.embed)
        2. a stack of n bert layers (used in self.encode)
        3. a linear transformation layer for [CLS] token (used in self.forward, as given)
        """

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        # embedding
        self.word_embedding = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.pos_embedding = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.tk_type_embedding = nn.Embedding(
            config.type_vocab_size, config.hidden_size
        )
        self.embed_layer_norm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.embed_dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is a constant, register to buffer
        position_ids = torch.arange(config.max_position_embeddings).unsqueeze(0)
        self.register_buffer("position_ids", position_ids)

        # bert encoder
        self.bert_layers = nn.ModuleList(
            [BertLayer(config) for _ in range(config.num_hidden_layers)]
        )

        # for [CLS] token
        self.pooler_dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.pooler_af = nn.Tanh()
        # for [CLS] token
        self.pooler_dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.pooler_af = nn.Tanh()

        self.init_weights()
        self.init_weights()

    def embed(self, input_ids):
        input_shape = input_ids.size()
        seq_length = input_shape[1]
    def embed(self, input_ids):
        input_shape = input_ids.size()
        seq_length = input_shape[1]

        # Get word embedding from self.word_embedding into input_embeds.
        inputs_embeds = None
        ### TODO
        inputs_embeds = self.word_embedding(input_ids)
        # Get word embedding from self.word_embedding into input_embeds.
        inputs_embeds = None
        ### TODO
        inputs_embeds = self.word_embedding(input_ids)

        # Get position index and position embedding from self.pos_embedding into pos_embeds.
        pos_ids = self.position_ids[:, :seq_length]
        # Get position index and position embedding from self.pos_embedding into pos_embeds.
        pos_ids = self.position_ids[:, :seq_length]

        pos_embeds = None
        ### TODO
        pos_embeds = self.pos_embedding(pos_ids)
        pos_embeds = None
        ### TODO
        pos_embeds = self.pos_embedding(pos_ids)

        # Get token type ids, since we are not consider token type, just a placeholder.
        tk_type_ids = torch.zeros(
            input_shape, dtype=torch.long, device=input_ids.device
        )
        tk_type_embeds = self.tk_type_embedding(tk_type_ids)

        # Add three embeddings together; then apply embed_layer_norm and dropout and return.
        ### TODO
        embeds = inputs_embeds.add(pos_embeds.add(tk_type_embeds))
        embeds = self.embed_layer_norm(embeds)
        embeds = self.embed_dropout(embeds)
        return embeds
        # Add three embeddings together; then apply embed_layer_norm and dropout and return.
        ### TODO
        embeds = inputs_embeds.add(pos_embeds.add(tk_type_embeds))
        embeds = self.embed_layer_norm(embeds)
        embeds = self.embed_dropout(embeds)
        return embeds

    def encode(self, hidden_states, attention_mask):
        """
        hidden_states: the output from the embedding layer [batch_size, seq_len, hidden_size]
        attention_mask: [batch_size, seq_len]
        """
        # get the extended attention mask for self attention
        # returns extended_attention_mask of [batch_size, 1, 1, seq_len]
        # non-padding tokens with 0 and padding tokens with a large negative number
        extended_attention_mask: torch.Tensor = get_extended_attention_mask(
            attention_mask, self.dtype
        )

        # pass the hidden states through the encoder layers
        for i, layer_module in enumerate(self.bert_layers):
            # feed the encoding from the last bert_layer to the next
            hidden_states = layer_module(hidden_states, extended_attention_mask)
        # pass the hidden states through the encoder layers
        for i, layer_module in enumerate(self.bert_layers):
            # feed the encoding from the last bert_layer to the next
            hidden_states = layer_module(hidden_states, extended_attention_mask)

        return hidden_states
        return hidden_states

    def forward(self, input_ids, attention_mask):
        """
        input_ids: [batch_size, seq_len], seq_len is the max length of the batch
        attention_mask: same size as input_ids, 1 represents non-padding tokens, 0 represents padding tokens
        """
        # get the embedding for each input token
        embedding_output = self.embed(input_ids=input_ids)
    def forward(self, input_ids, attention_mask):
        """
        input_ids: [batch_size, seq_len], seq_len is the max length of the batch
        attention_mask: same size as input_ids, 1 represents non-padding tokens, 0 represents padding tokens
        """
        # get the embedding for each input token
        embedding_output = self.embed(input_ids=input_ids)

        # feed to a transformer (a stack of BertLayers)
        sequence_output = self.encode(embedding_output, attention_mask=attention_mask)
        # feed to a transformer (a stack of BertLayers)
        sequence_output = self.encode(embedding_output, attention_mask=attention_mask)

        # get cls token hidden state
        first_tk = sequence_output[:, 0]
        first_tk = self.pooler_dense(first_tk)
        first_tk = self.pooler_af(first_tk)
        # get cls token hidden state
        first_tk = sequence_output[:, 0]
        first_tk = self.pooler_dense(first_tk)
        first_tk = self.pooler_af(first_tk)

        return {"last_hidden_state": sequence_output, "pooler_output": first_tk}

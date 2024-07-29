import torch
import torch.nn as nn
import math

class Gemma2RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        norm_x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm_x * (1 + self.weight.float()).type_as(x)


class Gemma2RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim))
        self.register_buffer("inv_freq", inv_freq)

    @torch.no_grad()
    def forward(self, x, position_ids, seq_len=None):
        self.inv_freq = self.inv_freq.to(x.device)
        position_ids = position_ids[:, None, :].float()
        inv_freq = self.inv_freq[None, :, None].expand(position_ids.shape[0], -1, x.shape[2])
        freqs = torch.matmul(inv_freq, position_ids)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Gemma2Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.scaling = config.query_pre_attn_scalar ** -0.5

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        self.rotary_emb = Gemma2RotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings)
        self.sliding_window = config.sliding_window

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,
    ):
        batch_size, seq_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        key_states, value_states = repeat_kv(key_states, self.num_key_value_groups), repeat_kv(
            value_states, self.num_key_value_groups
        )

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "sliding_window": self.sliding_window, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, cache_kwargs)

        qk_diff = (query_states * self.scaling).matmul(key_states.permute(0, 1, 3, 2))

        if self.config.attn_logit_softcapping is not None:
            qk_diff = torch.tanh(qk_diff / self.config.attn_logit_softcapping) * self.config.attn_logit_softcapping

        if attention_mask is not None:
            qk_diff += attention_mask

        attention_probs = torch.softmax(qk_diff, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attention_probs = torch.nn.functional.dropout(attention_probs, p=self.attention_dropout)
        attention_states = attention_probs.matmul(value_states)

        attention_states = attention_states.permute(0, 2, 1, 3).contiguous()
        attention_states = attention_states.view(batch_size, seq_len, self.num_heads * self.head_dim)

        attention_states = self.o_proj(attention_states)

        return attention_states


def repeat_kv(hidden_states, n_rep):
    batch, num_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_heads * n_rep, slen, head_dim)


class Gemma2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = torch.nn.functional.gelu

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Gemma2DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size

        self.attn = Gemma2Attention(config=config)
        self.attn_norm = Gemma2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp_norm = Gemma2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp_norm2 = Gemma2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = Gemma2MLP(config)
        self.sliding_window = config.sliding_window
        self.layer_idx = layer_idx

    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None, use_cache=False):
        hidden_states = self.attn_norm(hidden_states)
        hidden_states, _ = self.attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = hidden_states + self.mlp_norm(hidden_states)
        hidden_states = self.mlp(self.mlp_norm2(hidden_states)) + hidden_states
        return hidden_states


class Gemma2Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size

        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.norm = Gemma2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.decoder_layers = nn.ModuleList([Gemma2DecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.sliding_window_size = config.sliding_window

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        hidden_states = self.embedding(input_ids) if input_ids is not None else inputs_embeds
        hidden_states = hidden_states * math.sqrt(self.config.hidden_size)

        all_hidden_states = []
        if output_hidden_states:
            all_hidden_states.append(hidden_states)

        all_attention_weights = []

        batch_size = hidden_states.shape[0]
        if past_key_value is not None and past_key_value[0] is not None:
            target_length = past_key_value[0][0].shape[2]
        else:
            target_length = hidden_states.shape[1]

        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 0)

        # `min_dtype` is used for avoiding numeric overflow of attention masks
        min_dtype = torch.half
        prev_dtype = None
        for layer in self.decoder_layers:
            past_key_value_layer = (
                past_key_value[layer.layer_idx] if past_key_value is not None and layer.layer_idx < len(past_key_value) else None
            )

            if attention_mask is not None and not layer.layer_idx % 2 == 0:
                min_dtype = attention_mask.dtype
                causal_mask = torch.ones(batch_size, hidden_states.size(1), target_length, dtype=attention_mask.dtype).to(
                    hidden_states.device
                )
                # `rotary_embeddings` is a CPU-only operation
                causal_mask = causal_mask.type(min_dtype)
                causal_mask[:, :-1, :] = causal_mask[:, :-1, :] + causal_mask[:, 1:, :] * (attention_mask[:, 1:] == min_dtype)
                causal_mask = (
                    causal_mask.reshape(
                        (
                            batch_size,
                            hidden_states.size(1),
                            self.sliding_window_size,
                            target_length // self.sliding_window_size,
                        )
                    )
                   .permute(0, 2, 1, 3)
                   .reshape(batch_size, hidden_states.size(1), target_length)
                )

            hidden_states = layer(
                hidden_states=hidden_states,
                attention_mask=causal_mask if layer.layer_idx % 2 == 0 else attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value_layer,
                use_cache=use_cache,
            )

            if output_hidden_states:
                all_hidden_states.append(hidden_states)
            if output_attentions:
                all_attention_weights.append(attn_weights)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states.append(hidden_states)

        if output_attentions:
            attention_weights = torch.cat(all_attention_weights, dim=2)
            all_attention_weights.append(attention_weights)

        hidden_states = hidden_states.contiguous()

        return tuple(
            v
            for v in [
                hidden_states,
                all_hidden_states if output_hidden_states else None,
                all_attention_weights if output_attentions else None,
            ]
            if v is not None
        )


class Gemma2ForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = Gemma2Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        labels=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        if self.config.final_logit_softmax_cap is not None:
            logits = torch.tanh(logits / self.config.final_logit_softmax_cap) * self.config.final_logit_softmax_cap

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            loss = torch.nn.functional.cross_entropy(shift_logits, shift_labels)

        outputs = (logits,) + outputs[1:]
        return (loss,) + outputs if loss else outputs


def initialize_weights(module):
    if isinstance(module, torch.nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=std)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, torch.nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=std)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    elif isinstance(module, torch.nn.LayerNorm):
        torch.nn.init.zeros_(module.bias)
        torch.nn.init.ones_(module.weight)


# Initialize the model weights
std = 0.02
model = Gemma2ForCausalLM(config)
model.apply(initialize_weights)

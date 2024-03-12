# Copyright (c) Alibaba Cloud.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from transformers import PretrainedConfig


class Qwen2Config(PretrainedConfig):
    model_type = "qwen"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=5000,
        hidden_size=1024,
        num_hidden_layers=24,
        num_attention_heads=16,
        emb_dropout_prob=0.0,
        attention_dropout=0.0,
        layer_norm_epsilon=1e-6,
        initializer_range=0.02,
        max_position_embeddings=8192,
        scale_attn_weights=True,
        use_cache=True,
        bf16=False,
        fp16=False,
        fp32=True,
        kv_channels=128,
        num_key_value_heads=16,
        rotary_pct=1.0,
        rotary_emb_base=10000,
        use_dynamic_ntk=True,
        hidden_act="silu",
        use_logn_attn=True,
        use_flash_attn="auto",
        intermediate_size=2816,
        no_bias=True,
        tie_word_embeddings=False,
        use_cache_quantization=False,
        use_cache_kernel=False,
        use_sliding_window=False,
        sliding_window=32768,
        softmax_in_fp32=False,
        rope_theta=1000000.0,
        rms_norm_eps=1e-6,
        max_window_layers=21,
        pad_token_id=4999,
        bos_token_id=4999,
        eos_token_id=4999,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.emb_dropout_prob = emb_dropout_prob
        self.attention_dropout = attention_dropout
        self.rms_norm_eps = rms_norm_eps
        self.max_window_layers = max_window_layers
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.scale_attn_weights = scale_attn_weights
        self.use_cache = use_cache
        self.max_position_embeddings = max_position_embeddings
        self.seq_length = max_position_embeddings
        self.rope_theta = rope_theta
        self.bf16 = bf16
        self.fp16 = fp16
        self.fp32 = fp32
        self.kv_channels = kv_channels
        self.rotary_pct = rotary_pct
        self.rotary_emb_base = rotary_emb_base
        self.use_dynamic_ntk = use_dynamic_ntk
        self.use_logn_attn = use_logn_attn
        self.use_flash_attn = use_flash_attn
        self.num_key_value_heads = num_key_value_heads
        self.sliding_window = sliding_window
        self.hidden_act = hidden_act
        self.no_bias = no_bias
        self.use_cache_quantization = use_cache_quantization
        self.use_cache_kernel = use_cache_kernel
        self.softmax_in_fp32 = softmax_in_fp32
        self.use_sliding_window = use_sliding_window
        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs
        )
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id


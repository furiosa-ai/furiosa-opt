use furiosa_visa_std::prelude::*;

axes![
    D = 64,           // head_dim
    G = 7,            // gqa_ratio = 14 / 2
    H = 896,          // hidden_size = 14 * 64
    I = 2,            // interleave (ClipAdd dual-input)
    K = 128,          // kv_proj_size = N * D
    M = 4864,         // mlp_intermediate_size
    N = 2,            // num_kv_heads
    P = 32768,        // max_position
    Q = 896,          // q_proj_size
    R = 2,            // rope_rot
    S_decode = 128,   // sequence length (decode)
    S_prefill = 1024, // query sequence length (prefill)
    T = 1024,         // key/value sequence length
    V = 128,          // v_proj_size
    W_vocab = 151936, // vocab_size
    X = 16,           // broadcast
    Y = 4,            // broadcast
    Z = 2,            // broadcast
    C_kvcache = 1124, // kv_cache_len
    C_lmhead = 8192,  // lm_head weight chunk
    W_kbcast = 32,    // broadcast (K replication in attn_weight)
    W_norm = 64,      // broadcast (decoder norm weight, S_decode/2 replication)
];

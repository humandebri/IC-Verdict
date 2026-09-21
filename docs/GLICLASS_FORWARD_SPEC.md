# GLiClass / ModernBERT forward-pass specification
### Target checkpoint: `heman10x/rlcd-modernbert-151m` (frozen revision sha `70fa19828074e1199e4a793c4af4dba3bfd1d222`)
### Base: `knowledgator/gliclass-modern-base-v2.0` → ModernBERT-base + GLiClass `uni-encoder` head

All facts below carry a citation to the exact file/line, URL, or byte offset from which they were read.
Primary evidence sources:

| Tag | Source |
|---|---|
| `[GLI-MODEL]` | `https://raw.githubusercontent.com/knowledgator/GLiClass/main/gliclass/model.py` (tree sha `40baa67…` == tag `v0.1.20`) |
| `[GLI-SCORE]` | `…/gliclass/scorers.py` (same tree) |
| `[GLI-POOL]` | `…/gliclass/poolings.py` |
| `[GLI-CFG]` | `…/gliclass/config.py` |
| `[GLI-LAYERS]` | `…/gliclass/layers.py` |
| `[GLI-DATA]` | `…/gliclass/data_processing.py` |
| `[GLI-PIPE]` | `…/gliclass/pipeline.py` |
| `[HFCFG]` | `https://huggingface.co/heman10x/rlcd-modernbert-151m/raw/main/config.json` |
| `[TOKCFG]` | `https://huggingface.co/heman10x/rlcd-modernbert-151m/raw/main/tokenizer_config.json` |
| `[TOKJSON]` | `…/resolve/main/tokenizer.json` (3,583,596 bytes) |
| `[ST]` | `model.safetensors` header, read by HTTP range `bytes=0-16631` (8-byte LE u64 length = 16624, then JSON) |
| `[ONNX]` | `model.onnx` first 16 MiB, read by HTTP range; the whole graph node list lives in bytes `0 … 803 715`, weights start at `803 715` |
| `[JEZ]` | `https://raw.githubusercontent.com/Heman10x-NGU/openJev-verdict-2.0/main/…` — `core/formatting.py`, `core/engine_encoder.py`, `export/export_onnx.py`, `webgpu-demo/worker.js`, `artifacts/v2/*` |
| `[HF-API]` | `https://huggingface.co/api/models/heman10x/rlcd-modernbert-151m` and `/tree/main` |
| `[MBERT]` | `https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/modernbert/modeling_modernbert.py` |
| `[MBCFG]` | `…/models/modernbert/configuration_modernbert.py` |
| `[MASK]` | `…/src/transformers/masking_utils.py` |

---

## 0. TL;DR of the two facts most likely to be got wrong

1. **`model.logit_scale` is loaded but never used.** It appears in `model.safetensors` (`[ST]`) but is **absent from the exported ONNX graph** — the final graph node is the score `Einsum` and nothing else (`[ONNX]` bytes `803 621 … 803 692`). Reason: `[GLI-MODEL]:465-466` gates the scale on `if self.config.normalize_features:`, and `[HFCFG]` has `"normalize_features": false`. Corroborated by the shipped calibrator: fitted temperature is `1.4265` (`calibrator.json`), not `≈1/2.6592 = 0.376`.
2. **Layer 0 has no `attn_norm`.** `[MBERT]:309-312` sets `self.attn_norm = nn.Identity()` for `layer_idx == 0`, and indeed `model.encoder_model.layers.0.attn_norm.weight` is **absent** from `[ST]` (all other 21 layers have it). The ONNX graph confirms `layers.0/attn/Wqkv/MatMul` consumes `embeddings/norm/LayerNormalization_output_0` directly (`[ONNX]` byte `34 456`). A naive port will allocate 22 attn-norms and misalign every subsequent tensor.

Total parameters in `[ST]`: **151,378,177** = `143` tensors (the HF API reports `151378176`, off by one; the README and `train_manifest.json` both say `151,378,177`).

---

# A. INPUT CONSTRUCTION

## A.1 The canonical string

`[JEZ] core/formatting.py:43-49`

```python
def format_prompt(text, label_descriptions):
    label_prefix = "".join(f"{LABEL_MARKER}{label}" for label in label_descriptions)
    return f"{label_prefix}{SEP_MARKER}{text}"
```

with `LABEL_MARKER = "<<LABEL>>"`, `SEP_MARKER = "<<SEP>>"` (`core/formatting.py:20-23`).
So the string is literally `"<<LABEL>>" + desc1 + "<<LABEL>>" + desc2 + … + "<<SEP>>" + text` — **no separator between a description and the next `<<LABEL>>`**, no trailing `<<SEP>>`.

The `text` itself is templated (`core/formatting.py:40, 52-65`):

```
INPUT_TEMPLATE = "Question: {question}\n\nContext:\n{context}"
```

and if `question == ""` the text is just `context` (used for the `noul` query kind, `core/engine_encoder.py:120-124`).

This is byte-identical to the GLiClass library's own uni-encoder construction:

`[GLI-MODEL]`-adjacent `[GLI-PIPE]:561-586` (`UniEncoderZeroShotClassificationPipeline.prepare_input`):
```python
for label in labels: input_parts.append(f"{self.label_token}{label}")
input_parts.append(self.sep_token)
if prompt: input_parts.append(prompt)
return "".join(input_parts) + text + examples_str   # when config.prompt_first
```
and the training-side equivalent `[GLI-DATA]:279-290` (`prepare_prompt`) + `:314-329` (`…for_uniencoder`):
```python
if self.prompt_first:
    input_text = "".join(input_text) + str(example["text"]) + examples_text
```
`[HFCFG]` has `"prompt_first": true`, so labels precede the text. `item.get("prompt","")` is the empty string in this checkpoint's usage (`prompt` is never passed by `DecisionEngine`).

## A.2 Special token ids (verified in the shipped tokenizer)

From `[TOKJSON] added_tokens` and `[TOKCFG]`:

| id | token | notes |
|---:|---|---|
| 50280 | `[UNK]` | |
| 50281 | `[CLS]` | `bos_token_id`/`cls_token_id` in `[HFCFG] encoder_config` |
| 50282 | `[SEP]` | `eos_token_id`/`sep_token_id` |
| 50283 | `[PAD]` | `pad_token_id` (both GLiClass config and encoder config) |
| 50284 | `[MASK]` | |
| **50368** | **`<<LABEL>>`** | `class_token_index`, `special: true`, `normalized: false` |
| **50369** | **`<<SEP>>`** | `text_token_index`, `special: true`, `normalized: false` |

`vocab_size = 50370` in both `[HFCFG]` and `[HFCFG].encoder_config`, and `model.encoder_model.embeddings.tok_embeddings.weight` is `[50370, 768]` (`[ST]`). There is **no `<<EXAMPLE>>` token** in this tokenizer; `example_token_index: 50372` (`[HFCFG]`) is above the vocabulary and is *dead* (see A.5).

## A.3 Exact sequence layout for one request

`tokenizer.json` contains a `TemplateProcessing` post-processor (`[TOKJSON] post_processor`) that inserts specials on `encode`:

```
single = [ {SpecialToken: [CLS], type_id 0}, {Sequence: A, type_id 0}, {SpecialToken: [SEP], type_id 0} ]
special_tokens: [CLS] -> ids [50281], [SEP] -> ids [50282], [PAD] -> [50283]
```

The base model's tokenizer (`knowledgator/gliclass-modern-base-v2.0/resolve/main/tokenizer.json`) has an **identical** `TemplateProcessing` with the same ids — so training-time, Python-inference-time and browser-time all add the same two specials.

Therefore the full id sequence is:

```
pos:  0        1      2 …             |   |                         n-1
id:  [CLS]  <<LABEL>> desc1_tokens … <<LABEL>> desc2_tokens … <<SEP>> text_tokens … [SEP]
      50281   50368                     50368                 50369                 50282
```

Key consequences:

* **`pooling_strategy: "first"` (→ `FirstTokenPooling1D`, `[GLI-POOL]:12-16`, `x[:, 0, :]`) selects position 0, which is `[CLS]`, not the first `<<LABEL>>`.** This is confirmed at the ONNX level: `[ONNX]` byte `799 469` is a `Gather` node (`axis = 1`) whose inputs are `…/final_norm/LayerNormalization_output_0` and a scalar-int64 constant `0` defined at `[ONNX]` byte `1 101` (`/model/model/encoder_model/Constant_3`, int64 `0`), output `#/model/model/pooler/Gather_output_0`. The initial `[CLS]` is what makes the pooled vector meaningful; if you build the sequence without `[CLS]` you get the first label's own `<<LABEL>>` vector instead and every logit changes.
* `class_token_index: 50368` = `<<LABEL>>`. The per-class feature for label *k* (0-indexed, left-to-right) is the hidden state **at the position of the k-th `<<LABEL>>` token**.
* `text_token_index: 50369` is numerically equal to `<<SEP>>` in this tokenizer, not to a distinct marker. It is only used when `extract_text_features == true` (`[GLI-MODEL]:202-220`); `[HFCFG]` has it `false`, so it is inert. (Keep the value if you port the flag, but do not assume a separate "text token" exists.)
* `example_token_index: 50372` is out of vocabulary and only used by `_create_segment_ids` (`[GLI-MODEL]:421-444`) which is reached only when `use_segment_embeddings == true` (`[HFCFG]`: `false`). Inert.

## A.4 May a label description span multiple tokens?

Yes. The tokenizer is BPE (`[TOKJSON] model.type == "BPE"`, base vocab 50 280) and only the **`<<LABEL>>` markers themselves** are added tokens. `<<LABEL>>desc` is therefore `[50368, t1, t2, …, tm]`. The *span* of label *k* is delimited implicitly by the marker positions: from the k-th `<<LABEL>>` token up to (but excluding) the (k+1)-th `<<LABEL>>` token.

Pooling of the span depends on `class_token_pooling` (`[GLI-MODEL]:164-199`):

* `[HFCFG]` and `[JEZ] artifacts/v2/config.json` both say **`"class_token_pooling": "first"`** → `_extract_class_features_first` (`[GLI-MODEL]:226-255`): use **only** the `<<LABEL>>` token's own hidden state. The description tokens contribute to the sequence (and thus to all hidden states via attention) but are **not** pooled. Because `embed_class_token: true` (`[HFCFG]`), the selected position is the marker itself; if it were `false` the code shifts the selection one position right (`[GLI-MODEL]:244-248`) to take the first description token.
* `"average"` would instead average all tokens with `cumsumn(class_token)==k+1`, before the text boundary and `attention_mask==1` (`[GLI-MODEL]:257-296`). **Not active here.**

## A.5 attention_mask, padding, max length, truncation

`[JEZ] core/engine_encoder.py:125-134`:
```python
tokenized_inputs = self.tokenizer(
    prompts, padding=True, truncation=True,
    max_length=self.max_length,          # default 1024  (engine_encoder.py:61)
    return_tensors="pt",
).to(self.device)
outputs = self.model(**tokenized_inputs)  # NOTE: max_num_classes NOT passed
```

* **No `token_type_ids`.** `[TOKCFG] model_input_names = ["input_ids", "attention_mask"]`; the ONNX graph inputs are exactly `input_ids` and `attention_mask` (`[ONNX]` bytes `797 392`, `5 313`), and `export_onnx.py:84` names them so.
* **Truncation is right-side** (HF default `truncation_side="right"`), so the tail of `text` is dropped. With a `TemplateProcessing` post-processor, HF/Rust `tokenizers` reserve room for the specials: the body is truncated to `max_length − 2` and `[CLS]`/`[SEP]` are still added, so total length `≤ 1024` and the leading `<<LABEL>>…<<SEP>>` prefix is **never** truncated (it is at the front). `[HFCFG].tokenizer_config.model_max_length` is 8192 but the library/engine defaults are 1024 (`[GLI-PIPE]:181,551,615,679,923` all default `max_length=1024`).
  * `[JEZ] webgpu-demo/worker.js:323, 146` uses `{truncation:true, max_length:1024}` — same budget.
  * `[GLI-DATA]:227` uses `max_length=512` for the collator default, but that is training-only.
* **Padding**: `padding=True`/"longest" pads on the right with id 50283 and `attention_mask = 0` for those slots. The `DataCollatorWithPadding` path (`[GLI-DATA]:462-510`) pads to the batch max too.
* **`max_num_classes` is not passed** on the checked-in inference path (`engine_encoder.py:134`), so `[GLI-MODEL]:171` falls back to `self.config.max_num_classes = 25` (`[HFCFG]`). This is why the ONNX class axis is statically 25 (see B.6). `[HFCFG] "max_labels_alloc": "dynamic"` only affects `DataCollatorWithPadding._resolve_max_num_classes` / `pipeline._resolve_max_num_classes`, not this call site.
* Attention mask semantics inside the encoder: pad **keys** are masked out; pad **queries** are not specially handled (they attend to the visible keys and produce garbage which is never read). For a `batch=1`, unpadded, "longest" sequence the mask is all-ones and only the sliding-window constraint applies.

---

# B. FORWARD PASS

## B.1 Encoder = plain `ModernBertModel` (no GLiClass deviation)

`[GLI-MODEL]:387-398`: `config_name = "ModernBertConfig"` is not in `DECODER_MODEL_MAPPING` nor in `{"T5Config","MT5Config","UMT5Config","DebertaV2Config"}`, so `ModelClass = AutoModel`; `self.encoder_model = AutoModel.from_config(config.encoder_config)`. `[HFCFG].encoder_config.architectures` says `["ModernBertForMaskedLM"]` but `AutoModel` with a `modernbert` config instantiates **`ModernBertModel`** (no MLM head). The safetensors keys confirm `ModernBertModel` naming (`[ST]`), and there is no `decoder.*` tensor.

`AutoModel` is used with `trust_remote_code` **not** required — `model_type: "modernbert"` is a first-class transformers architecture. There is **no GLiClass-specific modification to the encoder**; the head is applied to the *final* hidden state.

Confirmed hyper-parameters (`[HFCFG].encoder_config`):

| field | value |
|---|---|
| `num_hidden_layers` | 22 |
| `hidden_size` | 768 |
| `num_attention_heads` | 12 (⇒ `head_dim = 64`) |
| `intermediate_size` | 1152 |
| `mlp_bias`, `attention_bias`, `norm_bias` | all `false` |
| `hidden_activation` | `"gelu"` (⇒ `ACT2FN["gelu"]` = **exact erf GELU**, not `gelu_new`/tanh) |
| `norm_eps` / `layer_norm_eps` | `1e-05` |
| `attention_dropout`, `mlp_dropout`, `embedding_dropout`, `classifier_dropout` | `0.0` |
| `local_attention` | 128 |
| `global_attn_every_n_layers` | 3 |
| `layer_types` | explicit list, index 0,3,6,9,12,15,18,21 = `"full_attention"`, all others `"sliding_attention"` |
| `rope_parameters` | `full_attention: {rope_theta: 160000.0, rope_type: "default"}`; `sliding_attention: {rope_theta: 10000.0, rope_type: "default"}` |
| `max_position_embeddings` | 8192 |
| `pad_token_id` | 50283 |
| `tie_word_embeddings` | true (irrelevant for `ModernBertModel`) |
| `position_embedding_type` | `"absolute"` (a vestigial DeBERTa/BERT key; ModernBERT uses RoPE — do not implement learned positions) |

The base model's config (`knowledgator/gliclass-modern-base-v2.0/config.json`) instead carries `global_rope_theta: 160000.0` / `local_rope_theta: 10000.0` (older transformers naming, `transformers_version 4.49.0`). The fine-tune config uses the transformers-5.x `rope_parameters` form (`transformers_version "5.17.0"`). **Both encode the same two thetas.**

### B.1.1 Encoder step by step (shapes for batch=1, sequence length `S`)

Notation: `E ∈ ℝ^{S×768}`, `LN_w` = `LayerNorm(768, eps=1e-5, elementwise_affine=True, bias=False)`.

1. **Embeddings** —— `[MBERT]:52-71`
   `H0 = LN_emb( tok_embeddings(input_ids) )`, `LN_emb.weight = model.encoder_model.embeddings.norm.weight [768]`, no bias, no dropout at eval.
   `tok_embeddings.weight [50370, 768]` (has `padding_idx=50283`, irrelevant for lookup at inference).
2. **Position ids** —— `[MBERT]:448-449`: `position_ids = arange(S)` (absolute, *including* pad slots).
3. **RoPE tables** —— `[MBERT]:94-163` + `[MBCFG]:160-167`. Two independent rope modules exist in the traced graph: `rotary_emb` (full) and `rotary_emb_1` (sliding) (`[ONNX]` byte `22802` ff., `29167` ff.).
   For a layer type with base `θ` and `head_dim = 64`:
   ```
   inv_freq[i] = 1 / θ ** (2i / 64)          i = 0 … 31        # 32 values, float32
   freqs[p, i] = p * inv_freq[i]             p = 0 … S-1
   emb         = concat(freqs, freqs)        # last dim 64   (cat, NOT interleaved)
   cos         = cos(emb) * 1.0
   sin         = sin(emb) * 1.0              # attention_scaling == 1.0
   ```
   `θ = 160000.0` for `full_attention` layers, `θ = 10000.0` for `sliding_attention` layers.
   Computed in **fp32** then cast to the activation dtype (`[MBERT]:157-163`).
4. **Mask construction** —— `[MBERT]:453-462`, `[MASK]:998-1101` (`create_bidirectional_mask`), `[MASK]:1239-1336` (`create_bidirectional_sliding_window_mask`), `[MASK]:142-159`.
   * `full_attention`: key `k` is visible to query `q` iff `attention_mask[k] == 1`. (No causal mask; `is_causal = False`, `[MBERT]:257`.)
   * `sliding_attention`: additionally `abs(q − k) <= sliding_window`, where `sliding_window = config.sliding_window = local_attention // 2 = 64` (`[MBCFG]:160-162`, `[MASK]:150` — "inclusive"). **Window = 129 keys per query (64 left + self + 64 right).** Verified in the exported graph as `Constant_33 = int64 64` feeding `Sub → Abs → LessOrEqual` (`[ONNX]` bytes `18069…18488`, value bytes at `18383`).
     *Caveat*: `ModernBertAttention.__init__` stores `self.sliding_window = config.sliding_window + 1 = 65` (`[MBERT]:250-255`) but that +1 is only consumed by the **flash-attention** path (inclusive boundaries); with the eager/SDPA mask path the effective window is 64. Use 64.
   * `[HFCFG].encoder_config._attn_implementation_autoset = false` and no `_attn_implementation` is stored, so transformers picks its default backend (`sdpa` when available). Eager/SDPA/flash all consume the *same* 4D mask, so the numerics of the mask are backend-independent. (One backend-dependent detail: `eager_attention_forward` casts to fp32 for the softmax, `[MBERT]:180`. An fp32 Rust port matches that exactly.)
5. **Per layer `l = 0 … 21`** —— `[MBERT]:304-333`
   ```
   x  = H_l
   y  = x if l == 0 else LN_attn_l(x)        # LN_attn_0 IS nn.Identity
   qkv = y @ Wqkv_l^T                        # Wqkv [2304, 768]; qkv: (S, 2304)
   qkv = qkv.view(S, 3, 12, 64)              # row-major, so
                                             #   rows 0..767   of Wqkv -> Q
                                             #   rows 768..1535 -> K
                                             #   rows 1536..2303 -> V ; within each, head h = cols [64h, 64h+64)
   q, k, v = qkv[:,0], qkv[:,1], qkv[:,2]    # each (S, 12, 64) -> transpose to (12, S, 64)
   q = q*cos + rotate_half(q)*sin            # rotate_half(z) = cat(-z[32:], z[:32])   [MBERT]:188-192
   k = k*cos + rotate_half(k)*sin            # (done in fp32 then cast back, [MBERT]:214-219)
   A = (q @ k^T) * (1/sqrt(64))              # (12, S, S)
   A += mask_bias                            # -inf (or a large negative) where masked
   A = softmax(A, axis=-1, dtype=fp32)       # then cast back
   O = A @ v                                 # (12, S, 64) -> transpose -> (S, 12, 64) -> reshape (S, 768)
   O = O @ Wo_l^T                            # Wo [768, 768]
   h = x + O
   g = LN_mlp_l(h)                           # LN_mlp_l.weight [768], always present
   u = g @ Wi_l^T                            # Wi [2304, 768]
   a, b = u[..., :1152], u[..., 1152:]       # chunk(2, dim=-1): FIRST half = activated branch,
                                             #                  SECOND half = gate
   h = h + ( gelu_erf(a) * b ) @ Wo_mlp_l^T  # Wo_mlp [768, 1152]
   H_{l+1} = h
   ```
   `gelu_erf(z) = 0.5 * z * (1 + erf(z / sqrt(2)))` — this is exactly what `ACT2FN["gelu"]` is, and exactly the node chain you can read in the export: `Div(→/√2) → Erf → Add(+1) → Mul(·0.5) → Mul(·u[:1152])` (`[ONNX]` bytes `62698 … 64244`, per layer).
   The GeGLU operand split order is confirmed by the export: `mlp/Slice_output_0` (first half) goes through the erf chain, `mlp/Slice_1_output_0` (second half) feeds `mlp/Mul_2` directly (`[ONNX]` bytes `61707, 62429, 64244`).
   All dropouts are `0.0`/eval-identity: `attention_dropout = 0.0`, `mlp_dropout = 0.0`, `embedding_dropout = 0.0` (`[HFCFG].encoder_config`). `out_drop` is `nn.Identity()` because `attention_dropout == 0` (`[MBERT]:260`).
6. **Final norm** —— `[MBERT]:476`
   `E = LN_final(H_22)`, `LN_final.weight = model.encoder_model.final_norm.weight [768]`, no bias, eps 1e-5.
   `GLiClassModel` uses this as `encoder_layer` because `encoder_layer_id == -1` (`[GLI-MODEL]:533-539`, `[HFCFG]`). **Not** `last_hidden_state` before the final norm, and not a hidden-state stack.

## B.2 Head, step by step (exact order)

`[GLI-MODEL]:446-469` (`GLiClassBaseModel.process_encoder_output`) is the whole head. Called from `[GLI-MODEL]:540-542`.

Let `E = final_norm(last_hidden_state)` (`S × 768`), `N = 25` (`config.max_num_classes`), `K = number of label descriptions actually passed`.

**(1) Per-class feature extraction — `[GLI-MODEL]:226-255`**

```
m_s        = (input_ids[s] == 50368)                       # (S,) bool
cum_s      = cumsum(m_s)                                   # (S,) int64
select[k,s]= m_s AND (cum_s - 1 == k)                      # (N=25, S) bool     k = 0 … 24
sel        = select.to(float32)                            # (25, S)
classes_embedding = einsum("ks,sd->kd", sel, E)            # (25, 768)
classes_embedding_mask = (arange(25) < sum(m_s)).to(dtype) # (25,) — NOT used for the forward output
```
Equivalent to: `classes_embedding[k] = E[pos of (k+1)-th <<LABEL>> token]`, and `0` (a zero vector) for every `k ≥ K`.
`embed_class_token == true` (`[HFCFG]`) means **no** right-shift by one.
The ONNX graph is this literally: `Equal → Cast → CumSum → Unsqueeze → Sub(-1) → Equal → And → Cast → Einsum("bks,bsd->bkd")` (`[ONNX]` bytes `797392 … 799316`; the folded `arange(25).view(1,25,1)` int64 constant is at byte `798553`, dims `[1,25,1]`).

**(2) Text feature — `[GLI-MODEL]:201-224` then `:453`**

`extract_text_features == false` (`[HFCFG]`) ⇒
```
text_tokens_embeddings = E            # the whole sequence, (S, 768)
text_tokens_mask       = attention_mask
pooled_output          = FirstTokenPooling1D(text_tokens_embeddings) = E[0]   # (768,)
```
`E[0]` is the `[CLS]` position (see A.3). ONNX: `Gather(axis=1, indices=0)` (`[ONNX]` byte `799469`).

**(3) `text_projector` — `[GLI-MODEL]:454`, `[GLI-LAYERS]:49-63`**

```
t = W_t2 · gelu_erf( W_t1 · E[0] + b_t1 ) + b_t2          # (768,)
```
* `model.text_projector.linear_1.weight [768, 768]`, `.bias [768]`, `linear_2.weight [768,768]`, `.bias [768]`
* `projector_hidden_act = "gelu"` (`[HFCFG]`) ⇒ exact erf GELU
* **biases are present** (`bias=True` hard-coded in `FeaturesProjector`, `[GLI-LAYERS]:53,56`) even though the encoder itself is bias-free
* `nn.Dropout(config.dropout)` with `dropout = 0.1` sits between act and `linear_2` (`[GLI-LAYERS]:55,61`) — eval ⇒ identity
* Line `455`: `pooled_output = self.dropout(pooled_output)` — a second dropout after `linear_2`, also identity at eval.

ONNX confirms it is a `Gemm(transB)` + erf-GELU + `Gemm(transB)` (`[ONNX]` bytes `799557 … 801361`).

**(4) `classes_projector` — `[GLI-MODEL]:459`, same `FeaturesProjector` class**

```
c_k = W_c2 · gelu_erf( W_c1 · classes_embedding[k] + b_c1 ) + b_c2      # (25, 768)
```
ONNX uses `MatMul`+`Add` (rank-3 input) instead of `Gemm` (`[ONNX]` bytes `801407 … 803461`) — mathematically the same.

> **Important:** because both projections have biases, the padded slots are **not** zero after projection. For `k ≥ K`, `classes_embedding[k] = 0` but `c_k = W_c2 · gelu_erf(b_c1) + b_c2`, a *single constant vector shared by all padded slots*. So padded logits are generally non-zero (and equal to each other). They must be discarded by slicing, not by a zero test.

**(5) Normalisation — `[GLI-MODEL]:456-461`**

`normalize_features == false` (`[HFCFG]`) ⇒ **both L2 normalisations are skipped** (`self.epsilon = 1e-8` is unused).

**(6) Scorer — `[GLI-MODEL]:463`**

`scorer_type: "simple"` ⇒ `SCORER2OBJECT["simple"] = ScorerDot` (`[GLI-SCORE]:42-50, 298-305`):

```
logits[k] = dot( t , c_k )                       # einsum("BD,BCD->BC")
```
`ScorerDot.__init__(self, *args, **kwargs)` is empty — **no parameters, no LayerNorm, no attention, no MLP, no bias.** The config keys `scorer_num_heads: 16`, `scorer_mlp_hidden_size: 1024`, `scorer_attn_dropout: 0.1`, `scorer_encoder_num_layers: 2` are accepted and **silently ignored** for `"simple"` (`[GLI-MODEL]:136-141` passes them as kwargs). This is why the safetensors file has zero `model.scorer.*` tensors.
`text_mask` is passed in but `ScorerDot.forward(**kwargs)` ignores it.

**(7) Logit scale — `[GLI-MODEL]:465-466`**

```python
if self.config.normalize_features:
    logits = logits * self.logit_scale.to(...)
```
`normalize_features == false` ⇒ **the scale is not applied.** `model.logit_scale` is a 0-dim `F32` parameter in `[ST]` but contributes nothing to inference. Independently verified three ways:
* it does not appear as a string anywhere in the ONNX graph region. `[ONNX]`: the node list is bytes `0 … 803 692` (immediately followed by `GraphProto.name = "main_graph"` at `803 692` and the first initializer's raw data at `803 715`), so a 16 MiB range read covers the entire node list plus the first initializers. `grep -c logit_scale` over those 16 MiB returns **0**, whereas the names it *does* use are all there (`text_projector` ×39, `classes_projector` ×41, plus `input_ids`, `attention_mask`, `logits`). Note that 2-D `nn.Linear` weights are stored under synthesised names (`onnx::MatMul_4434`, …) rather than their PyTorch names, but the node *output* names retain the module path (`…/text_projector/linear_1/Gemm_output_0`), and the bias/LayerNorm names are verbatim. A scalar used by a `Mul` would necessarily have surfaced as a node input name directly before that `Mul`;
* the last node in the graph is the scorer `Einsum` with equation `BD,BCD->BC`, output name `logits`, and no following `Mul`/`Div` (`[ONNX]` bytes `803621 … 803692`);
* `calibrator.json` fits `temperature = 1.4265` — if a 2.6592 logit scale were baked in, the fitted temperature would be ≈ `0.376`.

**(8) Output — `[GLI-MODEL]:548-555`**

`GLiClassOutput.logits` has shape `(batch, 25)`. Consumers slice `[..., :K]`:
* `[JEZ] core/engine_encoder.py:141-144` — `raw_logits[i, :num_classes]`
* `[JEZ] webgpu-demo/worker.js:350-352` — `rawOutput.slice(0, numCandidates)`
* `[GLI-PIPE]:270` — `logits[: len(labels)]`

**Post-processing (outside the model):** `softmax(logits[:K] / T)` (`[JEZ] worker.js:355-362`, `[JEZ] core/engine_encoder.py:169`). The ONNX graph itself ends at the raw logits. `T` is calibration data rather than part of the model, and the two calibrated artifacts do not agree: the author's `artifacts/v2/calibrator.json` uses `T = 1.0`, while the shipped `models/verdict-151m/calibrator.json` fits `T = 1.4265148639678955` for its 5-candidate scope -- which is the value `crates/verdict-candle/src/bin/verdict_infer.rs:145` and `docs/VERDICT_ENGINE.md` use. Earlier revisions of this section said "`T = 1.0` as shipped", conflating the author's artifact with the shipped one.

## B.3 Scratch summary of the head (single-request, no batch)

```rust
let e  = final_norm_last_hidden_state;               // [S, 768]
let cls = e.row(0);                                  // [768]      <-- [CLS]
let t  = projector(&text_projector, cls);            // [768]
let mut slot = [[0f32; 768]; 25];
let mut k = 0usize;
for s in 0..S {
    if input_ids[s] == 50368 { if k < 25 { slot[k] = e.row(s).to_vec(); } k += 1; }
}
let cls_emb = projector(&classes_projector, slot);   // [25, 768]
let logits: [f32; 25] = dot(t, cls_emb[j]);          // NO logit scale, NO L2 norm
// use logits[..K]
```

## B.4 Is the class count dynamic? Is it padded to 25?

* The **semantic** class count `K` is dynamic (one per request: 2–25 labels).
* The **tensor** width is **statically 25** on the checked-in inference path, because `max_num_classes` is not passed (`[JEZ] core/engine_encoder.py:134`) and `[GLI-MODEL]:171` falls back to `config.max_num_classes = 25`.
  The ONNX export hard-codes it: `dynamic_axes` marks only `logits` axis 0 (`export_onnx.py:71-75`), and the folded `arange(25)` constant `[1,25,1]` is baked in (`[ONNX]` byte `798553`).
* The library *can* make it dynamic: `pipeline._resolve_max_num_classes` with `max_labels_alloc == "dynamic"` returns `max(len(labels))` and passes it as `max_num_classes` (`[GLI-PIPE]:351-356, 404-407`). **The shipped ONNX does not do this.**
* **Masking of padded slots:** `classes_embedding_mask` (`[GLI-MODEL]:252-253`) is computed and passed to `get_loss` (multi-label focal loss only, `[GLI-MODEL]:339-341`), and is **never applied to `logits`**. Padded slots get the constant projected-bias vector described in B.2(4) and produce a constant logit. Correct handling is to ignore `logits[K..25]` entirely. Do **not** sqrt/softmax over 25.
* Capacity contract enforced upstream: `MAX_SUPPORTED_CANDIDATES = 25`, `MAX_SUBSTANTIVE_CANDIDATES = 24` (24 options + 1 `__insufficient_evidence__`) — `[JEZ] core/formatting.py:25-37`, mirrored in `webgpu-demo/prompt_contract.json`.

---

# C. TENSOR INVENTORY (`model.safetensors`)

Read directly from the safetensors header: `GET https://huggingface.co/heman10x/rlcd-modernbert-151m/resolve/main/model.safetensors` with `Range: bytes=0-16631`. The first 8 bytes are the little-endian `u64` `0x40F0` = **16624**, i.e. the header JSON is bytes `[8, 16632)`; the data section begins at `16632` and the first tensor's `data_offsets[0]` is `0` relative to it. `__metadata__` is **absent**. LFS sha256 of this file = `d252823994d47a7933217fc86449493299643af6a0c0d83d6bd5a7666d3253ef` (`[HF-API] /tree/main`), which **matches `artifacts/v2/bundle_manifest.json:safetensors_sha256`** — so the shipped `model.onnx` (sha256 `4ae01f82…`) was exported from exactly these weights.

* **Total tensor count: 143.** All `F32` (little-endian, C-contiguous row-major).
* **Total parameters: 151,378,177.**
* Ordering below is **numeric-natural** (layer `0,1,2,…,21`), which is the order the GLiClass/`ModernBertModel` modules expect. Note that the file's on-disk key order is *lexicographic* (`layers.0, layers.1, layers.10, layers.11, … layers.2, layers.20, …`); a Rust loader must key by name, not by position.

### Head (9 tensors, 2,362,369 params)

```
model.classes_projector.linear_1.bias                  F32  [768]                768
model.classes_projector.linear_1.weight                F32  [768, 768]        589824
model.classes_projector.linear_2.bias                  F32  [768]                768
model.classes_projector.linear_2.weight                F32  [768, 768]        589824
model.logit_scale                                      F32  []                     1   <- loaded, NEVER USED
model.text_projector.linear_1.bias                     F32  [768]                768
model.text_projector.linear_1.weight                   F32  [768, 768]        589824
model.text_projector.linear_2.bias                     F32  [768]                768
model.text_projector.linear_2.weight                   F32  [768, 768]        589824
```
(All four `F32 [768, 768]` are PyTorch `nn.Linear` weights, i.e. **`y = x @ Wᵀ + b`**.)

### Embeddings + final norm (3 tensors, 38,685,696 params)

```
model.encoder_model.embeddings.norm.weight             F32  [768]                   768
model.encoder_model.embeddings.tok_embeddings.weight   F32  [50370, 768]       38684160
model.encoder_model.final_norm.weight                  F32  [768]                   768
```

### Encoder layers — 131 tensors, 110,330,112 params

**Layer 0 is special: no `attn_norm`.** All layers have exactly 6 tensors except layer 0, which has 5.

```
model.encoder_model.layers.0.attn.Wo.weight            F32  [768, 768]         589824
model.encoder_model.layers.0.attn.Wqkv.weight          F32  [2304, 768]       1769472
model.encoder_model.layers.0.mlp.Wi.weight             F32  [2304, 768]       1769472
model.encoder_model.layers.0.mlp.Wo.weight             F32  [768, 1152]        884736
model.encoder_model.layers.0.mlp_norm.weight           F32  [768]                 768
                                                       (NO layers.0.attn_norm.weight)
```
For `l = 1 … 21` (21 layers × 6 tensors):
```
model.encoder_model.layers.{l}.attn.Wo.weight          F32  [768, 768]         589824
model.encoder_model.layers.{l}.attn.Wqkv.weight        F32  [2304, 768]       1769472
model.encoder_model.layers.{l}.attn_norm.weight        F32  [768]                 768
model.encoder_model.layers.{l}.mlp.Wi.weight           F32  [2304, 768]       1769472
model.encoder_model.layers.{l}.mlp.Wo.weight           F32  [768, 1152]        884736
model.encoder_model.layers.{l}.mlp_norm.weight         F32  [768]                 768
```
Per non-zero layer: `589824 + 1769472 + 768 + 1769472 + 884736 + 768 = 5,015,040`.
Per layer-0: `589824 + 1769472 + 1769472 + 884736 + 768 = 5,014,272`.
Total: `5,014,272 + 21 × 5,015,040 = 5,014,272 + 105,315,840 = 110,330,112`.
Cross-check: `38,685,696` (embeddings block) `+ 110,330,112 + 2,362,369` (head block) `= 151,378,177`. ✔

Expanded, layer by layer (all `F32`):
```
l=1    attn.Wo [768,768]  attn.Wqkv [2304,768]  attn_norm [768]  mlp.Wi [2304,768]  mlp.Wo [768,1152]  mlp_norm [768]
l=2    attn.Wo [768,768]  attn.Wqkv [2304,768]  attn_norm [768]  mlp.Wi [2304,768]  mlp.Wo [768,1152]  mlp_norm [768]
l=3    attn.Wo [768,768]  attn.Wqkv [2304,768]  attn_norm [768]  mlp.Wi [2304,768]  mlp.Wo [768,1152]  mlp_norm [768]
l=4    attn.Wo [768,768]  attn.Wqkv [2304,768]  attn_norm [768]  mlp.Wi [2304,768]  mlp.Wo [768,1152]  mlp_norm [768]
l=5    attn.Wo [768,768]  attn.Wqkv [2304,768]  attn_norm [768]  mlp.Wi [2304,768]  mlp.Wo [768,1152]  mlp_norm [768]
l=6    attn.Wo [768,768]  attn.Wqkv [2304,768]  attn_norm [768]  mlp.Wi [2304,768]  mlp.Wo [768,1152]  mlp_norm [768]
l=7    attn.Wo [768,768]  attn.Wqkv [2304,768]  attn_norm [768]  mlp.Wi [2304,768]  mlp.Wo [768,1152]  mlp_norm [768]
l=8    attn.Wo [768,768]  attn.Wqkv [2304,768]  attn_norm [768]  mlp.Wi [2304,768]  mlp.Wo [768,1152]  mlp_norm [768]
l=9    attn.Wo [768,768]  attn.Wqkv [2304,768]  attn_norm [768]  mlp.Wi [2304,768]  mlp.Wo [768,1152]  mlp_norm [768]
l=10   attn.Wo [768,768]  attn.Wqkv [2304,768]  attn_norm [768]  mlp.Wi [2304,768]  mlp.Wo [768,1152]  mlp_norm [768]
l=11   attn.Wo [768,768]  attn.Wqkv [2304,768]  attn_norm [768]  mlp.Wi [2304,768]  mlp.Wo [768,1152]  mlp_norm [768]
l=12   attn.Wo [768,768]  attn.Wqkv [2304,768]  attn_norm [768]  mlp.Wi [2304,768]  mlp.Wo [768,1152]  mlp_norm [768]
l=13   attn.Wo [768,768]  attn.Wqkv [2304,768]  attn_norm [768]  mlp.Wi [2304,768]  mlp.Wo [768,1152]  mlp_norm [768]
l=14   attn.Wo [768,768]  attn.Wqkv [2304,768]  attn_norm [768]  mlp.Wi [2304,768]  mlp.Wo [768,1152]  mlp_norm [768]
l=15   attn.Wo [768,768]  attn.Wqkv [2304,768]  attn_norm [768]  mlp.Wi [2304,768]  mlp.Wo [768,1152]  mlp_norm [768]
l=16   attn.Wo [768,768]  attn.Wqkv [2304,768]  attn_norm [768]  mlp.Wi [2304,768]  mlp.Wo [768,1152]  mlp_norm [768]
l=17   attn.Wo [768,768]  attn.Wqkv [2304,768]  attn_norm [768]  mlp.Wi [2304,768]  mlp.Wo [768,1152]  mlp_norm [768]
l=18   attn.Wo [768,768]  attn.Wqkv [2304,768]  attn_norm [768]  mlp.Wi [2304,768]  mlp.Wo [768,1152]  mlp_norm [768]
l=19   attn.Wo [768,768]  attn.Wqkv [2304,768]  attn_norm [768]  mlp.Wi [2304,768]  mlp.Wo [768,1152]  mlp_norm [768]
l=20   attn.Wo [768,768]  attn.Wqkv [2304,768]  attn_norm [768]  mlp.Wi [2304,768]  mlp.Wo [768,1152]  mlp_norm [768]
l=21   attn.Wo [768,768]  attn.Wqkv [2304,768]  attn_norm [768]  mlp.Wi [2304,768]  mlp.Wo [768,1152]  mlp_norm [768]
```

### Tensors that are NOT in the file (and must not be expected)

* `model.scorer.*` — `ScorerDot` has no parameters.
* `model.pooler.*` — `FirstTokenPooling1D` has none.
* `model.encoder_model.rotary_emb.*_inv_freq` — non-persistent buffers in `[MBERT]:113-114`; recompute them.
* `model.encoder_model.embeddings.drop`, `*.attn.out_drop`, `*.mlp.drop` — dropout has no parameters.
* `model.encoder_model.layers.0.attn_norm.weight` — see above.
* No `decoder.*` / LM head (`tie_word_embeddings` is moot here), no `class_embedding`, no `segment_embeddings` (`use_segment_embeddings: false`), no `lstm.*` (`use_lstm: false`), no `layer_wise_attention.*` (`squeeze_layers: false`).

### Byte-offset map (data section, relative to byte 16632 of the file)

```
model.encoder_model.embeddings.tok_embeddings.weight   [4,727,808 .. 159,464,448)   <- 151.6 MiB, dominates the file
model.encoder_model.layers.0.attn.Wo.weight            [159,467,520 .. 161,826,816)
… (layers in file order) …
model.encoder_model.layers.9.mlp_norm.weight           [600,784,896 .. 600,787,968)
model.logit_scale                                      [600,787,968 .. 600,787,972)
model.text_projector.*                                 [600,787,972 .. 605,512,708)
```
Full file size 605,529,340 bytes = 16,632 + 605,512,708. ✔

---

# D. OPEN QUESTIONS / RISKS

**D.1 — `logit_scale` (highest-impact).** Settled to high confidence by three independent lines of evidence (byte-level absence from the ONNX graph, the graph terminating at the scorer `Einsum`, and a fitted calibration temperature of 1.4265 rather than ≈0.376). *Residual risk:* the ONNX graph was produced by `torch.onnx.export(..., dynamo=False)` on the TorchScript path which constant-folds; a folded scalar could in principle be inlined. It is not, because there is no arithmetic node after the `Einsum`. **What would settle it beyond doubt:** run `onnxruntime` (or a Rust `onnx` reader) on `model.onnx` with a fixed 4-label prompt and compare to a Rust reference whose logits are a raw dot product. A one-off `pip install onnxruntime` in a throwaway venv is enough.

**D.2 — Version skew of the GLiClass library.** `[HFCFG].transformers_version` is `"5.17.0"` and the base model's is `"4.49.0"`. `gliclass` `config.py` on `main` (tag `v0.1.20`, tree `40baa67…`) contains **every** key present in `[HFCFG]` (`class_token_pooling`, `max_labels_alloc`, `example_token_index`, `encoder_layer_id`, `layer_wise`, `use_segment_embeddings`, `scorer_encoder_num_layers`, `dropout`), and `[HFCFG]` contains no key that `main`'s `config.py` lacks. So the fine-tune is consistent with `v0.1.20`. **Risk:** if a *newer* GLiClass changed the `normalize_features` gate on `logit_scale`, the answer flips. **What would settle it:** `pip download gliclass==<ver> -d /tmp && unzip -p ... modeling_gliclass.py | grep -n logit_scale` for the version in `scripts/train.py`/`pyproject.toml` of the checkpoint author's repo (not pinned in `requirements-train.txt`, which only lists `transformers>=4.48`).

**D.3 — Sliding-window width 64 vs 65.** `[MBCFG]:160-162` defines `config.sliding_window = local_attention // 2 = 64`, and `[MASK]:150` makes it inclusive (`|q−k| ≤ 64`, 129 keys). `[MBERT]:253` then stores `+1` (`65`) for the flash path. The ONNX export is mask-based, and the string `Constant_33 = 64` is present in the mask region (`[ONNX]` byte `18383`). Implement **64**. *Verification:* compare a forward pass against `onnxruntime` on a sequence > 129 tokens — a 65-vs-64 error is invisible on short prompts.

**D.4 — Attention backend numerics.** `[MBERT]:180` casts attention logits to fp32 for the softmax; `apply_rotary_pos_emb` (`[MBERT]:214-219`) also computes in fp32 and casts back. Softmax reduction order differs across eager/SDPA/flash, so bit-exact agreement with a specific backend is not guaranteed for long sequences. For a canister port, fp32 throughout with a straightforward softmax is the closest match to the `torch.onnx.export`/ORT CPU path that the shipped artifacts were validated against (`reports/v2/exp_e8_fp16_parity.json` is the author's own parity study).

**D.5 — `token_type_ids` absent.** `[TOKCFG] model_input_names = ["input_ids","attention_mask"]`. Do not synthesize type ids; ModernBERT does not use them.

**D.6 — `text_token_index == 50369 == <<SEP>>` is a latent collision.** Harmless here (`extract_text_features: false`) but if you ever enable `extract_text_features`, `_extract_class_features_averaged` (`[GLI-MODEL]:272-285`) will use the `<<SEP>>` marker as the "text boundary", and `_extract_class_features_first` is unaffected. Flag it in the port.

**D.7 — `example_token_index: 50372` exceeds `vocab_size`.** Inert (`use_segment_embeddings: false`), but a Rust port that validates config indices against `tok_embeddings` rows will reject this config. Clamp/ignore rather than error.

**D.8 — Which `max_length`?** The checked-in engine uses 1024, the library pipeline default is 1024, the training collator default was 512, and the tokenizer advertises `model_max_length = 8192` with `max_position_embeddings = 8192`. Any of these produces identical logits for prompts shorter than the budget. Pick the smallest your canister can afford — attention cost is `O(S²)` for the 8 full-attention layers. Never exceed 8192 (`max_position_embeddings`).

**D.9 — Tensor-order hazard.** The safetensors key order is lexicographic, not numeric. Loading by position will silently produce wrong weights. Load by exact name (the 143 names above).

**D.10 — `padding` in a canister.** With batch = 1 and no padding, no `attention_mask` handling is needed except that the sliding-window mask still applies. If you pad (e.g. to chunk the encoder), you must mask pad **keys** (set to −∞ before softmax); pad queries may be left unconstrained.

**D.11 — fp16 provenance.** `model_fp16.onnx` (303,785,047 bytes) exists in the repo and is what the browser prefers (`worker.js:210-214`). Its logits are *not* identical to fp32; the author measured the gap in `reports/v2/exp_e8_fp16_parity.json`. Do not calibrate a Rust port against fp16 reference values.

---

## Appendix — reproduction commands used

```bash
# GLiClass sources (tree sha 40baa67… == tag v0.1.20)
curl -s "https://api.github.com/repos/knowledgator/GLiClass/git/trees/main?recursive=1"
for f in model.py scorers.py poolings.py config.py layers.py data_processing.py \
         pipeline.py training.py utils.py; do
  curl -sL "https://raw.githubusercontent.com/knowledgator/GLiClass/main/gliclass/$f" -o "$f"
done

# safetensors header (8-byte LE u64 length, then JSON)
curl -sL -H "Range: bytes=0-7" \
  "https://huggingface.co/heman10x/rlcd-modernbert-151m/resolve/main/model.safetensors" | xxd
curl -sL -H "Range: bytes=8-16631" \
  "https://huggingface.co/heman10x/rlcd-modernbert-151m/resolve/main/model.safetensors" -o hdr.json

# ONNX graph (whole node list is in the first 803,715 bytes; weights start there)
curl -sL -H "Range: bytes=0-16777215" \
  "https://huggingface.co/heman10x/rlcd-modernbert-151m/resolve/main/model.onnx" -o onnx_head.bin
python3 -c "d=open('onnx_head.bin','rb').read(); print(d.count(b'logit_scale'))"   # -> 0

# LFS digests (for cross-checking against bundle_manifest.json)
curl -s "https://huggingface.co/api/models/heman10x/rlcd-modernbert-151m/tree/main"
```

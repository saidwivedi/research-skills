# Silent Bugs in Deep Learning

191 bugs that produce no errors but cause wrong results. Code runs, loss decreases,
model learns the wrong thing or underperforms. Organized by how often they bite.

---

## Tier 1: Almost Every PhD Hits These

### 1. Python list instead of `nn.ModuleList`

Storing submodules in a plain list. PyTorch never registers them — invisible to
`.parameters()`, `.to(device)`, `.state_dict()`, `.train()`/`.eval()`.

**Symptom:** Some layers never get updated. Model trains but converges worse.
Saving and loading silently drops those layers' weights.

**Detection:** Run `list(model.named_parameters())` — every learnable layer should appear.

```python
# BUG
self.layers = [nn.Linear(64, 64) for _ in range(5)]

# FIX
self.layers = nn.ModuleList([nn.Linear(64, 64) for _ in range(5)])
```

### 2. NumPy seed not propagated to DataLoader workers

With `num_workers > 0`, each worker inherits the same NumPy seed. All workers
produce identical "random" augmentations. PyTorch seeds its own RNG per-worker,
but `numpy.random` is not reseeded.

**Symptom:** Augmentation looks like it works but applies identical transforms
across workers. Lower accuracy than expected.

**Detection:** Log augmentation output from different workers — they'll be identical.

```python
# BUG
loader = DataLoader(dataset, num_workers=4, shuffle=True)

# FIX
def seed_worker(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

loader = DataLoader(dataset, num_workers=4, shuffle=True, worker_init_fn=seed_worker)
```

### 3. Adam vs AdamW (L2 penalty vs true weight decay)

`torch.optim.Adam` with `weight_decay` applies L2 regularization, not decoupled
weight decay. Interacts incorrectly with Adam's adaptive LR. Also applies weight
decay to BatchNorm gamma/beta, pulling them toward zero.

**Symptom:** Converges to subtly worse solution. No error.

```python
# BUG
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

# FIX: use AdamW, exclude BN/bias
no_decay = ['bias', 'bn', 'norm']
params = [
    {'params': [p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)], 'weight_decay': 1e-4},
    {'params': [p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
]
optimizer = torch.optim.AdamW(params, lr=1e-3)
```

### 4. Frozen params still in optimizer

Setting `requires_grad = False` after creating the optimizer. With Adam, momentum
buffers still apply updates based on stale state.

**Symptom:** "Frozen" layers change during training.

```python
# BUG: frozen params still in optimizer
optimizer = Adam(model.parameters(), lr=1e-3)
for p in model.encoder.parameters():
    p.requires_grad = False  # too late

# FIX: freeze first, then create optimizer
for p in model.encoder.parameters():
    p.requires_grad = False
optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
```

### 5. RGB/BGR channel order mismatch

OpenCV loads BGR. PIL loads RGB. Pretrained models expect RGB. If you load with
OpenCV and apply ImageNet normalization, red and blue channels are swapped.

**Symptom:** 1-5% accuracy drop. Images "look fine" to quick inspection.

**Detection:** Visualize a sample after all preprocessing (reverse normalization too).

```python
# BUG
img = cv2.imread('photo.jpg')  # BGR
img = transforms.Normalize(mean=[0.485, 0.456, 0.406], ...)(to_tensor(img))

# FIX
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert immediately after loading
```

### 6. Wrong normalization constants

Copy-pasting ImageNet `mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]`
for medical scans, satellite imagery, spectrograms, etc.

**Symptom:** Slower convergence or worse performance. No error.

**Fix:** Compute your dataset's actual per-channel mean and std. ImageNet stats
are only appropriate for natural images similar to ImageNet.

### 7. Resuming without optimizer/scheduler state

Loading only `model.state_dict()`. Adam's momentum resets to zero, LR schedule
restarts from the beginning.

**Symptom:** Loss spikes after resume, recovers slowly, final result worse than
continuous training.

```python
# BUG
torch.save(model.state_dict(), 'checkpoint.pt')

# FIX: save everything
torch.save({
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'scheduler': scheduler.state_dict(),
    'epoch': epoch,
    'scaler': scaler.state_dict(),  # if using AMP
}, 'checkpoint.pt')
```

### 8. `view()` vs `permute()` — scrambled data with correct shape

`.view()` reinterprets flat memory without moving data. Using it to swap dimension
order scrambles the content while producing the correct shape.

**Symptom:** Model trains on scrambled data. Loss still decreases (SGD is resilient)
but performance far worse than expected.

```python
x = torch.randn(2, 3, 4, 5)  # (batch, C, H, W)

# BUG: scrambles data
y = x.view(2, 4, 5, 3)

# FIX: correctly reorders
y = x.permute(0, 2, 3, 1)  # (batch, H, W, C)
```

### 9. Softmax on wrong dimension

`F.softmax(logits, dim=0)` normalizes across the batch, not classes. Each
sample's "probabilities" don't sum to 1.

**Symptom:** Near-random accuracy.

**Detection:** `assert torch.allclose(probs.sum(dim=-1), torch.ones(batch_size))`

```python
# BUG
probs = F.softmax(logits, dim=0)  # normalizes across batch

# FIX
probs = F.softmax(logits, dim=-1)  # normalizes across classes
```

### 10. Loss tensor kept in list instead of `loss.item()`

Appending the loss tensor (not the scalar) to a list keeps the entire computation
graph alive, preventing garbage collection.

**Symptom:** GPU memory grows linearly until OOM.

```python
# BUG
losses.append(loss)

# FIX
losses.append(loss.item())
```

---

## Tier 2: Common in Specific Workflows

### 11. Gradient accumulation with variable-length sequences

With `reduction='mean'`, each micro-batch's loss is averaged over its own token
count. Short sequences get disproportionate weight.

**When:** LLM training, any variable-length sequence task with gradient accumulation.

**Fix:** Use `reduction='sum'` and divide by total token count across the
accumulation window.

### 12. LR scheduler stepped at wrong time

(a) `scheduler.step()` before `optimizer.step()` skips the first LR value.
(b) In DDP, scheduler gets stepped once per GPU per iteration.
(c) After resume, LR resets if scheduler state not restored.

**Detection:** Log `optimizer.param_groups[0]['lr']` at every step.

### 13. BatchNorm running stats diverge from batch stats

Running mean/var maintained via EMA can lag behind if training traverses a
degenerate loss landscape. At eval time, stale running stats produce different
outputs than training-time batch stats.

**Symptom:** Large train-val gap that looks like overfitting but is BatchNorm
mismatch.

**Detection:** Compare running stats against actual batch stats after training.

**Fix:** Try LayerNorm or GroupNorm. Or freeze running stats after warmup.

### 14. BatchNorm with batch size 1-4

Per-batch mean/variance estimates are extremely noisy with tiny batches.
Normalization hurts rather than helps.

**When:** GPU-constrained training.

**Fix:** Use GroupNorm, InstanceNorm, or LayerNorm. Or `SyncBatchNorm` across GPUs.

### 15. Xavier init with ReLU activations

Xavier assumes gradient ~1 near zero (tanh, sigmoid). ReLU has gradient 0 for
half the input space — variance shrinks exponentially through layers.

**Symptom:** Deep networks (30+ layers) fail to converge or converge much slower.

```python
# BUG
nn.init.xavier_uniform_(m.weight)

# FIX
nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
```

### 16. Dying ReLU neurons

If a neuron receives large negative inputs, gradient is exactly 0 and it can
never recover. Effective capacity silently shrinks.

**Detection:** Monitor fraction of dead neurons: `(activations == 0).float().mean()`
If past ~50%, switch to LeakyReLU/ELU/GELU or reduce LR.

### 17. FP16 gradient underflow without GradScaler

Small gradients underflow to zero in FP16. Entire layers silently stop learning.

**When:** Mixed precision training.

**Fix:** Use `torch.cuda.amp.GradScaler`. Or use BF16 (larger dynamic range).

### 18. Missing `detach()` on RNN hidden states

Passing hidden state without `.detach()` causes the graph to grow unboundedly,
backpropagating through entire training history.

**Symptom:** Memory grows linearly until OOM. Or wrong gradients if it doesn't OOM.

```python
# FIX: truncate BPTT at each batch boundary
hidden = tuple(h.detach() for h in hidden)
```

### 19. Multi-task loss scale imbalance

Summing losses of different magnitudes. The larger loss dominates all gradients.

**Symptom:** One task trains well, others stagnate. Total loss looks fine.

**Detection:** Log per-task gradient norms. If they differ by 10x+, there's an imbalance.

**Fix:** Normalize by initial loss magnitudes, or use learned weights (uncertainty
weighting, GradNorm).

### 20. `model.zero_grad()` zeros ALL params in GAN training

With separate optimizers for G and D, `model.zero_grad()` zeros the other
optimizer's accumulated gradients.

**Fix:** Use `optimizer_G.zero_grad()` and `optimizer_D.zero_grad()` separately.

### 21. Causal mask off-by-one leaking one future token

`mask[i][j] = 1 if j <= i+1` leaks one future token. Model cheats during
training, fails at autoregressive inference.

**Symptom:** Suspiciously low perplexity during training, drops at inference.

**Detection:** Verify `mask.sum(dim=-1)` equals `[1, 2, 3, ..., seq_len]`.

---

## Tier 3: Niche but Devastating

### 22. `ignore_index` in CrossEntropyLoss doesn't fully ignore

Loss for padding positions is zeroed, but the softmax denominator still includes
the padding class logit. Padding logits affect gradients for all other classes.

**When:** NLP with padding tokens.

### 23. `non_blocking=True` GPU→CPU reads garbage

`tensor.to("cpu", non_blocking=True)` starts async transfer. Reading immediately
gets stale/uninitialized memory.

**Fix:** `torch.cuda.synchronize()` before reading, or don't use `non_blocking`
for GPU→CPU.

### 24. `cudnn.benchmark=True` with variable input sizes

Re-benchmarks on every new shape. Different runs select different algorithms.

**Symptom:** Slower than expected + non-deterministic results.

**Fix:** Only use when input sizes are constant.

### 25. DDP gradient sync skips unused params during accumulation

With `find_unused_parameters=True`, only params used in the last micro-batch are
synced. Different GPUs maintain different parameter versions.

**When:** Distributed + conditional architectures + gradient accumulation.

### 26. HuggingFace tokenizer silently omits BOS/EOS

Some tokenizers (notably GPT-2) don't add BOS/EOS even with `add_special_tokens=True`.

**Detection:** Decode tokenized output and check.

### 27. Gradient penalty without `create_graph=True`

Penalty value is correct but its gradient w.r.t. model params is zero. The
penalty has no effect on training.

**When:** WGAN-GP.

### 28. `IterableDataset` not shuffled between epochs

Unlike `Dataset + DataLoader(shuffle=True)`, iterable datasets are not shuffled
by the DataLoader. Model sees identical ordering every epoch.

**Detection:** Plot per-batch loss within epoch — same pattern each epoch = no shuffle.

### 29. Tokenizer/model vocab mismatch after adding special tokens

Adding tokens to tokenizer without `model.resize_token_embeddings(len(tokenizer))`.
New token IDs may silently map to wrong embeddings.

### 30. Contrastive learning dimensional collapse

Representations collapse to a low-dimensional subspace. Loss looks normal.

**Detection:** Monitor effective rank of the representation matrix.

**When:** Self-supervised contrastive learning.

### 31. `register_buffer` with `None` excluded from state_dict

Buffer initialized as `None` is dropped on save/load. Works during training,
breaks after checkpoint restore.

### 32. `deepcopy` doesn't copy `.grad` fields

Copying a model mid-training loses gradient history. EMA or checkpoint-based
workflows silently break.

### 33. Flash Attention not disabling dropout in eval

Some implementations don't set dropout to 0 during `model.eval()`.

**Detection:** Run eval twice on same data — different results = dropout still active.

### 34. `torch.compile` silently skips hooks

Forward hooks may stop working after `torch.compile`. Feature extraction breaks
with no error.

**Detection:** Test with and without `torch.compile`.

### 35. Empty batch with `reduction='mean'` returns NaN

If a batch has zero elements after filtering, `CrossEntropyLoss(reduction='mean')`
returns NaN (0/0). Corrupts all weights through the optimizer.

### 36. Hardware bit flips (silent data corruption)

GPU memory corruption without error signals. Documented by NVIDIA for large-scale
training.

**Detection:** Periodic weight checksums. Compare across replicas.

---

## Tier 4: Transformer-Specific Silent Bugs

Bugs specific to attention mechanisms, positional encoding, encoder-decoder architectures,
and LLM inference. All produce no errors but cause wrong results.

### 37. Reshape without permute corrupts multi-head attention data

After splitting or merging attention heads, `reshape()`/`view()` after `transpose()`
reinterprets memory layout without moving data, silently scrambling Q/K/V across heads.

**Symptom:** Model trains but converges poorly. Each attention head receives data
that is a mix of other heads' data. No error because shapes are correct.

**Detection:** After any `transpose()` or `permute()`, call `.contiguous()` before
`.view()`. Compare intermediate tensors against `nn.MultiheadAttention` output.

```python
# BUG: reshape after transpose without contiguous
x = x.transpose(1, 2)          # [B, heads, seq, d] -> [B, seq, heads, d]
x = x.reshape(B, seq, heads*d) # data layout still [B, heads, seq, d] in memory

# FIX
x = x.transpose(1, 2).contiguous().view(B, seq, heads * d)
```

**Sources:** [DeepSpeed PR #5664](https://github.com/deepspeedai/DeepSpeed/pull/5664),
[Diffusers Issue #10303](https://github.com/huggingface/diffusers/issues/10303)

### 38. Missing attention scaling factor (1/sqrt(d_k))

Forgetting to divide QK^T by `sqrt(d_k)` before softmax. Dot products grow with
dimension, pushing softmax into saturation where gradients vanish.

**Symptom:** Model trains extremely slowly or converges to a poor solution. Softmax
outputs are near-one-hot for random inputs at initialization. No error.

**Detection:** Check that softmax input magnitudes are O(1), not O(d_k). Verify
`/ math.sqrt(d_k)` is present.

```python
# BUG
attn = F.softmax(Q @ K.transpose(-2, -1), dim=-1)

# FIX
attn = F.softmax(Q @ K.transpose(-2, -1) / math.sqrt(d_k), dim=-1)
```

### 39. SDPA causal mask top-left vs bottom-right alignment

PyTorch's `F.scaled_dot_product_attention(is_causal=True)` aligns the causal mask
to the top-left corner. During KV-cache inference (`seqlen_q != seqlen_kv`), this
produces the wrong mask — tokens attend to future positions.

**Symptom:** Correct output during training (seqlen_q == seqlen_kv), wrong output
during generation with KV cache. No error.

**Detection:** Compare generation output with and without KV cache. Use
`torch.nn.attention.bias.causal_lower_right` (PyTorch >= 2.3) instead of `is_causal=True`.

**Source:** [PyTorch Issue #108108](https://github.com/pytorch/pytorch/issues/108108)

### 40. RoPE interleave vs half-split mismatch

Meta's LLaMA uses interleaved RoPE dimension pairs `(d0,d1), (d2,d3), ...` while
HuggingFace originally used half-split `[d0..d_{n/2-1}]` and `[d_{n/2}..d_{n-1}]`.
Using the wrong convention with pretrained weights silently corrupts positional information.

**Symptom:** Model generates plausible but subtly wrong text. Perplexity is higher
than expected for a pretrained model. No error.

**Detection:** Check whether `rotate_half` splits dimensions in half or interleaves
pairs. Match the convention used during pretraining. Compare first-token outputs
against the official implementation.

**Sources:** [HuggingFace Issue #25199](https://github.com/huggingface/transformers/issues/25199),
[HuggingFace Issue #31859](https://github.com/huggingface/transformers/issues/31859)

### 41. Padding tokens without attention mask

Passing padded `input_ids` without `attention_mask`. The model attends to padding
tokens, corrupting hidden states for real tokens.

**Symptom:** Batched inference gives different (worse) results than single-sample
inference. No error or warning in many library versions.

**Detection:** Compare single-sample vs batched output for identical input. Always
pass `attention_mask` alongside `input_ids`.

```python
# BUG
outputs = model(input_ids=padded_ids)

# FIX
outputs = model(input_ids=padded_ids, attention_mask=attention_mask)
```

**Source:** [HuggingFace Troubleshooting](https://huggingface.co/docs/transformers/troubleshooting)

### 42. SDPA dropout ignores model.eval()

PyTorch's `F.scaled_dot_product_attention` had `train=True` hardcoded for dropout,
ignoring `model.eval()`. Dropout remained active during inference.

**Symptom:** Non-deterministic inference. Eval metrics are noisy or slightly worse
than expected. No error.

**Detection:** Run eval twice on identical input — different logits = dropout active.
Explicitly pass `dropout_p=0.0` during inference.

**Source:** [PyTorch Issue #124464](https://github.com/pytorch/pytorch/issues/124464)

### 43. F.dropout vs nn.Dropout in custom attention

`F.dropout(x, p=0.1)` without the `training` flag defaults to `training=True` and
ignores `model.eval()`. Unlike `nn.Dropout`, functional dropout does not check
the module's training state.

**Symptom:** Same as #42 — non-deterministic inference.

```python
# BUG: always applies dropout, even in eval
attn_weights = F.dropout(attn_weights, p=0.1)

# FIX option 1
attn_weights = F.dropout(attn_weights, p=0.1, training=self.training)

# FIX option 2
self.attn_dropout = nn.Dropout(0.1)
attn_weights = self.attn_dropout(attn_weights)
```

**Source:** [PyTorch Issue #26338](https://github.com/pytorch/pytorch/issues/26338)

### 44. Boolean vs additive attention mask convention swapped

PyTorch's `MultiheadAttention` treats boolean `True` as "masked out" (not attended),
but float masks use `-inf` for masked positions. Swapping conventions silently inverts
the mask — the model attends to exactly the wrong positions.

**Symptom:** Performance degrades but model trains. Causal generation leaks future
tokens or blocks all tokens. No error.

**Detection:** Visualize attention weights for a known input. Verify: boolean
`True` = masked, float `-inf` = masked, float `0.0` = attended.

**Source:** [PyTorch Issue #92554](https://github.com/pytorch/pytorch/issues/92554)

### 45. KV cache position ID offset not updated

During autoregressive generation with KV cache, position IDs for new tokens must
be offset by the cached sequence length. If always starting from 0, each generated
token gets the wrong positional encoding.

**Symptom:** First generated token is correct, then output becomes increasingly
wrong or repetitive. No error.

**Detection:** Verify `position_ids = cache_length + current_step`. Compare
generation with and without KV cache — outputs should match.

**Source:** [HuggingFace Issue #29149](https://github.com/huggingface/transformers/issues/29149)

### 46. FP16 attention logit overflow

In FP16 (max ~65504), QK^T dot products can overflow for large `d_k` or large
activation magnitudes, producing `inf` that turns to NaN after softmax.

**Symptom:** Intermittent NaN loss at certain sequence lengths or after many
training steps when weights grow. Often misattributed to learning rate or data.

**Detection:** Monitor for `inf`/`nan` in attention logits. Upcast to FP32 before
softmax, or use BF16 (wider dynamic range).

```python
# BUG: fp16 overflow
attn = (Q.half() @ K.half().transpose(-2, -1)) / math.sqrt(d_k)  # can overflow

# FIX: upcast for attention computation
attn = (Q.float() @ K.float().transpose(-2, -1)) / math.sqrt(d_k)
attn = F.softmax(attn, dim=-1).to(Q.dtype)
```

**Source:** [arXiv 2510.04212](https://arxiv.org/html/2510.04212v1)

### 47. Left-padding without correct position IDs

Decoder-only models need left-padding for batched generation, but position IDs must
skip padding positions. A naive `arange(0, seq_len)` gives padding tokens non-zero
positions and shifts real tokens' positions.

**Symptom:** Batched generation gives different (worse) results than single-sample
generation. No error.

**Detection:** Compare single-sample vs batched output.

```python
# BUG: position_ids ignore padding
position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

# FIX: derive from attention_mask
position_ids = attention_mask.long().cumsum(-1) - 1
position_ids.masked_fill_(attention_mask == 0, 1)
```

**Source:** [HuggingFace Issue #22707](https://github.com/huggingface/transformers/issues/22707)

### 48. Pre-norm vs post-norm silent swap (PyTorch default)

PyTorch `nn.Transformer` defaults to `norm_first=False` (post-norm). For deep
networks (12+ layers) without careful warmup, post-norm causes training instability.
Most modern LLMs (GPT-2/3, LLaMA, Mistral) use pre-norm.

**Symptom:** Training diverges, loss spikes, or converges to a worse solution. Often
misattributed to learning rate issues. No error.

**Detection:** Explicitly set `norm_first=True` for pre-norm. For post-norm, verify
Xavier init and learning rate warmup are configured.

**Source:** [PyTorch Issue #55270](https://github.com/pytorch/pytorch/issues/55270)

### 49. Double LayerNorm in PyTorch nn.Transformer

`TransformerEncoder(norm=nn.LayerNorm(...))` applies a final LayerNorm on top of
the last `TransformerEncoderLayer`, which already contains its own LayerNorm.
The output gets normalized twice.

**Symptom:** Slightly degraded performance. Training runs fine. No error.

**Detection:** Inspect `TransformerEncoder`'s `norm` argument. Set to `None` if
using post-norm layers that already normalize their output.

**Source:** [PyTorch Issue #24930](https://github.com/pytorch/pytorch/issues/24930)

### 50. Weight tying mismatch (lm_head vs embed_tokens)

When `tie_word_embeddings=True` in config but checkpoint has separate weights (or
vice versa), loading silently overwrites trained `lm_head` weights with `embed_tokens`
or creates an untied random copy.

**Symptom:** Model outputs gibberish (random lm_head) or subtle quality degradation
(trained lm_head overwritten). No error on load.

**Detection:** `assert model.lm_head.weight.data_ptr() == model.embed_tokens.weight.data_ptr()`.
Check `config.json` for `tie_word_embeddings` and compare with checkpoint keys.

**Source:** [HuggingFace Issue #43404](https://github.com/huggingface/transformers/issues/43404)

### 51. GQA/MQA key-value head expansion on wrong dimension

In Grouped Query Attention, K/V heads must be repeated to match Q heads. If
`repeat_interleave` is applied along the wrong dimension (e.g., sequence instead
of head), heads are silently misaligned.

**Symptom:** Model runs but produces subtly wrong attention patterns. Performance
is worse than expected. No error if shapes happen to broadcast.

**Detection:** Assert `num_heads % num_kv_heads == 0`. Verify the dimension argument
to `repeat_interleave`.

```python
# BUG: wrong repeat dimension (seq instead of head)
kv = kv.repeat_interleave(n_rep, dim=1)

# FIX: repeat along head dimension
kv = kv.repeat_interleave(n_rep, dim=2)  # dim=2 is heads in [B, S, H, D]
```

### 52. Cross-attention Q/K/V source swap

In encoder-decoder cross-attention, queries come from the decoder and keys/values
from the encoder. Swapping these compiles and runs (shapes may be compatible) but
computes the wrong thing.

**Symptom:** Model trains but translation/generation quality is far worse than
expected. No error because tensor shapes may still align.

**Detection:** Audit cross-attention calls:
`attn(query=decoder_hidden, key=encoder_output, value=encoder_output)`.

### 53. Gradient accumulation loss averaging bug (variable-length sequences)

Naive gradient accumulation averages per-step losses equally, but with variable-length
sequences, the correct denominator is total non-padding tokens across all accumulation
steps — not the number of steps. Affects DDP/multi-GPU setups too.

**Symptom:** Higher training loss than equivalent full-batch training. Models with
large gradient accumulation steps converge worse. No error.

**Detection:** Compare loss at gradient_accumulation_steps=1 vs >1 with same
effective batch size.

**Source:** [Unsloth Blog](https://unsloth.ai/blog/gradient),
[HuggingFace Blog](https://huggingface.co/blog/gradient_accumulation)

### 54. Sliding window attention off-by-one / wrong layer assignment

(a) Window of size W masks W+1 or W-1 tokens depending on implementation. (b)
In some models (e.g., Qwen2), sliding window was applied to the wrong layers due
to an inverted condition — only full attention was actually used.

**Symptom:** Model works but loses the efficiency benefit or long-context behavior
of sliding window. Performance degrades on long sequences. No error.

**Detection:** Visualize the attention mask for a short sequence. Verify which
layers use sliding window by inspecting `layer_idx` conditions.

**Source:** [HuggingFace Issue #35896](https://github.com/huggingface/transformers/issues/35896)

### 55. SDPA NaN from fully-masked rows

When padding causes an entire row of the attention matrix to be masked (all keys
masked for a query position), softmax of all `-inf` produces NaN. The memory-efficient
SDPA kernel does not always handle this.

**Symptom:** NaN in attention output for specific batch elements with heavy padding.
May only appear at certain sequence lengths. No error raised.

**Detection:** `assert (attn_mask.sum(dim=-1) > 0).all()`. Handle all-masked rows
by replacing their softmax output with zeros.

**Source:** [PyTorch Issue #103963](https://github.com/pytorch/pytorch/issues/103963)

### 56. Label smoothing silently using wrong decoder_input_ids

In HuggingFace Trainer with `label_smoothing > 0`, labels are removed from the
forward call so the model returns logits only. For seq2seq models, this causes the
model to default to using `input_ids` as `decoder_input_ids` instead of the correct
label-derived decoder inputs.

**Symptom:** Model trains on the wrong target sequence. Metrics degrade but training
loop runs fine. No error.

**Detection:** Ensure `decoder_input_ids` is explicitly passed when using label
smoothing with encoder-decoder models.

**Source:** [HuggingFace Issue #10452](https://github.com/huggingface/transformers/issues/10452)

### 57. Flash Attention numerical divergence at scale

Flash Attention 2 introduces ~0.0005 max error vs standard attention at unit scale.
But when Q/K magnitudes are large (100x), the error amplifies to ~65 in FP16,
producing meaningfully different outputs.

**Symptom:** Outputs differ between Flash Attention and standard attention backends.
Can cause instability when switching backends mid-training. No error.

**Detection:** Compare outputs at the actual scale of your activations, not just
unit scale. Keep activation magnitudes moderate.

**Source:** [Flash Attention Issue #434](https://github.com/Dao-AILab/flash-attention/issues/434)

### 58. Sliding window KV cache eviction boundary bug

With sliding window attention + KV cache, the cache must evict entries outside the
window. If eviction is off-by-one or not triggered at `total_seq_len == window_size`,
tokens attend to stale/wrong cached states.

**Symptom:** Generation degrades specifically at the boundary where sequence length
exceeds the window size. Correct for short sequences, wrong for long ones. No error.

**Detection:** Test generation at `window_size`, `window_size + 1`, and
`2 * window_size`. Compare with non-cached output.

**Source:** [HuggingFace Issue #37574](https://github.com/huggingface/transformers/issues/37574)

### 59. DynamicNTK RoPE scaling computes wrong base frequency

vLLM's DynamicNTKScalingRotaryEmbedding initializes the RoPE base differently from
HuggingFace Transformers. The HF implementation re-computes the base only for inputs
exceeding `max_position_embeddings`, while other implementations apply scaling
unconditionally — degrading performance for inputs within the original context window.

**Symptom:** Performance degradation on sequences that fit within the original
context length. No error.

**Detection:** Compare perplexity on short sequences between your implementation
and the reference. Check whether base frequency computation is conditional on
sequence length.

**Source:** [vLLM Issue #3488](https://github.com/vllm-project/vllm/issues/3488)


---

## Tier 5: Diffusion Model Silent Bugs

### 60. Non-zero terminal SNR — signal leaks at t=T

Most noise schedules (linear, cosine) never reach SNR=0 at the final timestep.
A small amount of signal persists at t=T, so the model never learns to denoise
from pure noise. At sampling time, you start from pure Gaussian noise — a
distribution the model never saw during training.

**Symptom:** Slightly washed-out or biased samples. Mean color of generated images
doesn't match the data. Often attributed to "not enough training."

**Detection:** Check `alpha_cumprod[-1]` — if it's not ~0, there's signal leak.
Use zero-terminal-SNR schedule or enforce `alpha_cumprod[-1] = 0`.

**Sources:** [Lin et al. (WACV 2024, arXiv:2305.08891)](https://arxiv.org/abs/2305.08891),
[arXiv:2309.15842](https://arxiv.org/abs/2309.15842)

### 61. Cosine schedule not adjusted for resolution

The cosine noise schedule was designed for 64x64. At 256x256+, the same schedule
adds too little noise at early timesteps because higher-resolution images have
more redundancy — the model can reconstruct from less noise than expected.

**Symptom:** Model undertrained on low-noise timesteps. Samples lack fine detail
or have slight artifacts. FID is 10-20% worse than expected.

**Detection:** Plot effective SNR vs timestep for your resolution. Shift the schedule
so noise is meaningful at all timesteps for the target resolution.

**Source:** [Chen (2023, arXiv:2301.10972)](https://arxiv.org/abs/2301.10972)

### 62. Off-by-one timestep indexing

`alpha_cumprod[t]` vs `alpha_cumprod[t-1]` — using the wrong index shifts the entire
noise schedule by one step. The noise added during training doesn't match what the
sampler expects.

**Symptom:** Slightly blurry or noisy outputs. FID worse than reference by 5-15%.

**Detection:** Verify `alpha_cumprod[0]` is close to 1 (not 0) and the forward
process at t=0 adds almost no noise.

**Source:** [diffusers Issue #2585](https://github.com/huggingface/diffusers/issues/2585)

### 63. Prediction target mismatch (epsilon vs v vs x0)

Training with epsilon-prediction loss but sampling with x0-prediction formula (or
vice versa). The denoising step applies the wrong reparameterization.

**Symptom:** Model trains normally but sampling produces garbage or very noisy outputs.

**Detection:** Verify `model_output_type` in training matches `prediction_type` in
sampling. Check the loss target tensor against `noise` vs `x_start` vs `v`.

### 64. Uniform loss weighting causes conflicting gradients

Weighting all timesteps equally means high-noise steps (large loss) dominate
gradient magnitude while low-noise steps (fine detail) are underweighted. Gradients
from different timestep ranges can actively conflict.

**Symptom:** Model converges but produces either blurry (low-noise undertrained) or
noisy (high-noise undertrained) samples depending on which timesteps dominate.

**Detection:** Use Min-SNR-gamma weighting to balance gradient magnitudes across
timesteps. Log per-timestep-bin loss and gradient norms.

**Source:** [Min-SNR (arXiv:2303.09556)](https://arxiv.org/abs/2303.09556)

### 65. Missing conditioning dropout for classifier-free guidance

CFG requires training with random conditioning dropout (typically 10% of samples
have their text/class condition replaced with null). Without this, the model has
no unconditional pathway, so CFG at inference has nothing to interpolate from.

**Symptom:** CFG has no effect or produces artifacts. `guidance_scale > 1` doesn't
improve sample quality.

**Detection:** Verify conditioning dropout is applied during training. Check that
`p_uncond` (typically 0.1) is configured in the training script.

### 66. CFG formula applied in wrong direction

Correct: `eps_guided = eps_uncond + scale * (eps_cond - eps_uncond)`. Wrong:
`eps_cond + scale * (eps_uncond - eps_cond)`, which moves AWAY from the condition.

**Symptom:** Higher guidance scale produces worse, less conditioned samples.

**Detection:** At scale=1.0, output should match conditional prediction exactly.
At scale>1.0, outputs should be more aligned with the condition.

```python
# BUG: wrong direction
noise_pred = cond_pred + scale * (uncond_pred - cond_pred)

# FIX
noise_pred = uncond_pred + scale * (cond_pred - uncond_pred)
```

### 67. Missing VAE scaling factor in latent diffusion

Stable Diffusion multiplies VAE latents by 0.18215 (SD 1.x) to bring latent
variance close to 1. Forgetting this means the diffusion model trains on a
distribution with wrong variance.

**Symptom:** Training loss converges but decoded images are distorted or washed out.

**Detection:** Check `latents = vae.encode(x).latent_dist.sample() * scale_factor`
and decoding divides by the same factor.

**Source:** [diffusers Issue #437](https://github.com/huggingface/diffusers/issues/437)

### 68. SDXL VAE numerical instability in FP16

The SDXL VAE produces NaN outputs when run in FP16 due to large intermediate
activations in the decoder. The VAE works fine in FP32.

**Symptom:** NaN or black/corrupted images only when using FP16 VAE decode.
Switching to FP32 for VAE fixes it.

**Detection:** Always run the VAE decode in FP32: `vae.to(dtype=torch.float32)`.

### 69. EMA applied before learning rate warmup

EMA updates during the warmup phase average over random/untrained weights. The
EMA model starts with a poor initialization that takes many more steps to overcome.

**Symptom:** EMA model lags far behind the online model early in training. Even
after convergence, EMA model may be slightly worse than expected.

**Detection:** Start EMA updates only after warmup completes, or use EMA decay
warmup (start with lower decay, increase to target).

### 70. Evaluating non-EMA weights

Generating samples or computing FID with the online (non-EMA) model instead of
the EMA model. The online model has higher-variance weights.

**Symptom:** FID is worse than reported in reference implementations using the
same architecture and training.

**Detection:** Verify evaluation uses `ema_model`, not `model`. Log FID for both.

### 71. Timestep embedding cancelled by normalization

GroupNorm or LayerNorm applied after timestep embedding injection can normalize
away the timestep signal if the embedding is additive and the norm recenters.

**Symptom:** Model partially ignores timestep. Denoising behavior is similar across
different noise levels. Quality degrades but training loss still decreases.

**Detection:** Inject timestep after normalization, or use adaptive normalization
(AdaGN) where timestep modulates scale/shift of the norm layer.

**Source:** [arXiv:2405.14126](https://arxiv.org/abs/2405.14126)

### 72. Unscaled skip connections in deep UNets

As UNet depth increases, skip connections accumulate and the magnitude of
activations grows at each decoder level. Without scaling, deep decoder layers
receive disproportionately large inputs.

**Symptom:** Training instability in deep UNets (20+ blocks). Gradients explode
in early decoder layers.

**Detection:** Scale skip connections by `1/sqrt(2)` at each level, or use
learnable skip connection weights.

**Source:** [ScaleLong (arXiv:2310.13545)](https://arxiv.org/abs/2310.13545)

### 73. Exposure bias — train on ground-truth noise, sample from predictions

During training, each denoising step starts from the exact noisy sample prescribed
by the forward process. During sampling, each step starts from the model's own
(imperfect) prediction. Errors accumulate over the sampling chain.

**Symptom:** Sample quality degrades more than expected as sampling steps decrease.
Long sampling chains partially mask the problem.

**Sources:** [arXiv:2308.15321](https://arxiv.org/abs/2308.15321),
[arXiv:2301.11706](https://arxiv.org/abs/2301.11706)

### 74. DDIM eta parameter misapplied

DDIM with `eta=0` is deterministic, `eta=1` matches DDPM stochasticity. Some
implementations apply eta to the wrong variance term or confuse it with the
noise scaling factor, producing incorrect intermediate noise levels.

**Symptom:** DDIM samples are noisier or less diverse than expected. Quality does
not improve as expected when adjusting eta.

**Detection:** At `eta=0`, running the same latent twice must produce identical
output. At `eta=1`, output should match DDPM quality.

### 75. DPM-Solver++ step size ratio bug with Karras sigmas

DPM-Solver++ assumes a specific relationship between consecutive sigma values.
Karras sigmas use a different spacing than the solver expects, causing the
higher-order corrections to use wrong step size ratios.

**Symptom:** Quality is worse than Euler sampling at the same step count, despite
DPM-Solver++ being theoretically superior.

**Detection:** Verify sigma schedule is compatible with the solver. Match the sigma
schedule to what the solver was derived for.

### 76. UNet skip connection resolution mismatch

Encoder feature map at resolution HxW gets concatenated with decoder feature map at
(H+/-1)x(W+/-1) due to odd input dimensions or stride/padding mismatch.

**Symptom:** Artifacts at image edges. Quality slightly worse than expected.

**Detection:** Print shapes at every skip connection. Use `F.interpolate` to match
sizes explicitly rather than relying on transposed convolution output size.

### 77. Cross-attention in UNet attending to wrong conditioning

In text-to-image diffusion, cross-attention should attend to text embeddings.
Using self-attention everywhere or swapping Q/K sources in cross-attention blocks.

**Symptom:** Model generates images that don't respond to text prompts. Training
loss still decreases because the model learns unconditional generation.

**Detection:** Generate images with very different prompts — if outputs are similar,
text conditioning isn't working.

### 78. Noise prediction residual formula with wrong coefficients

The denoising formula has multiple terms that can be mixed up. Wrong placement
of parentheses or coefficient errors in
`x_{t-1} = (x_t - beta_t/sqrt(1-alpha_bar_t) * eps_pred) / sqrt(alpha_t)`.

**Symptom:** Samples are noisy or blurry even with many steps.

**Detection:** Apply forward then reverse process to known data — should approximately
recover the original.

### 79. Text encoder not frozen during diffusion fine-tuning

Fine-tuning the text encoder alongside the UNet when it should be frozen. The text
encoder adapts to overfit the training captions, degrading generalization.

**Symptom:** Great results on training prompts, poor generalization to new prompts.

**Detection:** Check `text_encoder.requires_grad_(False)`.

### 80. Variance schedule mismatch between training and inference

Training uses `linear` schedule but inference config specifies `scaled_linear` or
a different number of timesteps.

**Symptom:** Samples are noisy, washed out, or oversaturated.

**Detection:** Log `scheduler.config` and compare with training config.

### 81. DPM-Solver order exceeds available steps

DPM-Solver with `order=3` needs at least 3 steps. Using 2 steps with order 3
silently falls back to order 1 in some implementations, or produces wrong results.

**Symptom:** Quality much worse than expected at low step counts.

**Detection:** Verify solver order <= number of steps.

---

## Tier 6: VQ-VAE / VAE Silent Bugs

### 82. Beta applied to wrong loss term in VQ-VAE

The original VQ-VAE paper defines `commitment_cost` multiplying the commitment
loss `||z_e - sg(e)||^2`, but some implementations accidentally apply beta to the
codebook loss `||sg(z_e) - e||^2` instead, or to both.

**Symptom:** Codebook either doesn't update (beta on codebook loss) or encoder
drifts from codebook (beta missing from commitment loss).

**Detection:** Verify beta multiplies only the commitment term. Check loss
components independently.

**Sources:** [taming-transformers Issue #57](https://github.com/CompVis/taming-transformers/issues/57),
[taming-transformers Issue #46](https://github.com/CompVis/taming-transformers/issues/46)

### 83. EMA codebook update computed before quantization

The EMA update should use the encoder outputs assigned to each code AFTER the
nearest-neighbor lookup. If computed before quantization or with stale assignments,
the codebook drifts from the encoder.

**Symptom:** Codebook slowly diverges. Reconstruction quality degrades over training.

**Source:** [pytorch-vq-vae Issue #5](https://github.com/zalandoresearch/pytorch-vq-vae/issues/5)

### 84. Commitment loss growing without bound

Without proper stop-gradient, the commitment loss `||z_e - e||^2` can grow as
encoder and codebook chase each other. The gradient pushes z_e toward e while
codebook EMA pulls e toward z_e, but at different rates.

**Symptom:** Commitment loss increases monotonically during training. Eventually
destabilizes.

**Source:** [vector-quantize-pytorch Issue #27](https://github.com/lucidrains/vector-quantize-pytorch/issues/27)

### 85. Straight-through estimator gradient mismatch

The STE copies gradients from the quantized output to the encoder output unchanged.
But the quantized and unquantized values differ, so the gradient is evaluated at
the wrong point. This bias accumulates.

**Symptom:** Encoder outputs gradually drift from codebook vectors. Training is
stable but subtly suboptimal.

**Detection:** Monitor `||z_e - quantized||` over training — should stay small.
Consider rotation trick for lower-bias STE.

**Source:** [Rotation trick (arXiv:2410.06424)](https://arxiv.org/abs/2410.06424)

### 86. Codebook collapse — all inputs map to same few codes

EMA codebook update with too-high decay (0.999) and small batches. Dead codes
never get reassigned. The model uses 10 out of 1024 codes.

**Symptom:** Reconstruction quality saturates early. Codebook utilization <5%.

**Detection:** Monitor `num_active_codes / total_codes` every epoch. Use codebook
reset, lower EMA decay (0.99), or random restart of dead codes.

**Sources:** [SoundStream](https://arxiv.org/abs/2107.03312),
[lucidrains vector-quantize-pytorch](https://github.com/lucidrains/vector-quantize-pytorch)

### 87. Wrong codebook initialization

Initializing codebook vectors from a standard normal when encoder outputs are in a
very different range (e.g., post-LayerNorm). All inputs initially map to the same
nearest code.

**Symptom:** Slow early training. Many codes start dead and never recover.

**Detection:** Initialize from first batch of encoder outputs or use k-means init.

**Source:** [vector-quantize-pytorch Issue #19](https://github.com/lucidrains/vector-quantize-pytorch/issues/19)

### 88. KL loss uses sum instead of mean (or vice versa)

`KL = sum(kl_per_dim)` gives a loss that scales with latent dimension. If
reconstruction loss uses mean over pixels, the KL/reconstruction ratio changes
when you change latent size, breaking the beta-VAE balance.

**Symptom:** Changing latent dimension drastically changes reconstruction quality
even with the same beta.

**Detection:** Ensure both KL and reconstruction use consistent reduction. Typically
mean over batch, sum over dimensions for KL.

**Source:** [FluxML Issue #73](https://github.com/FluxML/Flux.jl/issues/73)

### 89. Posterior collapse from powerful decoder

With a powerful decoder (e.g., autoregressive, deep transformer), the model learns
to ignore the latent code `z` because the decoder can model the data alone. KL
goes to zero.

**Symptom:** KL loss drops to near-zero early in training. Samples from the prior
look like unconditional model outputs, not conditioned on z.

**Detection:** Monitor KL per dimension. Use cyclical annealing, free bits, or a
weaker decoder.

**Source:** [Cyclical annealing (arXiv:1903.10145)](https://arxiv.org/abs/1903.10145)

### 90. Reparameterization trick: variance vs log-variance confusion

VAE encoder outputs `(mu, sigma)` but code treats second output as `log_var` (or
vice versa). Using `sigma` where `log_var` is expected means `exp(sigma/2)` produces
huge variance.

**Symptom:** Training unstable or posterior immediately collapses.

**Detection:** Check `z = mu + exp(log_var * 0.5) * eps`. Print std of sampled z
— should be order 1, not huge or tiny.

### 91. VQ-GAN adaptive weight produces NaN

The adaptive weight `lambda = grad_rec / grad_gan` balances reconstruction and
adversarial loss. When `grad_gan` is near zero (early training or bad discriminator),
the weight explodes to inf/NaN.

**Symptom:** NaN loss after discriminator warmup period ends.

**Detection:** Clamp the adaptive weight: `lambda = min(grad_rec / (grad_gan + eps), max_weight)`.

### 92. VQ-GAN discriminator starts too early

PatchGAN discriminator activated from step 0 overwhelms the generator before it
can learn basic reconstruction.

**Symptom:** Reconstruction quality degrades or never improves. Generator converges
to adversarial noise.

**Detection:** Start discriminator after N warmup steps (typically 10k-50k). Monitor
generator and discriminator loss balance.

### 93. Expanded distance formula produces negative distances

The expanded L2 formula `||a-b||^2 = ||a||^2 - 2*a*b + ||b||^2` can produce small
negative values due to floating point errors, which become NaN under sqrt.

**Symptom:** NaN in codebook lookup. Intermittent, depends on input magnitudes.

**Detection:** Clamp distances: `dist = torch.clamp(dist, min=0.0)`. Or use
`torch.cdist` which handles this internally.

**Source:** [sonnet Issue #158](https://github.com/google-deepmind/sonnet/issues/158)

### 94. DDP codebook EMA desync across ranks

Each GPU computes local codebook usage counts and embedding sums. Without
`all_reduce`, the EMA update uses partial statistics, causing codebook drift
between ranks.

**Symptom:** Model works on single GPU but quality degrades in multi-GPU training.

**Detection:** Add `all_reduce` on counts and embedding sums before EMA update.
Compare codebook utilization across ranks.

**Source:** [vq-vae-2 Issue #45](https://github.com/rosinality/vq-vae-2-pytorch/issues/45)

### 95. FP16 distance computation overflow in codebook lookup

With FP16, the expanded distance formula `||a||^2 + ||b||^2 - 2*a*b` can overflow
when codebook vectors have large magnitudes (FP16 max ~65504).

**Symptom:** NaN in quantization layer. Happens intermittently as codebook vectors
grow during training.

**Detection:** Compute distances in FP32 even when training in FP16.

### 96. FSQ activation collapse

Finite Scalar Quantization (FSQ) uses bounded activations (e.g., tanh) to
discretize. If the encoder learns to always output values near quantization
boundaries, effective codebook size drops.

**Symptom:** Low effective codebook utilization despite using FSQ correctly.

**Detection:** Monitor distribution of pre-quantization activations per dimension.
They should cover the full range, not cluster at boundaries.

**Source:** [iFSQ (arXiv:2601.17124)](https://arxiv.org/abs/2601.17124)

### 97. Gumbel-Softmax temperature not annealed

Fixed high temperature keeps codes soft (blurry reconstruction). Fixed low
temperature causes gradient vanishing through the Gumbel-Softmax.

**Symptom:** High temp: blurry outputs. Low temp: training stalls.

**Detection:** Anneal temperature from ~1.0 to ~0.1 over training.

### 98. Perceptual + GAN + reconstruction loss scale not calibrated

Three loss terms with different magnitudes. If reconstruction dominates, output is
blurry. If GAN dominates, output has artifacts.

**Symptom:** Systematic artifacts (too smooth, too sharp, or wrong textures).

**Detection:** Log each loss term separately. Adjust weights so each contributes
meaningfully.

### 99. Beta-VAE schedule resets on resume

KL annealing schedule restarts from 0 after checkpoint resume, causing a second
collapse period.

**Symptom:** Quality dip after resume, followed by slow recovery.

**Detection:** Save and restore the current beta value / schedule step in checkpoint.

---

## Tier 7: GAN-Specific Silent Bugs

### 100. Missing detach on fake samples for discriminator training

Must detach fake samples when training D to prevent gradients flowing into G
through the D update.

**Symptom:** Training is unstable. G and D losses oscillate wildly.

```python
# BUG: G gets gradients from D update
d_fake = discriminator(generator(z))

# FIX
d_fake = discriminator(generator(z).detach())
```

### 101. WGAN weight clipping collapses to trivially simple functions

WGAN with weight clipping forces discriminator weights into [-c, c]. With small c,
the discriminator can only represent very simple functions and provides poor
gradients to the generator.

**Symptom:** Generated samples are blurry, low quality despite stable training.
Discriminator weights cluster at +c and -c (bimodal distribution).

**Detection:** Use gradient penalty (WGAN-GP) instead of weight clipping.

**Source:** [WGAN-GP paper (arXiv:1704.00028)](https://arxiv.org/abs/1704.00028)

### 102. WGAN-GP without `create_graph=True` in gradient penalty

The gradient penalty requires second-order gradients. Without `create_graph=True`
in `torch.autograd.grad`, the penalty value is correct but its gradient w.r.t.
model params is zero. The penalty has no training effect.

**Symptom:** Training behaves as if gradient penalty is absent. Lipschitz constraint
not enforced.

```python
# BUG: penalty gradient is zero
grad = torch.autograd.grad(d_interp.sum(), x_hat)[0]

# FIX: enable higher-order gradients
grad = torch.autograd.grad(d_interp.sum(), x_hat, create_graph=True)[0]
```

### 103. StyleGAN AdaIN produces water-droplet artifacts

Adaptive Instance Normalization in StyleGAN1 creates characteristic blob artifacts
because it normalizes per-feature-map statistics, which the generator exploits to
smuggle signal through the normalization.

**Symptom:** Periodic blob/droplet artifacts in generated images, especially at
higher resolutions.

**Detection:** Use weight demodulation (StyleGAN2) instead of AdaIN.

**Source:** [StyleGAN2 paper (arXiv:1912.04958)](https://arxiv.org/abs/1912.04958)

### 104. Spectral normalization vectors not persisted across forward passes

Spectral norm requires maintaining the u/v vectors across iterations for the
power iteration method to converge. If vectors are reinitialized each forward
pass, the spectral norm estimate is inaccurate.

**Symptom:** Training instability despite using spectral norm. Lipschitz constant
not properly controlled.

**Detection:** Verify u/v vectors are registered as buffers with `register_buffer`.

**Source:** [SN-GAN paper (arXiv:1802.05957)](https://arxiv.org/abs/1802.05957)

### 105. Missing EMA on generator weights

Most modern GANs (StyleGAN, BigGAN) use EMA of generator weights for evaluation.
Without EMA, the generator's fast-moving weights produce noisier samples.

**Symptom:** FID is 5-20% worse than reference implementations. Quality fluctuates
between evaluations.

**Detection:** Add EMA on generator weights. Compare FID with and without EMA.

### 106. Generator output range doesn't match discriminator input range

Generator outputs tanh [-1, 1] but training images are [0, 1], or vice versa.
The discriminator learns a trivial feature (value range) to distinguish real/fake.

**Symptom:** Discriminator immediately achieves near-perfect accuracy. Generator
loss doesn't decrease.

**Detection:** Verify real and fake images have identical value ranges.

### 107. BatchNorm in discriminator leaks batch statistics

BatchNorm creates correlations between samples in a batch, allowing the
discriminator to exploit batch-level statistics rather than individual samples.

**Symptom:** Discriminator accuracy is high but generator can't learn.

**Detection:** Use LayerNorm, InstanceNorm, or SpectralNorm in the discriminator.

### 108. Conditional GAN label leaking through batch structure

In conditional GANs, if real images of the same class are always grouped together,
the discriminator learns batch structure rather than per-image quality.

**Symptom:** Discriminator accuracy is high but generated images don't match
conditions.

**Detection:** Shuffle class labels within each batch.

### 109. Generator and discriminator update ratio wrong

Too many D steps: D is too strong, G gets no useful gradient (mode collapse).
Too few D steps: D can't keep up, G gets meaningless gradients.

**Symptom:** Mode collapse or training oscillation.

**Detection:** Monitor D(real) and D(fake) — both should be in a reasonable range,
not saturated at 0 or 1.

---

## Tier 8: Reinforcement Learning Silent Bugs

### 110. Not distinguishing terminal vs truncated episodes

An episode ending because the agent reached a terminal state (done=True) is
fundamentally different from truncation due to time limit. Terminal states have
V(s')=0, but truncated states do not. Treating both as terminal biases the value
function to underestimate states near time limits.

**Symptom:** Agent avoids states that tend to occur near time limits. Policy is
suboptimal in a way that correlates with episode length.

**Detection:** Use separate `terminated` and `truncated` flags. For truncated
episodes, bootstrap: `target = r + gamma * V(s')`.

**Source:** [Stable-Baselines3 Issue #651](https://github.com/DLR-RM/stable-baselines3/issues/651)

### 111. PPO advantage normalization at wrong granularity

Normalizing advantages per mini-batch instead of per rollout. This re-centers
advantages within each mini-batch, changing the effective policy gradient direction
between mini-batches from the same rollout.

**Symptom:** Training is less stable than expected. Performance is sensitive to
mini-batch size.

**Detection:** Normalize advantages across the entire rollout buffer, not per
mini-batch.

**Source:** [37 Implementation Details of PPO](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)

### 112. PPO value function clipping uses max instead of min

The clipped value loss should take the maximum of clipped and unclipped losses
(pessimistic bound). Using min instead makes the clipping have no effect.

**Symptom:** Value function updates are too aggressive, destabilizing training.

**Detection:** Verify: `value_loss = max(unclipped_loss, clipped_loss)`.

**Source:** [Ray Issue #19291](https://github.com/ray-project/ray/issues/19291)

### 113. DQN overestimation bias from using same network for selection and evaluation

Using the same Q-network to both select and evaluate actions: `target = r + gamma * max_a Q(s', a)`.
The max operator introduces systematic positive bias in Q-value estimates.

**Symptom:** Q-values grow much larger than actual returns. Policy appears to plateau
despite increasing Q-values.

**Detection:** Use Double DQN: select action with online network, evaluate with
target network.

### 114. PPO per-token clipping silences rare tokens in LLM fine-tuning

When applying PPO to language model fine-tuning (RLHF), per-token ratio clipping
disproportionately clips rare tokens whose probability changed more. The model
quickly stops exploring vocabulary.

**Symptom:** Vocabulary diversity drops during RLHF training. Model converges to
a narrow set of common tokens.

**Detection:** Monitor per-token clip fraction. Consider sequence-level KL penalty
instead of per-token clipping.

### 115. Discount factor too low makes distant rewards invisible

With gamma=0.9, a reward 50 steps in the future is discounted by 0.9^50 = 0.005.
Effectively invisible. The agent becomes extremely myopic.

**Symptom:** Agent optimizes only for immediate rewards and ignores long-horizon
objectives.

**Detection:** Compute the effective horizon: `1/(1-gamma)`. With gamma=0.9 the
horizon is 10 steps. Use gamma=0.99 (horizon 100) or gamma=0.999 (horizon 1000).

### 116. EnvPool returns wrong observation on episode boundary

Some vectorized environment implementations (EnvPool) return the first observation
of the new episode when an episode ends, overwriting the terminal observation
before the agent can use it for value estimation.

**Symptom:** Value function is wrong at episode boundaries. Subtle performance
degradation.

**Detection:** Store terminal observation in info dict before auto-reset overwrites it.

### 117. Action space normalization mismatch

Environment expects actions in [-1, 1] but policy outputs unbounded values.
Actions are silently clipped by the environment.

**Symptom:** Agent can't reach extreme actions. Converges to suboptimal behavior.

**Detection:** Log raw policy outputs vs environment action bounds. Use tanh
squashing with log-prob correction.

### 118. Replay buffer sampling across episode boundaries

Sampling consecutive transitions that cross episode boundaries treats the first
state of a new episode as the next state of the previous episode's terminal state.

**Symptom:** Value function produces incorrect estimates near episode boundaries.

**Detection:** Store done flags and mask terminal transitions:
`next_value = (1 - done) * V(next_state)`.

### 119. Entropy bonus not scaled with action dimension

Fixed entropy coefficient regardless of action dimensionality. High-dimensional
action spaces need different entropy scaling.

**Symptom:** In high-dim: premature entropy collapse. In low-dim: entropy stays
too high.

**Detection:** Monitor entropy relative to maximum entropy. Use automatic entropy
tuning (SAC-style).

---

## Tier 9: Graph Neural Network Silent Bugs

### 120. GAT computes static attention (not dependent on query)

The original GAT's attention mechanism `a^T [Wh_i || Wh_j]` decomposes into
separate terms for source and target — the ranking of attention weights for
node i's neighbors is independent of node i's features.

**Symptom:** Attention weights are less expressive than expected. Performance
plateaus below what dynamic attention achieves.

**Detection:** Use GATv2 which computes `a^T LeakyReLU(W[h_i || h_j])`, making
attention truly query-dependent.

**Source:** [GATv2 (arXiv:2105.14491)](https://arxiv.org/abs/2105.14491)

### 121. Over-smoothing from too many GNN layers

After 5+ message-passing layers, all node representations converge to the same
value. Deeper is not better in GNNs.

**Symptom:** Adding layers initially helps, then accuracy degrades. All node
embeddings become similar.

**Detection:** Monitor cosine similarity between all node pairs per layer. If it
approaches 1.0, use 2-3 layers, skip connections, or GCNII.

**Source:** [arXiv:1909.03211](https://arxiv.org/abs/1909.03211)

### 122. Oversquashing from fixed-width bottleneck

Information from exponentially many neighbors must pass through a fixed-width
node vector. Distant information is exponentially compressed.

**Symptom:** Model fails on tasks requiring long-range graph reasoning despite
using many layers.

**Detection:** Use graph rewiring, virtual nodes, or graph transformers for
long-range dependencies.

**Source:** [arXiv:2411.17429](https://arxiv.org/abs/2411.17429)

### 123. Batching without incrementing edge indices

In mini-batched GNNs (PyG), edge indices must be offset by cumulative node count.
If the offset is wrong or missing, edges connect nodes across different graphs.

**Symptom:** Node features leak between graphs in a batch. Accuracy looks higher
than single-graph evaluation.

**Detection:** Use PyG `DataLoader` (handles offsets automatically). Verify
`batch.edge_index.max() < batch.x.size(0)`.

### 124. Laplacian eigenvector sign ambiguity

Laplacian eigenvectors are only defined up to sign (both v and -v are valid).
If the sign is randomly assigned per graph, positional encodings are inconsistent.

**Symptom:** Positional encoding-based models (e.g., Transformer + Laplacian PE)
learn slowly or fail to generalize across graphs.

**Detection:** Use SignNet or sign-invariant architectures for Laplacian PE.

**Source:** [arXiv:2411.12732](https://arxiv.org/abs/2411.12732)

### 125. Missing self-loops in GCN

Standard GCN adds self-loops `A_hat = A + I` before normalization. Without them,
each node's own features are not included in the aggregation.

**Symptom:** Accuracy 5-10% worse than expected.

**Detection:** `edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))`.

### 126. Message passing direction reversed in directed graphs

Messages should flow from source to target (or vice versa depending on convention).
Using the wrong direction reverses information flow.

**Symptom:** Model trains but performance doesn't improve, especially for tasks
where directionality matters.

**Detection:** Verify PyG convention: `edge_index[0]` = source, `edge_index[1]` = target.

### 127. Normalization coefficient wrong for adjacency matrix

GCN uses symmetric `D^{-1/2} A D^{-1/2}`. Using random-walk `D^{-1} A` or no
normalization gives different results. High-degree nodes dominate without proper
normalization.

**Symptom:** Model works but accuracy differs from reference.

**Detection:** Verify normalization matches the paper being implemented.

### 128. Feature leakage through global pooling in node classification

Using `global_mean_pool` for node classification leaks label information from
other nodes.

**Symptom:** Accuracy is suspiciously high. Drops on unseen nodes.

**Detection:** Node classification should use node-level outputs, not graph-level
pooling.

### 129. GNN assumes undirected graph but receives directed edges

Many GNN implementations add reverse edges internally. If the user already provides
both directions, edges are doubled, changing effective normalization.

**Symptom:** Feature magnitudes larger than expected. Performance slightly differs
from reference.

**Detection:** Check if framework auto-converts to undirected. If so, provide only
one direction per edge.

---

## Tier 10: Object Detection Silent Bugs

### 130. Box coordinate format mismatch (xyxy vs xywh vs cxcywh)

Different frameworks use different box formats. Mixing them up silently produces
wrong IoU calculations, wrong NMS behavior, and wrong loss gradients.

**Symptom:** mAP is near zero despite loss converging. Or model partially works
but struggles with box position accuracy.

**Detection:** Assert format explicitly. Encode a known box, decode it, verify
coordinates match.

### 131. NMS suppresses valid detections of nearby distinct objects

NMS assumes overlapping detections are duplicates. In crowded scenes (pedestrian
detection, cell counting), nearby distinct objects are suppressed.

**Symptom:** Missing detections in crowded scenes. Recall drops significantly on
dense benchmarks.

**Detection:** Use Soft-NMS or set-prediction methods (DETR) for crowded scenarios.

### 132. DETR bipartite matching instability / slow convergence

Hungarian matching is unstable early in training when predictions are random —
the matching flips between assignments every step, preventing consistent gradient
signal.

**Symptom:** DETR takes 300+ epochs to converge (vs ~100 for anchor-based methods).
Loss is noisy for many epochs.

**Detection:** Use deformable attention (Deformable DETR) for faster convergence,
or DN-DETR with denoising training for stable matching.

### 133. DETR object queries with no spatial prior

Learned object queries in vanilla DETR have no initial spatial bias. Each query
must learn from scratch which spatial region to attend to.

**Symptom:** Very slow convergence. Some queries specialize to similar regions
(redundant), others never activate.

**Detection:** Use anchor-based queries (DAB-DETR) or conditional spatial queries.

### 134. Anchor-based detectors fail on small objects due to IoU discretization

Small objects have very few possible IoU values with anchors (e.g., a 4x4 object
can have IoU of 0, 0.25, 0.5, 0.75, or 1.0 with a same-size anchor). The IoU
threshold assigns most anchors as negative.

**Symptom:** Small object AP is much worse than large object AP.

**Detection:** Use lower IoU threshold for small objects, or switch to anchor-free
detection.

### 135. FPN feature dilution across levels

Repeated downsampling and upsampling in FPN dilutes features. By the time
high-level semantic features reach P3, they've lost significant information.

**Symptom:** Detection at lower FPN levels (small objects) is weaker than expected.

**Detection:** Use BiFPN or PANet for better cross-level feature fusion.

### 136. Background/ignore regions counted as false positives

Regions marked as "ignore" or "crowd" in annotations counted as false positives
during evaluation, artificially lowering AP.

**Symptom:** mAP is lower than expected, especially on COCO.

**Detection:** Verify evaluation code handles `iscrowd` flag. Use `pycocotools`.

### 137. FPN feature level assignment off-by-one

Objects assigned to the wrong FPN level. Small objects at P5 instead of P3.

**Symptom:** Small object AP much worse, large object AP unaffected.

**Detection:** Verify assignment formula: `level = floor(k0 + log2(sqrt(wh)/224))`.

### 138. Image resize changes aspect ratio without adjusting box coordinates

Resizing training images to square without adjusting box coordinates proportionally
per dimension.

**Symptom:** Boxes are shifted/stretched. Model learns wrong coordinates.

**Detection:** Visualize boxes overlaid on resized images.

---

## Tier 11: Segmentation Silent Bugs

### 139. Skip connection size mismatch from unpadded convolutions

In U-Net, if convolutions don't use padding, spatial dimensions shrink at each
layer. The encoder feature map size doesn't match the decoder at the same level.

**Symptom:** Runtime error (if explicit), or silent crop/pad that loses boundary
information.

**Detection:** Print shapes at every skip connection. Use `padding='same'` or
center-crop encoder features to match decoder.

### 140. Transposed convolution checkerboard artifacts

Transposed convolutions with stride that doesn't evenly divide the kernel size
produce overlapping receptive fields, creating a checkerboard pattern in the output.

**Symptom:** Regular grid-like artifacts in segmentation masks or upsampled features.

**Detection:** Use `nn.Upsample` + regular convolution instead of transposed conv,
or ensure kernel_size is divisible by stride.

**Source:** [Distill — Deconvolution and Checkerboard Artifacts](https://distill.pub/2016/deconv-checkerboard/)

### 141. Cross-entropy dominated by background class

In semantic segmentation, background class often represents 90%+ of pixels. Standard
cross-entropy optimization focuses on getting background right and ignores foreground.

**Symptom:** High overall pixel accuracy but near-zero IoU for small/rare classes.

**Detection:** Use class-weighted CE, focal loss, or Dice loss.

### 142. Overlapping instances merged in binary segmentation

Using binary (foreground/background) segmentation for instance segmentation.
Touching instances are merged into one connected component.

**Symptom:** Instance count is systematically lower than ground truth in crowded
scenes.

**Detection:** Use instance segmentation (Mask R-CNN) or add boundary class/distance
transform for separation.

### 143. Ignore index not handled in Dice loss

`CrossEntropyLoss(ignore_index=255)` handles ignore pixels, but custom Dice loss
often doesn't filter them.

**Symptom:** Dice score lower than expected. Model wastes capacity on ignore regions.

**Detection:** Mask out ignore pixels before computing Dice: `mask = (target != 255)`.

### 144. `align_corners` mismatch in upsampling

`F.interpolate(align_corners=True)` vs `False` produces different results at
boundaries. Mismatch between training and inference shifts predictions.

**Symptom:** Segmentation boundaries consistently offset by ~1 pixel. mIoU is
1-2% lower than expected.

**Detection:** Explicitly set `align_corners` consistently in all resize operations.

### 145. Argmax before upsampling (quantization artifact)

Applying argmax at low resolution then upsampling the class map produces blocky
boundaries.

**Symptom:** Segmentation boundaries are blocky/aliased.

**Detection:** Upsample logits first, then apply argmax at full resolution.

### 146. Multi-class Dice loss computes global instead of per-class

Global Dice computes intersection/union across all classes together. Large classes
dominate.

**Symptom:** Large classes have good IoU, small classes have near-zero IoU.

**Detection:** Use per-class Dice (compute per class, then average).

---

## Tier 12: Seq2Seq / Text Generation Silent Bugs

### 147. Beam search length normalization includes prompt tokens

Length penalty divides log-probability by total sequence length including the
prompt. Longer prompts produce lower per-token scores, biasing beam search toward
shorter completions.

**Symptom:** Generated completions are systematically shorter with longer prompts.

**Detection:** Normalize by generation length only, excluding prompt tokens.

**Source:** [vLLM Issue #2606](https://github.com/vllm-project/vllm/issues/2606)

### 148. Beam search performance degrades with large beam width

Beyond a certain width (typically 5-10), beam search produces worse outputs than
smaller beams. Larger beams converge to high-probability but low-quality sequences.

**Symptom:** Increasing beam width past a point degrades BLEU/quality.

**Detection:** Sweep beam width and plot quality metric. Use beam width 4-5 as default.

**Source:** [ICML 2019 — On NMT Search Errors](https://arxiv.org/abs/1904.09751)

### 149. Exposure bias — train with teacher forcing, evaluate autoregressively

During training, each step receives the ground-truth previous token. During
inference, each step receives the model's own (potentially wrong) prediction.
Errors compound.

**Symptom:** Low training loss but poor generation quality. BLEU lower than expected.

**Detection:** Compare teacher-forced validation loss with autoregressive generation
quality. Use scheduled sampling to reduce the gap.

### 150. Padding tokens attended in decoder cross-attention

Cross-attention in decoder attends to padded positions in encoder output. Padding
tokens contribute meaningless values to the context vector.

**Symptom:** Performance worse with longer sequences (more padding). Batched
inference differs from single-sample.

**Detection:** Pass encoder padding mask to cross-attention.

**Source:** [HuggingFace Issue #19581](https://github.com/huggingface/transformers/issues/19581)

### 151. BOS token not prepended to decoder input

Decoder input should start with BOS token. If first target token is used instead,
the model never learns to generate the first token from scratch.

**Symptom:** First generated token is often wrong.

**Detection:** Verify `decoder_input_ids[0] = bos_token_id`.

### 152. EOS token not handled in beam search

Beams that produce EOS should be marked finished. If they continue, the model
generates garbage past the intended endpoint.

**Symptom:** Generated sequences longer than expected with garbage after EOS.

**Detection:** Verify finished beams are stored separately and not extended.

### 153. Source and target share vocabulary but use different token IDs

Using separate tokenizers with the same vocabulary but different ID mappings.
Embeddings are silently misaligned.

**Symptom:** Translation quality much worse than expected.

**Detection:** Verify `tokenizer_src("hello") == tokenizer_tgt("hello")`.

### 154. Label smoothing double-applied with framework loss

Using label smoothing in both the model and the training framework (e.g.,
HuggingFace Trainer). Effective smoothing is stronger than intended.

**Symptom:** Model outputs are overly uncertain. BLEU lower than expected.

**Detection:** Check label smoothing is applied in exactly one place.

---

## Tier 13: Contrastive Learning Silent Bugs

### 155. False negatives in batch — positive pairs treated as negatives

In in-batch contrastive learning, two samples from the same class (or two
augmentations of the same underlying data) can end up as negatives. The model is
penalized for correctly recognizing similarity.

**Symptom:** Representations are less effective for downstream tasks despite good
contrastive accuracy. Hurts more with large batches.

**Source:** [WACV 2022](https://openaccess.thecvf.com/content/WACV2022/html/Chuang_Debiased_Contrastive_Learning_WACV_2022_paper.html)

### 156. SimCLR with small batch size

SimCLR's in-batch negatives are less diverse with small batches. Below ~256
effective negatives, the contrastive objective becomes too easy and representations
are poor.

**Symptom:** Downstream transfer performance is much worse than reported with
same architecture but smaller batch.

**Detection:** Use MoCo (memory bank) if batch size is limited, or increase batch
size with gradient accumulation.

### 157. CLIP logit_scale overflow without clamping

CLIP's learned temperature `logit_scale = exp(t)` can grow without bound during
training. Large values cause overflow in the similarity matrix.

**Symptom:** Training loss becomes NaN after many steps. Temperature grows
monotonically.

**Detection:** Clamp logit_scale: `self.logit_scale.data.clamp_(max=4.6052)` (= ln(100)).

### 158. Dimensional collapse — representations use only a subspace

Representations collapse to a low-dimensional subspace. Loss looks normal because
contrast still works in the subspace.

**Symptom:** High contrastive accuracy but poor downstream transfer. Representation
covariance matrix has many near-zero eigenvalues.

**Detection:** Monitor effective rank of the representation matrix. Use feature
decorrelation (Barlow Twins, VICReg).

**Source:** [arXiv:2311.05139](https://arxiv.org/abs/2311.05139)

### 159. Stop-gradient missing in BYOL/SimSiam

BYOL and SimSiam require stop-gradient on one branch to prevent collapse. Without
it, both branches collapse to a trivial constant.

**Symptom:** Loss drops immediately to zero. All representations become identical.

**Detection:** Verify `stop_gradient` is applied to the target branch output.

### 160. Incomplete augmentation pipeline

Missing key augmentations (e.g., no color jitter in SimCLR) allows the model to
solve the contrastive task using shortcut features (color histogram matching)
instead of semantic features.

**Symptom:** High contrastive accuracy but poor downstream transfer. Model learns
to match low-level statistics.

**Detection:** Test downstream transfer. Ablate each augmentation to verify it
contributes.

### 161. Multi-GPU contrastive learning without cross-GPU negatives

Each GPU only uses in-batch negatives from its own batch. With 4 GPUs, you have
4x fewer negatives than the effective batch size.

**Symptom:** Performance is worse with more GPUs (opposite of expected).

**Detection:** Gather representations across all GPUs:
`torch.distributed.all_gather()`.

### 162. Feature normalization applied at wrong point

L2-normalizing before the projection head constrains its input. Normalizing after
(before contrastive loss) is standard. Doing both or neither changes the loss.

**Symptom:** Neither: loss dominated by feature magnitude. Both: projection head
is overconstrained.

**Detection:** Normalize after projection head only:
`z = F.normalize(projector(encoder(x)), dim=-1)`.

### 163. Momentum encoder not updated or updated too fast (MoCo)

`m=0.999` works well. `m=0.9`: momentum encoder tracks too closely (no benefit).
Update skipped: performance degrades as key encoder diverges.

**Symptom:** m too low: no improvement over SimCLR. Skipped: degradation over time.

**Detection:** Verify momentum update runs every step.

---

## Tier 14: NeRF / 3D Reconstruction Silent Bugs

### 164. Quadrature instability in volume rendering

Numerical quadrature for volume rendering accumulates errors along the ray. With
piecewise-constant density approximation, small changes in sample positions cause
large changes in rendered color.

**Symptom:** Training is noisy. Rendered images flicker between evaluation steps.

**Detection:** Use more samples per ray, or piecewise-linear (zip-NeRF style)
integration for stability.

**Source:** [arXiv:2310.20685](https://arxiv.org/abs/2310.20685)

### 165. Point-sampled positional encoding causes aliasing (mip-NeRF)

Standard NeRF encodes a single 3D point with high-frequency PE. But a pixel's
footprint covers a cone, not a point. Fine details alias when the cone is larger
than the PE frequency.

**Symptom:** Renders have aliasing artifacts at different scales. Multi-scale
consistency is poor.

**Detection:** Use integrated positional encoding (mip-NeRF) that encodes the full
cone volume instead of a point.

**Source:** [mip-NeRF (arXiv:2103.13415)](https://arxiv.org/abs/2103.13415)

### 166. NDC space applied to non-forward-facing scenes

Normalized Device Coordinate space linearizes depth for forward-facing scenes.
Applying NDC to 360-degree scenes severely distorts geometry behind the camera.

**Symptom:** Geometry is correct for the front hemisphere but severely distorted
or missing for rear views.

**Detection:** Only use NDC for forward-facing captures. For 360 scenes, use
world-space coordinates with contraction (mip-NeRF 360).

### 167. White vs black background mismatch

Training with white background but rendering against black (or vice versa).
The NeRF learns to compensate by making empty space produce the wrong background
color.

**Symptom:** Semi-transparent objects or colored halos around geometry.

**Detection:** Match background color between training and rendering. Use random
backgrounds during training for better separation.

**Source:** [gaussian-splatting Issue #1038](https://github.com/graphdeco-inria/gaussian-splatting/issues/1038)

### 168. Camera coordinate convention mismatch (OpenGL vs OpenCV)

OpenGL uses Y-up, OpenCV uses Y-down. Using the wrong convention flips or rotates
rendered images.

**Symptom:** Rendered images are upside down, mirrored, or rotated 180 degrees.

**Detection:** Render a simple scene with known orientation. Convert between
conventions: flip Y and Z axes of camera-to-world matrices.

### 169. Ray direction not normalized

Ray directions must be unit vectors for correct distance calculations. Unnormalized
rays cause incorrect depth estimation and density integration.

**Symptom:** Geometry is stretched or compressed. Objects at different distances
are distorted.

**Detection:** `assert torch.allclose(rays_d.norm(dim=-1), torch.ones(...))`.

### 170. Near/far plane bounds too tight or too loose

Near plane too far: front geometry clipped. Far plane too close: background clipped.
Both too loose: samples wasted on empty space.

**Symptom:** Missing geometry or very blurry renders.

**Detection:** Set near/far from the dataset's camera parameters. Visualize sample
locations along rays.

### 171. Density activation uses wrong function

NeRF uses ReLU for density. Mip-NeRF uses softplus. Sigmoid caps density at 1.0,
preventing opaque surfaces.

**Symptom:** Semi-transparent renders. Objects look ghostly.

**Detection:** Verify density activation allows unbounded positive values.

### 172. Hierarchical sampling ignores coarse network weights

The fine network should sample more densely where the coarse network predicted high
density. If fine network uses uniform sampling, hierarchical scheme provides no
benefit.

**Symptom:** No improvement from adding hierarchical sampling.

**Detection:** Visualize sample locations from hierarchical sampling — they should
cluster near surfaces.

### 173. 3D Gaussian Splatting — gradient for position uses wrong Jacobian

The projection Jacobian for Gaussian mean position must account for the perspective
divide. If the Jacobian ignores depth or uses an affine approximation too far from
the center, position gradients are wrong.

**Symptom:** Gaussians slowly drift from correct positions. Reconstruction quality
plateaus below the expected PSNR.

**Detection:** Compare numerical and analytic gradients for the projection.

---

## Tier 15: Flow Matching / Normalizing Flow Silent Bugs

### 174. SD3 logit-normal timestep sampling bug

Stable Diffusion 3 uses logit-normal timestep sampling to focus training on
intermediate timesteps. An implementation bug shifted the mean, causing too much
weight on near-noise timesteps and too little on near-data timesteps.

**Symptom:** Model undertrained on fine details (low-noise regime). Samples lack
sharpness.

**Detection:** Plot actual timestep distribution and compare with intended
logit-normal.

**Source:** [diffusers Issue #8534](https://github.com/huggingface/diffusers/issues/8534)

### 175. Euler solver on curved ODE paths

Flow matching with optimal transport produces curved probability paths. Euler
solver (straight-line per step) has high truncation error on curved paths,
requiring many more steps than theoretically necessary.

**Symptom:** Quality improves dramatically with more steps but is bad at 10-20 steps
where other methods work well.

**Detection:** Compare Euler vs higher-order solvers (RK45, midpoint). If quality
difference is large, the path is too curved for Euler.

### 176. Wrong velocity at boundary — v diverges at t=1

For some flow matching formulations, the target velocity `v_t = (x_1 - x_t) / (1-t)`
diverges as t approaches 1. If the model is evaluated at t=1, the output is garbage.

**Symptom:** ODE solver produces NaN or extreme values near t=1. Final step corrupts
the sample.

**Detection:** Clamp t to [0, 1-eps] or use a formulation that handles the boundary.

### 177. Straight-line interpolation direction reversed

`x_t = (1-t)*noise + t*data` vs `x_t = t*noise + (1-t)*data`. Reversed direction
means the model learns the wrong velocity field.

**Symptom:** Model trains but generates noise instead of data (or vice versa).

**Detection:** At t=0, x_t should be noise. At t=1, x_t should be data.

### 178. Residual flow not enforcing Lipschitz constraint

Continuous normalizing flows with residual blocks must enforce Lipschitz constant < 1
for invertibility. Without spectral normalization or other constraints, the flow
may not be invertible.

**Symptom:** `f_inverse(f(x)) != x`. Sampling produces out-of-distribution outputs.

**Detection:** Test round-trip: `assert (f_inverse(f(x)) - x).abs().max() < 1e-4`.

### 179. Jacobian log-determinant computed incorrectly (normalizing flows)

For normalizing flows, the log-determinant must be exact. Wrong sign, missing
absolute value, or incorrect formula produces wrong log-likelihoods.

**Symptom:** Log-likelihood values are wrong (possibly positive for continuous
distributions). Model trains but sampling produces wrong distribution.

**Detection:** Verify with numerical Jacobian on small inputs.

### 180. Hutchinson trace estimator with too few samples

Stochastic trace estimation (for FFJORD) with one sample per step has high
variance. The model optimizes a noisy objective.

**Symptom:** Training is unstable. Log-likelihood estimates are noisy.

**Detection:** Use >=5 Hutchinson samples. Compare with exact trace on small models.

### 181. Invertibility broken in coupling layers

Coupling layer must be exactly invertible. Numerical errors in affine transforms
`exp(s)*x + t` / `(x - t) / exp(s)` accumulate through many layers.

**Symptom:** `f_inverse(f(x)) != x`. Sampling produces blurry outputs.

**Detection:** Test round-trip reconstruction error.

### 182. Time conditioning range mismatch

Flow matching uses t in [0, 1] but model receives t in [0, T] (diffusion convention).
The model learns the wrong velocity at each time point.

**Symptom:** ODE integration produces wrong distribution.

**Detection:** Verify t range matches training convention.

---

## Tier 16: Cross-Domain Silent Bugs

### 183. Object detection positive/negative sampling ratio too extreme

If positive:negative ratio is 1:1000, the detector learns to predict "background"
for everything.

**Symptom:** High precision but zero recall.

**Detection:** Use hard negative mining or focal loss. Log the positive/negative ratio.

### 184. GAN eval mode — generator in train mode during FID

Generating FID samples with generator in training mode (dropout active, BN using
batch stats).

**Symptom:** FID is noisy between evaluations and higher than expected.

**Detection:** `generator.eval()` before generating FID samples.

### 185. RL reward normalization dividing by zero early in training

Normalizing rewards by running std when buffer has too few samples. Std is
near-zero, causing normalized rewards to explode.

**Symptom:** Huge policy updates at start of training.

**Detection:** `reward_norm = reward / max(running_std, eps)`.

### 186. GNN edge dropout disconnects graph

Randomly dropping edges can disconnect the graph, creating isolated nodes that
receive no messages.

**Symptom:** Training is unstable. Some nodes have zero-valued representations.

**Detection:** Verify connectivity after dropout, or handle isolated nodes explicitly.

### 187. Contrastive evaluation with different augmentations than training

Evaluating with augmentations that differ from training. Features are aligned
to training augmentations.

**Symptom:** Lower downstream accuracy than expected.

**Detection:** Use standard evaluation augmentations (center crop, resize only).

### 188. Object detection evaluation at wrong IoU threshold

COCO uses AP@[0.5:0.95], PASCAL VOC uses AP@0.5. Wrong threshold makes results
non-comparable.

**Symptom:** Scores don't match reported baselines.

**Detection:** Verify evaluation protocol matches the benchmark standard.

### 189. NeRF background model not separated from foreground

Without separate background modeling, the density field wastes capacity representing
infinite background.

**Symptom:** Background appears as floating blobs. Foreground less sharp.

**Detection:** Use separate background model or mask background rays.

### 190. Flow matching with wrong noise distribution

Using Gaussian noise when model was derived for uniform (or vice versa). The
probability path doesn't match the velocity field.

**Symptom:** Model trains but generated distribution doesn't match data.

**Detection:** Verify source distribution matches the flow matching derivation.

### 191. RL discount factor applied at wrong level

Applying gamma to rewards instead of value estimates, or in both places (double
discounting).

**Symptom:** Agent over-values or under-values future rewards.

**Detection:** Verify `return = reward + gamma * V(next_state)`, not
`return = gamma * reward + gamma * V(next_state)`.

---

## Sources

### General PyTorch / Deep Learning
- [Tanel Parnamaa — NumPy worker seed bug](https://tanelp.github.io/posts/a-bug-that-plagues-thousands-of-open-source-ml-projects/)
- [Zach Mueller — Gradient accumulation reproducibility](https://muellerzr.github.io/blog/gradient_accumulation_part2.html)
- [Aditya Rana — PyTorch weight initialization](https://adityassrana.github.io/blog/theory/2020/08/26/Weight-Init.html)
- [Elana Simon — MPS backend silent bug](https://elanapearl.github.io/blog/2025/the-bug-that-taught-me-pytorch/)
- [ppwwyyxx — Silent bugs in DL libraries](https://ppwwyyxx.com/blog/2020/Fight-Against-Silent-Bugs-in-Deep-Learning-Libraries/)
- [TrainCheck (OSDI 2025)](https://arxiv.org/abs/2506.14813)
- [NVIDIA — Silent Data Corruption in AI](https://www.opencompute.org/documents/sdc-in-ai-ocp-whitepaper-final-pdf)
- [Mindee — Danger of Batch Normalization](https://www.mindee.com/blog/batch-normalization)
- [arXiv 2510.04212 — Low-precision transformer training failures](https://arxiv.org/html/2510.04212v1)

### Transformer-Specific
- [DeepSpeed PR #5664 — Sequence parallel reshape fix](https://github.com/deepspeedai/DeepSpeed/pull/5664)
- [Diffusers Issue #10303 — head_to_batch_dim permutation bug](https://github.com/huggingface/diffusers/issues/10303)
- [PyTorch Issue #108108 — SDPA causal mask alignment](https://github.com/pytorch/pytorch/issues/108108)
- [PyTorch Issue #124464 — SDPA dropout ignores eval mode](https://github.com/pytorch/pytorch/issues/124464)
- [PyTorch Issue #26338 — F.dropout eval mode behavior](https://github.com/pytorch/pytorch/issues/26338)
- [PyTorch Issue #92554 — Boolean vs float attention mask](https://github.com/pytorch/pytorch/issues/92554)
- [PyTorch Issue #55270 — Transformer pre/post norm](https://github.com/pytorch/pytorch/issues/55270)
- [PyTorch Issue #24930 — Double LayerNorm](https://github.com/pytorch/pytorch/issues/24930)
- [PyTorch Issue #103963 — SDPA NaN from masked rows](https://github.com/pytorch/pytorch/issues/103963)
- [HuggingFace Issue #25199 — LLaMA RoPE mismatch](https://github.com/huggingface/transformers/issues/25199)
- [HuggingFace Issue #31859 — RoPE interleave mismatch](https://github.com/huggingface/transformers/issues/31859)
- [HuggingFace Issue #29149 — Position IDs in generation](https://github.com/huggingface/transformers/issues/29149)
- [HuggingFace Issue #22707 — Batch vs single generation](https://github.com/huggingface/transformers/issues/22707)
- [HuggingFace Issue #43404 — Weight tying mismatch](https://github.com/huggingface/transformers/issues/43404)
- [HuggingFace Issue #35896 — Qwen2 sliding window wrong layers](https://github.com/huggingface/transformers/issues/35896)
- [HuggingFace Issue #37574 — SWA KV cache boundary](https://github.com/huggingface/transformers/issues/37574)
- [HuggingFace Issue #10452 — Label smoothing decoder_input_ids](https://github.com/huggingface/transformers/issues/10452)
- [HuggingFace Troubleshooting — Padding without attention mask](https://huggingface.co/docs/transformers/troubleshooting)
- [Unsloth — Gradient accumulation loss fix](https://unsloth.ai/blog/gradient)
- [HuggingFace Blog — Fixing gradient accumulation](https://huggingface.co/blog/gradient_accumulation)
- [Flash Attention Issue #434 — Numerical divergence at scale](https://github.com/Dao-AILab/flash-attention/issues/434)
- [vLLM Issue #3488 — DynamicNTK RoPE mismatch](https://github.com/vllm-project/vllm/issues/3488)

### Diffusion Models
- [Lin et al. (WACV 2024) — Common Diffusion Noise Schedules are Flawed (arXiv:2305.08891)](https://arxiv.org/abs/2305.08891)
- [Signal leak bias (arXiv:2309.15842)](https://arxiv.org/abs/2309.15842)
- [Chen (2023) — Resolution-dependent noise schedule (arXiv:2301.10972)](https://arxiv.org/abs/2301.10972)
- [diffusers Issue #2585 — Off-by-one timestep indexing](https://github.com/huggingface/diffusers/issues/2585)
- [Min-SNR weighting (arXiv:2303.09556)](https://arxiv.org/abs/2303.09556)
- [diffusers Issue #437 — VAE scaling factor](https://github.com/huggingface/diffusers/issues/437)
- [Timestep embedding cancelled by normalization (arXiv:2405.14126)](https://arxiv.org/abs/2405.14126)
- [ScaleLong — Unscaled skip connections (arXiv:2310.13545)](https://arxiv.org/abs/2310.13545)
- [Exposure bias in diffusion (arXiv:2308.15321)](https://arxiv.org/abs/2308.15321)
- [Exposure bias in diffusion (arXiv:2301.11706)](https://arxiv.org/abs/2301.11706)
- [Stable Diffusion VAE scaling factor](https://github.com/CompVis/stable-diffusion)

### VQ-VAE / VAE
- [taming-transformers Issue #57 — Beta applied to wrong loss](https://github.com/CompVis/taming-transformers/issues/57)
- [taming-transformers Issue #46 — VQ loss formulation](https://github.com/CompVis/taming-transformers/issues/46)
- [pytorch-vq-vae Issue #5 — EMA update ordering](https://github.com/zalandoresearch/pytorch-vq-vae/issues/5)
- [vector-quantize-pytorch Issue #27 — Commitment loss growing](https://github.com/lucidrains/vector-quantize-pytorch/issues/27)
- [Rotation trick for STE (arXiv:2410.06424)](https://arxiv.org/abs/2410.06424)
- [vector-quantize-pytorch Issue #19 — Codebook initialization](https://github.com/lucidrains/vector-quantize-pytorch/issues/19)
- [FluxML Issue #73 — KL sum vs mean](https://github.com/FluxML/Flux.jl/issues/73)
- [Cyclical annealing for posterior collapse (arXiv:1903.10145)](https://arxiv.org/abs/1903.10145)
- [sonnet Issue #158 — Negative distances in expanded formula](https://github.com/google-deepmind/sonnet/issues/158)
- [vq-vae-2 Issue #45 — DDP codebook EMA desync](https://github.com/rosinality/vq-vae-2-pytorch/issues/45)
- [SoundStream (arXiv:2107.03312)](https://arxiv.org/abs/2107.03312)
- [iFSQ — FSQ activation collapse (arXiv:2601.17124)](https://arxiv.org/abs/2601.17124)

### GANs
- [WGAN-GP paper (arXiv:1704.00028)](https://arxiv.org/abs/1704.00028)
- [StyleGAN2 paper (arXiv:1912.04958)](https://arxiv.org/abs/1912.04958)
- [SN-GAN — Spectral Normalization (arXiv:1802.05957)](https://arxiv.org/abs/1802.05957)

### Reinforcement Learning
- [Stable-Baselines3 Issue #651 — Terminal vs truncated](https://github.com/DLR-RM/stable-baselines3/issues/651)
- [37 Implementation Details of PPO (ICLR Blog Track)](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)
- [Ray Issue #19291 — PPO value function clipping](https://github.com/ray-project/ray/issues/19291)

### Graph Neural Networks
- [GATv2 (arXiv:2105.14491)](https://arxiv.org/abs/2105.14491)
- [Over-smoothing (arXiv:1909.03211)](https://arxiv.org/abs/1909.03211)
- [Oversquashing (arXiv:2411.17429)](https://arxiv.org/abs/2411.17429)
- [Laplacian eigenvector sign ambiguity (arXiv:2411.12732)](https://arxiv.org/abs/2411.12732)

### Object Detection
- [Distill — Deconvolution and Checkerboard Artifacts](https://distill.pub/2016/deconv-checkerboard/)

### Seq2Seq / Text Generation
- [vLLM Issue #2606 — Beam search length normalization](https://github.com/vllm-project/vllm/issues/2606)
- [ICML 2019 — On NMT Search Errors](https://arxiv.org/abs/1904.09751)
- [HuggingFace Issue #19581 — Padding in cross-attention](https://github.com/huggingface/transformers/issues/19581)

### Contrastive Learning
- [WACV 2022 — Debiased Contrastive Learning (False negatives)](https://openaccess.thecvf.com/content/WACV2022/html/Chuang_Debiased_Contrastive_Learning_WACV_2022_paper.html)
- [Dimensional collapse (arXiv:2311.05139)](https://arxiv.org/abs/2311.05139)
- [SimCLR (ICML 2020)](https://arxiv.org/abs/2002.05709)
- [MoCo (CVPR 2020)](https://arxiv.org/abs/1911.05722)

### NeRF / 3D Reconstruction
- [Quadrature instability (arXiv:2310.20685)](https://arxiv.org/abs/2310.20685)
- [mip-NeRF — Point-sampled PE aliasing (arXiv:2103.13415)](https://arxiv.org/abs/2103.13415)
- [gaussian-splatting Issue #1038 — Background color mismatch](https://github.com/graphdeco-inria/gaussian-splatting/issues/1038)

### Flow Matching / Normalizing Flows
- [diffusers Issue #8534 — SD3 logit-normal timestep bug](https://github.com/huggingface/diffusers/issues/8534)
- [FFJORD (ICLR 2019)](https://arxiv.org/abs/1810.01367)

### Segmentation
- [mmsegmentation Issue #3172 — Dice loss ignore_index](https://github.com/open-mmlab/mmsegmentation/issues/3172)

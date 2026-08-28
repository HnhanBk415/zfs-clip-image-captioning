# ZeroCap End-to-End — GPT-2 Base

> Quyết định đã chốt: **ZeroCap dùng GPT-2 Base (`gpt2`) + CLIP ViT-B/32** để đồng bộ backbone với ClipCap và phục vụ comparison công bằng hơn. Trong report nên gọi là **ZeroCap adaptation / matched-backbone ZeroCap**, không phải exact reproduction tuyệt đối của bản gốc.

## 1. Mục tiêu hệ thống

Input:

```text
raw image hoặc image_id
```

Output:

```python
{
    "image_id": "...jpg",
    "caption": "A dog running through a grassy field.",
    "generated_token_ids": [...],
    "num_generated_tokens": 9,
    "stop_reason": "period",
    "generation_time_sec": 12.5,
    "end_to_end_time_sec": 12.8,
    "clip_model": "openai/clip-vit-base-patch32",
    "gpt_model": "gpt2"
}
```

Ý tưởng cốt lõi:

```text
GPT-2 Base
→ biết sinh ngôn ngữ

CLIP ViT-B/32
→ biết text nào hợp với ảnh

ZeroCap
→ dùng CLIP guidance để optimize GPT-2 context tại inference
→ không train GPT-2
→ không train CLIP
→ không dùng ground-truth caption khi generate
```

## 2. Backbone

```text
GPT-2: gpt2
hidden size = 768

CLIP: openai/clip-vit-base-patch32
image/text embedding dim = 512
```

Lý do chọn Base:

```text
ClipCap  → GPT-2 Base
ZeroCap  → GPT-2 Base
```

Như vậy comparison tập trung vào:

```text
ClipCap:
few-shot
train Mapping Network

ZeroCap:
zero-shot
optimize context tại inference
```

chứ không bị nhiễu bởi khác biệt model size.

## 3. Cấu hình ban đầu

```python
gpt_model_name = "gpt2"
clip_model_name = "openai/clip-vit-base-patch32"

prompt = "Image of a"
max_new_tokens = 15

top_k = 512
inner_iterations = 5

clip_temperature = 0.01
clip_loss_scale = 1.0
fluency_weight = 0.2

step_size = 0.3
grad_norm_factor = 0.9

fusion_factor = 0.99
beam_size = 5

reset_context_delta = True
stop_token = "."
seed = 0
```

Debug profile:

```python
top_k = 5
inner_iterations = 1
beam_size = 1
max_new_tokens = 3
```

## 4. Folder structure

```text
src/
└── zerocap/
    ├── __init__.py
    ├── types.py
    ├── config.py
    ├── model_loader.py
    ├── image_encoder.py
    ├── clip_guidance.py
    ├── context_optimizer.py
    ├── decoding.py
    ├── generator.py
    └── captioner.py

scripts/
├── run_zerocap.py
├── benchmark_zerocap.py
└── evaluate_zerocap.py

tests/
├── test_zerocap_models.py
├── test_zerocap_image.py
├── test_zerocap_context.py
├── test_zerocap_guidance.py
├── test_zerocap_generation.py
└── test_zerocap_integration.py
```

## 5. End-to-end flow

```text
INPUT IMAGE
   ↓
1. Load Frozen CLIP + GPT-2 Base
   ↓
2. Raw image → CLIP preprocessing
   ↓
3. Frozen CLIP Image Encoder
   ↓
E_img [1,512]
   ↓
4. Normalize E_img
   ↓
5. Tokenize prompt "Image of a"
   ↓
6. OUTER LOOP sinh từng token
   ↓
7. Tính P_original từ GPT-2
   ↓
8. Lấy past_key_values từ current caption
   ↓
9. Tạo context_delta ΔC
   ↓
10. INNER LOOP optimize context
   ↓
11. past_guided = past_original + ΔC
   ↓
12. GPT-2 forward với guided context
   ↓
P_guided
   ↓
13. Lấy Top-K candidate tokens
   ↓
14. Ghép full current caption + candidate token
   ↓
15. CLIP Text Encoder
   ↓
candidate text embeddings
   ↓
16. Cosine similarity với E_img
   ↓
17. Tạo CLIP target distribution
   ↓
18. Tính CLIP guidance loss
   ↓
19. Tính fluency loss
   ↓
20. Total loss
   ↓
21. Backward chỉ vào ΔC
   ↓
22. Normalized-gradient update ΔC
   ↓
23. Lặp N inner iterations
   ↓
24. Final guided forward
   ↓
25. Special-token / repetition rules
   ↓
26. Fusion P_guided + P_original
   ↓
27. Beam-search chọn next token
   ↓
28. Append token
   ↓
29. Stop?
      ├─ No → quay lại OUTER LOOP
      └─ Yes
           ↓
30. Hoàn thành các beam caption
   ↓
31. CLIP rerank final beams
   ↓
32. Decode + bỏ prompt
   ↓
CAPTION
   ↓
33. Runtime + diagnostics
```

## 6. STEP 1 — Load và freeze models

Load:

```text
GPT-2 Base
GPT-2 tokenizer
CLIP ViT-B/32
CLIP processor/tokenizer
```

Sau đó:

```python
gpt_model.eval()
clip_model.eval()

for p in gpt_model.parameters():
    p.requires_grad = False

for p in clip_model.parameters():
    p.requires_grad = False
```

Output:

```text
Frozen GPT-2 Base
Frozen CLIP
```

PASS:

```text
all GPT-2 params requires_grad == False
all CLIP params requires_grad == False
models nằm đúng device
```

## 7. STEP 2 — Raw image → CLIP input

Input:

```text
PIL.Image
```

Qua CLIP processor:

```text
resize
crop
normalize
tensorize
```

Output:

```text
pixel_values
≈ [1,3,224,224]
```

PASS:

```text
finite
batch size = 1
```

## 8. STEP 3 — Image → E_img

```text
pixel_values
↓
Frozen CLIP Image Encoder
↓
E_img [1,512]
```

Normalize:

```python
E_img = E_img / E_img.norm(dim=-1, keepdim=True)
```

PASS:

```text
shape = [1,512]
finite
norm ≈ 1
```

Nếu dùng cache:

```text
lookup bằng image_id
```

Không dùng:

```python
features[:1]
```

## 9. STEP 4 — Prompt

```text
"Image of a"
```

Tokenize:

```python
current_ids = gpt_tokenizer(
    "Image of a",
    return_tensors="pt"
).input_ids
```

Output:

```text
current_ids [1,seq_len]
```

Prompt không chứa thông tin ảnh.

## 10. STEP 5 — OUTER LOOP

Mỗi outer iteration sinh một token:

```python
for token_step in range(max_new_tokens):
    ...
```

Flow:

```text
optimize context
↓
next-token distribution
↓
chọn token
↓
append
↓
check stop
```

## 11. STEP 6 — P_original

GPT-2 chạy bình thường với caption hiện tại:

```text
current caption
↓
Frozen GPT-2
↓
P_original(next token)
```

Ví dụ:

```text
man    0.30
woman  0.18
dog    0.07
car    0.06
```

`P_original` đại diện cho language prior gốc.

Phải detach vì nó là reference.

## 12. STEP 7 — `past_key_values`

Chia current tokens:

```text
prefix = tất cả token trừ token cuối
last_token = token cuối
```

Ví dụ:

```text
Image | of | a

prefix = Image of
last_token = a
```

Chạy:

```text
prefix
↓
GPT-2
↓
past_key_values = C_original
```

Với GPT-2 Base:

```text
12 transformer layers
hidden size = 768
12 attention heads
head dim = 64
```

Logic:

```text
Layer 0
├─ Key
└─ Value

Layer 1
├─ Key
└─ Value

...

Layer 11
├─ Key
└─ Value
```

## 13. STEP 8 — Tạo `context_delta`

Tạo `ΔC` có cùng structure với `C_original`.

Ban đầu:

```text
ΔC = 0
```

Mỗi Key/Value delta:

```text
requires_grad=True
```

Guided context:

```text
C_guided = C_original + ΔC
```

Optimize:

```text
ΔC ✅
```

Không optimize:

```text
GPT-2 weights ❌
CLIP weights ❌
E_img ❌
token IDs ❌
```

## 14. STEP 9 — INNER LOOP

Mỗi token optimize context N lần:

```text
inner iteration 1
inner iteration 2
...
inner iteration N
```

Ví dụ:

```text
N = 5
```

Mục tiêu:

```text
tìm ΔC giúp next-token distribution phù hợp ảnh hơn
```

## 15. STEP 10 — Guided GPT-2 forward

```text
C_original + ΔC
        +
last_token
   ↓
Frozen GPT-2
   ↓
P_guided
```

Ban đầu:

```text
ΔC = 0
→ P_guided gần P_original
```

Sau optimization:

```text
P_guided được CLIP kéo về nội dung ảnh
```

## 16. STEP 11 — Top-K candidates

Từ:

```text
P_guided [vocab_size]
```

lấy:

```text
Top-K token IDs
```

Final-like:

```text
K = 512
```

Debug:

```text
K = 5 hoặc 10
```

Ví dụ:

```text
man
dog
car
woman
person
```

## 17. STEP 12 — Full candidate captions

Sai:

```text
"dog"
"man"
"car"
```

Đúng:

```text
current caption + candidate token
```

Ví dụ:

```text
Current:
"Image of a"

Candidates:
"Image of a dog"
"Image of a man"
"Image of a car"
```

## 18. STEP 13 — CLIP Text Encoder

Input:

```text
K candidate captions
```

Output:

```text
E_text [K,512]
```

Normalize:

```python
text_features = text_features / text_features.norm(
    dim=-1,
    keepdim=True
)
```

PASS:

```text
finite
shape đúng
norm≈1
```

## 19. STEP 14 — Cosine similarity

Ta có:

```text
E_img [1,512]
E_text [K,512]
```

Tính:

```python
similarities = E_text @ E_img.T
```

Ví dụ:

```text
"Image of a dog"   0.75
"Image of a man"   0.21
"Image of a car"   0.17
```

CLIP đóng vai trò semantic judge.

## 20. STEP 15 — CLIP target distribution

```text
similarities
↓
/ temperature
↓
softmax
↓
target_topk
```

Ví dụ:

```text
dog   0.88
man   0.07
car   0.03
```

Sau đó:

```text
target_vocab [vocab_size]
```

ban đầu toàn 0.

Chỉ Top-K positions có probability.

PASS:

```text
target_vocab.sum() ≈ 1
```

## 21. STEP 16 — CLIP guidance loss

Concept:

```python
loss_clip = -(
    target_vocab * log(P_guided)
).sum()
```

Mục tiêu:

```text
P_guided
→ gần distribution CLIP muốn
```

Nếu CLIP thích `"dog"` mà GPT-2 vẫn thích `"man"`:

```text
loss_clip lớn
```

## 22. STEP 17 — Fluency loss

Giữ guided distribution không lệch quá xa GPT-2 gốc:

```text
KL(P_guided || P_original)
```

Concept:

```python
loss_fluency = (
    P_guided
    * (
        log(P_guided)
        - log(P_original)
    )
).sum()
```

Ý nghĩa:

```text
CLIP loss
→ đúng ảnh

Fluency loss
→ vẫn tự nhiên như GPT-2
```

## 23. STEP 18 — Total loss

```text
L_total
=
clip_loss_scale × L_clip
+
fluency_weight × L_fluency
```

Ví dụ:

```text
clip_loss_scale = 1.0
fluency_weight = 0.2
```

Không có ground-truth Flickr8k caption trong loss.

## 24. STEP 19 — Backward

```python
loss_total.backward()
```

Gradient mong muốn:

```text
context_delta.grad ✅
```

Không mong muốn:

```text
GPT-2 parameter grad ❌
CLIP parameter grad ❌
```

PASS:

```text
delta có grad
grad finite
model weights không đổi
```

## 25. STEP 20 — Normalized-gradient update

Không dùng Adam nếu muốn bám ZeroCap logic.

Với từng Key/Value delta của từng layer:

```python
normalized_grad = (
    grad /
    (
        grad.norm() ** grad_norm_factor
        + eps
    )
)

delta_new = (
    delta
    - step_size * normalized_grad
)
```

Sau mỗi inner iteration:

```text
detach delta cũ
↓
delta mới
↓
requires_grad=True
```

Không giữ graph của iteration trước.

PASS:

```text
delta_new finite
```

## 26. STEP 21 — Lặp inner loop

Ví dụ minh họa:

```text
Iteration 1
P(dog) = 0.07

Iteration 2
P(dog) = 0.18

Iteration 3
P(dog) = 0.33

Iteration 4
P(dog) = 0.49

Iteration 5
P(dog) = 0.61
```

Sau N vòng:

```text
optimized ΔC*
```

## 27. STEP 22 — Final guided forward

```text
C_final = C_original + ΔC*
```

Run GPT-2:

```text
last token + C_final
↓
P_guided_final
```

Đây là guided distribution dùng cho decoding.

## 28. STEP 23 — Special token/repetition rules

Trước khi chọn token:

```text
forbidden-token suppression
repetition penalty
không cho "." xuất hiện quá sớm
tăng khả năng "." khi caption đủ dài
```

Mục tiêu:

```text
không token rác
không lặp vô hạn
không dừng quá sớm
không caption quá dài
```

## 29. STEP 24 — Probability fusion

Có:

```text
P_guided
P_original
```

Fusion:

```text
log P_final
=
fusion_factor × log P_guided
+
(1-fusion_factor) × log P_original
```

Ví dụ:

```text
fusion_factor = 0.99
```

Sau đó normalize:

```text
P_final.sum() ≈ 1
```

## 30. STEP 25 — Beam search

```text
beam_size = 5
```

Giữ:

```text
5 caption hypotheses tốt nhất
```

Mỗi beam gồm:

```text
token sequence
score
past/context state
stop state
```

Mỗi outer iteration:

```text
expand beam
↓
score candidates
↓
keep top beam_size
```

## 31. STEP 26 — Append token

Ví dụ:

```text
Before:
"Image of a"

Selected:
"dog"

After:
"Image of a dog"
```

Sau đó quay lại outer loop.

## 32. STEP 27 — Stop condition

Dừng nếu gặp:

```text
"."
EOS
max_new_tokens
```

Lưu:

```text
stop_reason
```

Ví dụ:

```text
period
eos
max_tokens
```

## 33. STEP 28 — Final CLIP reranking

Beam search có thể tạo:

```text
Beam 1:
"Image of a dog running in grass."

Beam 2:
"Image of a dog standing outside."

Beam 3:
"Image of a man with a dog."
```

CLIP encode từng full caption.

Tính:

```text
similarity(full_caption, image)
```

Chọn caption có CLIP similarity cao nhất.

## 34. STEP 29 — Decode + strip prompt

Generated:

```text
"Image of a dog running through a grassy field."
```

Có thể trả:

```text
"A dog running through a grassy field."
```

Output chính:

```text
caption: str
```

## 35. STEP 30 — Runtime

Nên báo hai loại.

Generation time:

```text
cached E_img
→ caption
```

End-to-end time:

```text
raw image
→ CLIP preprocessing
→ image encoding
→ ZeroCap generation
→ caption
```

CUDA timing:

```python
torch.cuda.synchronize()
start = time.perf_counter()

...

torch.cuda.synchronize()
elapsed = time.perf_counter() - start
```

## 36. Public interface

```python
captioner = ZeroCapCaptioner(config)

result = captioner.generate_caption(
    image=image,
    image_id=image_id
)
```

Output:

```python
{
    "image_id": "123.jpg",
    "caption": "A dog running through a grassy field.",
    "generated_token_ids": [...],
    "num_generated_tokens": 9,
    "stop_reason": "period",
    "generation_time_sec": 12.5,
    "end_to_end_time_sec": 12.8,
    "clip_model": "openai/clip-vit-base-patch32",
    "gpt_model": "gpt2"
}
```

## 37. Data integration

VAL/TEST lấy từ fixed split.

Generation chỉ đọc:

```text
image
image_id
```

Không đọc reference captions.

Ground truth chỉ được đọc ở evaluator.

Nếu dùng cached feature:

```text
image_id
↓ lookup
E_img
```

Tuyệt đối không dùng:

```python
features[:1]
```

## 38. Milestone triển khai

```text
M0  Load + freeze model
M1  1 image → E_img
M2  Prompt → P_original
M3  past_key_values
M4  context_delta cùng structure
M5  delta có gradient
M6  Top-K + full candidate captions
M7  CLIP similarities
M8  CLIP target distribution
M9  CLIP + fluency loss
M10 normalized gradient update
M11 generate 1 token
M12 generate 3 tokens debug
M13 full caption
M14 beam search
M15 final CLIP reranking
M16 1 raw VAL image end-to-end
M17 benchmark 5 VAL images
M18 fixed TEST
M19 evaluation
```

## 39. Tests

```text
Models:
- GPT-2 frozen
- CLIP frozen
- device đúng

Image:
- E_img shape [1,512]
- finite
- norm≈1

Context:
- past_key_values đúng structure
- delta cùng structure
- delta gradients finite

Guidance:
- candidate text chứa current caption
- CLIP similarities finite
- target sum≈1
- target chỉ non-zero trong Top-K

Loss:
- CLIP loss finite
- fluency loss finite
- total loss finite

Update:
- delta thay đổi
- model weights không đổi

Probability:
- P_original sum≈1
- P_guided sum≈1
- P_final sum≈1

Generation:
- caption non-empty
- không loop vô hạn
- stop reason hợp lệ
```

## 40. Benchmark 5 VAL images

Sau khi 1 ảnh end-to-end PASS:

```text
warm-up 1 ảnh
↓
không tính
↓
5 VAL images
↓
caption + runtime
```

Lưu:

```text
image_id
caption
generation_time_sec
end_to_end_time_sec
num_generated_tokens
stop_reason
```

## 41. Final TEST

Sau khi tune xong trên VAL:

```text
fixed TEST
↓
ZeroCap inference
↓
save predictions
```

Không tune lại bằng TEST.

ClipCap và ZeroCap phải chạy cùng test image IDs.

## 42. Evaluation

Metrics:

```text
BLEU-1
BLEU-4
METEOR
ROUGE-L
CIDEr
SPICE
CLIPScore hoặc RefCLIPScore
```

Runtime:

```text
average
median
p95
```

Ngoài ra lưu:

```text
caption lỗi/rỗng
stop reason
predictions
full config
seed
model names
```

## 43. So sánh với ClipCap

```text
                 CLIP ViT-B/32
                      +
                  GPT-2 Base
                       │
             ┌─────────┴─────────┐
             ↓                   ↓
          ZeroCap             ClipCap
          zero-shot           few-shot
             │                   │
 optimize context ΔC       train Mapper
 at inference time          1/5/10/25/100%
             │                   │
             └─────────┬─────────┘
                       ↓
                 SAME TEST SET
```

## 44. Khác biệt optimize giữa hai nhánh

ClipCap:

```text
Input training:
image embedding + GT caption

Optimize:
Mapping Network parameters

Output:
trained mapper checkpoint
```

ZeroCap:

```text
Input inference:
E_img
+ current caption
+ P_original
+ past_key_values

Optimize:
context_delta ΔC

Output inner optimization:
optimized context
→ guided next-token distribution

Output whole system:
caption
```

## 45. Definition of Done

```text
[ ] GPT-2 Base frozen
[ ] CLIP frozen
[ ] Raw image → E_img
[ ] E_img normalized
[ ] Fixed VAL/TEST integration
[ ] Prompt tokenize đúng
[ ] P_original đúng
[ ] past_key_values lấy được
[ ] context_delta cùng structure
[ ] delta có gradient
[ ] GPT-2 weights không update
[ ] CLIP weights không update
[ ] Top-K candidates đúng
[ ] Candidate text chứa full current caption
[ ] CLIP similarities finite
[ ] CLIP target sum = 1
[ ] CLIP target chỉ non-zero trong Top-K
[ ] CLIP loss finite
[ ] KL(P_guided || P_original) finite
[ ] Normalized-gradient update chạy
[ ] Inner loop N iterations chạy
[ ] Final guided distribution hợp lệ
[ ] Forbidden/repetition rules chạy
[ ] Fusion probability sum = 1
[ ] Beam search chạy
[ ] Stop condition chạy
[ ] Final CLIP reranking chạy
[ ] Full caption non-empty
[ ] Runtime recorded
[ ] 1 raw VAL image end-to-end PASS
[ ] 5 VAL benchmark PASS
[ ] Fixed TEST generation PASS
[ ] Evaluation metrics được lưu
[ ] generate_caption(image) ready for API
```

## 46. Thứ tự code chính thức

```text
STEP 1  config.py + model_loader.py
STEP 2  image_encoder.py
STEP 3  inspect past_key_values
STEP 4  context_delta utilities
STEP 5  clip_guidance.py
STEP 6  loss functions
STEP 7  context_optimizer.py
STEP 8  generate one token
STEP 9  decoding rules + fusion
STEP 10 beam search
STEP 11 outer autoregressive generator
STEP 12 final CLIP reranking
STEP 13 ZeroCapCaptioner
STEP 14 1-image integration test
STEP 15 5-image benchmark
STEP 16 fixed TEST evaluation
STEP 17 FastAPI integration
```

## 47. Một câu nhớ toàn bộ

```text
Ảnh
↓
CLIP embedding
↓
GPT-2 Base đề xuất token
↓
CLIP chấm candidate captions
↓
loss
↓
gradient chỉnh past-key/value context delta
↓
guided probability
↓
fusion + beam search
↓
append token
↓
lặp
↓
full caption
↓
CLIP rerank
↓
caption + runtime
```

## 48. Bước tiếp theo

Backbone đã khóa:

```text
GPT-2 Base
CLIP ViT-B/32
```

Nên bước code tiếp theo:

```text
STEP 1:
config.py
+
model_loader.py
+
test freeze/device
```

PASS rồi mới sang:

```text
STEP 2:
raw image
→ E_img [1,512]
```

Sau đó:

```text
STEP 3:
prompt
→ GPT-2 Base
→ inspect past_key_values
```

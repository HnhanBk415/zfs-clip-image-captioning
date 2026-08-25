# ZeroCap End-to-End — Giải thích cặn kẽ từ setup đến evaluation

> Mục tiêu của tài liệu này: giúp hiểu **ZeroCap từ bản chất**, biết **mỗi bước nhận gì → làm gì → tại sao → đầu ra là gì → bước sau dùng gì**, rồi triển khai thành pipeline end-to-end hoàn chỉnh.

---

# 1. ZeroCap là gì?

ZeroCap giải bài toán **zero-shot image captioning**.

Ta có hai model pretrained:

```text
CLIP
→ hiểu mối liên hệ giữa ảnh và text
→ biết câu nào hợp với ảnh

GPT-2
→ sinh ngôn ngữ tốt
→ nhưng không nhìn được ảnh
```

ZeroCap kết hợp hai model:

```text
GPT-2 đề xuất cách viết
        +
CLIP chấm xem câu nào hợp ảnh
        ↓
ZeroCap dùng gradient để điều chỉnh context của GPT-2
        ↓
sinh caption từng token
```

Điểm quan trọng:

```text
Không train trên Flickr8k
Không dùng Mapper của ClipCap
Không update CLIP weights
Không update GPT-2 weights
Không dùng ground-truth caption khi generate
Chỉ optimize context trong lúc inference
```

---

# 2. Ý tưởng trực quan nhất

Giả sử ảnh:

```text
🐕 Một con chó đang chạy trên bãi cỏ
```

GPT-2 đọc:

```text
"Image of a"
```

và tự đoán:

```text
man      0.30
woman    0.17
car      0.11
dog      0.07
```

GPT-2 chưa nhìn ảnh nên `"man"` có thể cao hơn `"dog"`.

ZeroCap lấy các candidate:

```text
"Image of a man"
"Image of a woman"
"Image of a car"
"Image of a dog"
```

đưa qua CLIP.

CLIP chấm:

```text
"Image of a man"     → 0.22
"Image of a woman"   → 0.18
"Image of a car"     → 0.19
"Image of a dog"     → 0.73
```

CLIP đang nói:

```text
Ảnh này hợp với "dog" hơn "man".
```

ZeroCap dùng gradient để điều chỉnh **context tạm thời của GPT-2** sao cho:

```text
P(dog) tăng
P(man) giảm
```

nhưng vẫn giữ GPT-2 nói câu tự nhiên.

---

# 3. ZeroCap có HAI vòng lặp

Đây là điều quan trọng nhất để hiểu vì sao ZeroCap chậm.

## Outer Loop — sinh từng token

```text
token 1
token 2
token 3
...
```

## Inner Loop — optimize context cho một token

Trước khi chọn mỗi token:

```text
optimization step 1
optimization step 2
optimization step 3
...
optimization step N
```

Ví dụ caption có 15 token và mỗi token optimize 5 lần:

```text
15 × 5 = 75 optimization rounds
```

Mỗi round còn có GPT-2 forward + CLIP text forward + backward.

Đó là lý do inference ZeroCap khá lâu.

---

# 4. ZeroCap nằm ở đâu trong project hiện tại?

Project có thể tách như sau:

```text
Flickr8k
   ↓
COMMON PREPROCESSING
   ↓
Fixed Train / Val / Test
   ↓
       ┌─────────────────────┐
       │                     │
       ▼                     ▼
   FEW-SHOT CLIPCAP       ZEROCAP
       │                     │
 train subsets             VAL / TEST image
 1/5/10/25/100%              │
       │                     │
 tokenized captions           │
       │                     │
 ClipCapDataset               │
       │                     │
 Mapper                       │
       │                     │
 Frozen GPT-2            Frozen CLIP + GPT-2
       │                     │
 CE Loss                 Context optimization
       │                     │
 train Mapper                │
       │                     │
 Caption                    Caption
```

## Preprocessing dùng chung

ZeroCap có thể reuse:

```text
1. Validate data
2. Parse captions
3. Normalize captions
4. Split theo image
5. Freeze Train / Val / Test
```

Đặc biệt:

```text
ClipCap và ZeroCap phải evaluate trên cùng fixed TEST image IDs
```

ZeroCap không cần:

```text
train_1pct.pt
train_5pct.pt
train_10pct.pt
train_25pct.pt
train_100pct.pt
ClipCapDataset
Mapper
prefix embedding
supervised labels
training DataLoader
```

CLIP feature cache chỉ reuse nếu **cùng CLIP model + preprocessing + feature convention**.

---

# 5. Folder structure đề xuất

```text
src/
├── preprocessing/
│   └── ...
├── mapping_network/              # ClipCap
│   └── ...
└── zerocap/
    ├── __init__.py
    ├── config.py
    ├── model_loader.py
    ├── clip_guidance.py
    ├── context_optimization.py
    ├── generate.py
    └── zerocap.py

tests/
└── test_zerocap.py

scripts/
├── run_zerocap.py
├── benchmark_zerocap.py
└── evaluate_zerocap.py

outputs/
├── zerocap_runtime.csv
├── zerocap_predictions.json
└── zerocap_metrics.json
```

---

# 6. End-to-End Pipeline tổng quát

```text
INPUT IMAGE
   ↓
1. CLIP image preprocessing
   ↓
2. Frozen CLIP Image Encoder
   ↓
image embedding E_img
   ↓
3. Khởi tạo prompt GPT-2
   ↓
4. GPT-2 tạo past_key_values C
   ↓
5. GPT-2 sinh next-token distribution gốc P_original
   ↓
6. Tạo trainable context perturbation ΔC
   ↓
7. Chạy GPT-2 với context C + ΔC
   ↓
8. Lấy Top-K candidate tokens
   ↓
9. Ghép candidate token vào caption hiện tại
   ↓
10. Frozen CLIP Text Encoder
   ↓
candidate text embeddings
   ↓
11. Tính cosine similarity với E_img
   ↓
12. Tạo CLIP-guidance target distribution
   ↓
13. Tính image-guidance loss
   ↓
14. Tính language-preservation loss
   ↓
15. Tính ZeroCap guidance objective
   ↓
16. Backward vào ΔC
   ↓
17. Update ΔC
   ↓
18. Lặp context optimization N bước
   ↓
19. Sinh guided next-token distribution
   ↓
20. Chọn next token
   ↓
21. Append token vào caption
   ↓
22. Nếu chưa EOS / max_length → quay lại
   ↓
23. Decode token IDs
   ↓
OUTPUT CAPTION
   ↓
24. Ghi inference time
```

---

# 7. Step 0 — Setup môi trường

Kiểm tra:

```python
import torch

print("PyTorch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())

if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
```

Mục tiêu:

```text
PyTorch chạy được
CUDA = True
```

Nếu CUDA không có thì ZeroCap vẫn chạy CPU nhưng rất chậm.

---

# 8. Step 1 — Load CLIP + GPT-2

Ta cần:

```text
GPT-2 model
GPT-2 tokenizer
CLIP model
CLIP tokenizer / processor
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

## Tại sao freeze?

Vì ZeroCap zero-shot:

```text
GPT-2 weights không học
CLIP weights không học
```

## Output

```text
Frozen GPT-2 ✅
Frozen CLIP ✅
```

---

# 9. Step 2 — Chọn 1 ảnh VAL để develop

Trong giai đoạn code:

```text
VAL
↓
1 image
```

Không dùng TEST để tune.

Rule:

```text
TRAIN
→ ClipCap training

VAL
→ debug ZeroCap
→ tune hyperparameter
→ benchmark 5 ảnh

TEST
→ final evaluation
```

Ground-truth caption của ảnh VAL **không được đưa vào generation**.

---

# 10. Step 3 — CLIP image preprocessing

Input:

```text
image.jpg
```

Pipeline:

```text
image.jpg
   ↓
CLIP processor
   ↓
resize
normalize
tensorize
   ↓
pixel_values
```

Ví dụ:

```text
pixel_values
shape ≈ [1, 3, 224, 224]
```

## Tại sao?

CLIP được pretrained với một preprocessing cụ thể.

---

# 11. Step 4 — Frozen CLIP Image Encoder

```text
pixel_values
   ↓
CLIP Image Encoder
   ↓
E_img
```

Nếu dùng CLIP ViT-B/32:

```text
E_img ≈ [1, 512]
```

Normalize:

```python
E_img = E_img / E_img.norm(
    dim=-1,
    keepdim=True
)
```

## E_img là gì?

Một vector semantic:

```text
[0.12, -0.45, 0.08, ..., 0.31]
```

Không phải label `"dog"`.

## Output

```text
E_img [1,512]
```

---

# 12. Dùng `clip_features.pt` được không?

Được nếu:

```text
cached CLIP model
==
ZeroCap CLIP model
```

Check:

```python
print(clip_feature_data["clip_model"])
print(project_config.CLIP_MODEL_NAME)
```

Nếu không chắc thì trong giai đoạn debug:

```text
raw image
→ encode trực tiếp
```

sẽ an toàn hơn.

---

# 13. Step 5 — Khởi tạo GPT-2 prompt

Ví dụ:

```text
"Image of a"
```

Tokenize:

```python
input_ids = gpt_tokenizer.encode(
    "Image of a",
    return_tensors="pt"
)
```

## Prompt để làm gì?

Nếu không có prompt, GPT-2 có thể bắt đầu:

```text
The...
I...
In...
```

Prompt giúp GPT-2 đi theo kiểu:

```text
"Image of a ____"
```

Prompt **không chứa thông tin ảnh**.

---

# 14. Step 6 — GPT-2 tạo `past_key_values`

Đây là phần quan trọng.

GPT-2 khi đọc prompt sẽ tạo attention cache:

```text
past_key_values
```

Gọi:

```text
C
```

Concept:

```text
"Image of a"
    ↓
GPT-2
    ↓
past_key_values C
```

Mỗi transformer layer có:

```text
Key cache
Value cache
```

Dạng logic:

```text
C =
(
  (K1, V1),
  (K2, V2),
  ...
  (Kn, Vn)
)
```

Shape từng tensor thường dạng:

```text
[B, num_heads, seq_len, head_dim]
```

## Rất quan trọng

Tensor:

```python
context_embeds = torch.zeros(1, 5, 768)
```

không phải `past_key_values`.

Nó gần với soft prompt / virtual prefix optimization.

ZeroCap đúng hướng cần:

```text
past_key_values + context_delta
```

---

# 15. Step 7 — GPT-2 sinh distribution gốc

Chạy GPT-2 trước khi CLIP can thiệp:

```text
prompt
 ↓
GPT-2
 ↓
logits
 ↓
softmax
 ↓
P_original
```

Ví dụ:

```text
man      0.30
woman    0.17
car      0.11
dog      0.07
```

## Tại sao cần giữ?

Đây là distribution tự nhiên của GPT-2.

Sau này ta sẽ dùng nó để giữ ngữ pháp.

## Output

```text
P_original [1, vocab_size]
```

---

# 16. Step 8 — Tạo context perturbation ΔC

Ta có context gốc:

```text
C
```

Tạo:

```text
ΔC
```

có structure giống `past_key_values`.

Ban đầu:

```text
ΔC = 0
```

và:

```text
requires_grad = True
```

Context mới:

```text
C_guided = C + ΔC
```

## Thứ được optimize

```text
GPT-2 weights ❌
CLIP weights  ❌
ΔC            ✅
```

Đây là bản chất ZeroCap.

---

# 17. Step 9 — Chạy GPT-2 với context đã perturb

```text
C + ΔC
  ↓
Frozen GPT-2
  ↓
logits
  ↓
softmax
  ↓
P_guided
```

Ban đầu:

```text
ΔC = 0
```

nên:

```text
P_guided ≈ P_original
```

Sau nhiều gradient steps, P_guided sẽ bị image guidance điều khiển.

---

# 18. Step 10 — Lấy Top-K candidate tokens

Không thể đưa toàn vocabulary vào CLIP mỗi round.

Ta lấy:

```python
top_probs, top_indices = torch.topk(
    probs,
    k=K,
    dim=-1
)
```

Debug:

```text
K = 5 hoặc 10
```

Output:

```text
candidate token IDs
candidate probabilities
```

Ví dụ:

```text
man
woman
car
dog
building
```

---

# 19. Step 11 — Ghép candidate vào current caption

Sai kiểu:

```text
"dog"
"man"
"car"
```

Nên làm:

```text
Current:
"Image of a"

Candidates:
"Image of a man"
"Image of a woman"
"Image of a car"
"Image of a dog"
```

## Tại sao?

CLIP cần đánh giá **câu hoàn chỉnh hiện tại**, không chỉ từ riêng lẻ.

Pseudo-code:

```python
prefix_text = tokenizer.decode(
    current_ids[0],
    skip_special_tokens=True
)

candidate_texts = [
    prefix_text + tokenizer.decode([token_id])
    for token_id in candidate_ids
]
```

## Output

```text
K candidate captions
```

---

# 20. Step 12 — CLIP Text Encoder

```text
candidate captions
   ↓
CLIP tokenizer
   ↓
Frozen CLIP Text Encoder
   ↓
candidate text embeddings
```

Nếu CLIP dim 512:

```text
[K,512]
```

Normalize:

```python
text_features = (
    text_features
    / text_features.norm(dim=-1, keepdim=True)
)
```

---

# 21. Step 13 — Cosine similarity

Ta có:

```text
E_img
```

và:

```text
E_text
```

Tính:

```python
sims = (
    text_features @ E_img.T
).squeeze(-1)
```

Ví dụ:

```text
"Image of a man"   0.22
"Image of a dog"   0.73
"Image of a car"   0.19
```

Đây chính là nơi ảnh ảnh hưởng vào generation.

---

# 22. Step 14 — Tạo CLIP guidance distribution

Từ similarity:

```text
scores
 ↓
temperature
 ↓
softmax
 ↓
P_CLIP
```

Ví dụ:

```text
dog    0.88
man    0.05
car    0.04
```

CLIP đang nói:

```text
"Token dog phù hợp ảnh nhất."
```

---

# 23. Step 15 — Image Guidance Loss

Có:

```text
P_guided từ GPT-2
P_CLIP từ CLIP
```

Nếu:

```text
GPT-2 thích man
CLIP thích dog
```

loss phải lớn.

Gradient:

```text
L_CLIP
   ↑
GPT-2 computation
   ↑
C + ΔC
     ↑
     ΔC
```

Mục tiêu:

```text
P_guided
→ gần distribution mà CLIP muốn
```

---

# 24. Step 16 — Language Preservation Loss

Nếu chỉ nghe CLIP, câu có thể thành:

```text
dog grass green animal running outside...
```

Do đó ta giữ P_guided không lệch quá xa P_original.

Concept:

```text
P_guided
vs
P_original
```

Có thể dùng KL-divergence hoặc objective tương đương.

Ý nghĩa:

```text
CLIP guidance
→ nói đúng ảnh

Language preservation
→ nói tự nhiên
```

---

# 25. Step 17 — Tổng ZeroCap Objective

Concept:

```text
L_total
=
L_image_guidance
+
λ × L_language
```

Hai lực:

```text
CLIP
→ "nói đúng ảnh"

GPT-2 prior
→ "nói đúng ngôn ngữ"
```

Không có:

```text
ground-truth Flickr8k caption
```

trong loss này.

---

# 26. Step 18 — Backward

```python
total_loss.backward()
```

Gradient mong muốn:

```text
GPT-2 parameter gradients  ❌
CLIP parameter gradients   ❌
context_delta gradients    ✅
```

Luồng:

```text
LOSS
 ↑
GPT-2 computation
 ↑
C + ΔC
     ↑
     ΔC
```

---

# 27. Step 19 — Update ΔC

Sau backward:

```text
ΔC_old
 ↓
gradient update
 ↓
ΔC_new
```

Sau đó:

```text
C_guided = C + ΔC_new
```

rồi chạy GPT-2 lại.

---

# 28. Step 20 — Inner Loop N lần

Ví dụ:

```text
Optimization 1
P(dog): 0.07 → 0.14

Optimization 2
P(dog): 0.14 → 0.25

Optimization 3
P(dog): 0.25 → 0.39

Optimization 4
P(dog): 0.39 → 0.51

Optimization 5
P(dog): 0.51 → 0.62
```

Số trên chỉ minh họa.

N càng lớn:

```text
guidance có thể mạnh hơn
runtime tăng
```

Nên tune trên VAL.

---

# 29. Step 21 — Guided next-token distribution

Sau optimization:

```text
P_original:
man 0.35
dog 0.08

P_guided:
man 0.08
dog 0.61
```

Ảnh đã bẻ hướng GPT-2.

---

# 30. Step 22 — Chọn next token

Có thể:

```text
greedy
top-k
beam search
sampling
```

Debug đầu tiên nên dùng cách đơn giản.

Ví dụ:

```text
next_token = "dog"
```

---

# 31. Step 23 — Append token

```text
Before:
"Image of a"

After:
"Image of a dog"
```

Token đầu tiên hoàn thành.

---

# 32. Step 24 — Outer Loop tiếp tục

Current caption:

```text
"Image of a dog"
```

Lại làm:

```text
GPT-2 distribution
↓
past_key_values
↓
ΔC
↓
Top-K
↓
candidate captions
↓
CLIP score
↓
guidance loss
↓
language loss
↓
optimize context
↓
select next token
```

Ví dụ:

```text
"Image of a dog"
→ running

"Image of a dog running"
→ through

"Image of a dog running through"
→ a

...
```

---

# 33. Step 25 — Điều kiện dừng

Dừng khi:

```text
EOS
```

hoặc:

```text
max_length
```

Có thể thêm repetition penalty nếu model lặp.

---

# 34. Step 26 — Decode token IDs

```python
caption = gpt_tokenizer.decode(
    generated_ids,
    skip_special_tokens=True
)
```

Ví dụ:

```text
"Image of a dog running through a grassy field."
```

Có thể strip prompt để trả:

```text
"A dog running through a grassy field."
```

## Output chính

```text
caption: str
```

---

# 35. Step 27 — Đo inference time

Bao toàn bộ:

```text
image
→ ZeroCap
→ caption
```

CUDA timing:

```python
import time
import torch

torch.cuda.synchronize()
start = time.perf_counter()

caption = generate_caption(image)

torch.cuda.synchronize()
elapsed = time.perf_counter() - start
```

Output:

```python
{
    "caption": "A dog running through a grassy field.",
    "inference_time_sec": 14.82
}
```

---

# 36. Tại sao ZeroCap chậm?

Ví dụ:

```text
15 generated tokens
×
5 context optimization steps/token
=
75 optimization rounds
```

Mỗi round:

```text
GPT-2 forward
+
Top-K
+
K candidate captions
+
CLIP Text Encoder
+
loss
+
backward
```

Nên runtime cao hơn captioning thông thường rất nhiều.

---

# 37. 1 ảnh PASS rồi mới benchmark 5 ảnh

Không benchmark 5 ảnh khi 1 ảnh còn chưa end-to-end.

Definition of Done cho 1 ảnh:

```text
image load                 ✅
CLIP embedding finite      ✅
past_key_values có         ✅
context_delta có gradient  ✅
CLIP/GPT-2 weights frozen  ✅
candidate texts đúng       ✅
CLIP score finite          ✅
generate được token        ✅
generate full caption      ✅
caption không rỗng         ✅
```

---

# 38. Benchmark 5 ảnh VAL

Không dùng TEST để tune.

```text
VAL
↓
warm-up 1 ảnh
↓
không tính warm-up
↓
5 ảnh benchmark
```

Save:

```text
image_id,caption,time_sec
img1.jpg,"...",12.4
img2.jpg,"...",13.8
img3.jpg,"...",14.1
img4.jpg,"...",12.9
img5.jpg,"...",13.5
```

Tính:

```text
average
min
max
```

---

# 39. Ước lượng kích thước test cuối

```text
avg_time =
(t1+t2+t3+t4+t5)/5
```

Ví dụ:

```text
avg = 14 sec/image
```

thì:

```text
100 images
≈ 1400 sec
≈ 23.3 min
```

Từ đó mới quyết định:

```text
50
100
500
full test
```

tùy GPU và deadline.

---

# 40. Final Evaluation

Sau khi tune xong trên VAL:

```text
fixed TEST
↓
ZeroCap inference
↓
generated captions
```

Lúc này mới dùng ground-truth:

```text
generated caption
+
Flickr8k references
↓
BLEU
CIDEr
```

Và:

```text
image + generated caption
↓
CLIPScore
```

Nên report:

```text
BLEU-1
BLEU-4
CIDEr
CLIPScore
Average inference time
```

---

# 41. Data leakage / test contamination

Không nên:

```text
run nhiều config trên TEST
↓
xem metric
↓
chọn config tốt nhất
```

Đúng:

```text
VAL
→ tune K
→ tune N
→ tune λ
→ tune max_length
→ tune decoding
→ benchmark runtime

TEST
→ final run
```

---

# 42. Hai người chia việc

## Người 1 — Core ZeroCap

```text
model_loader
↓
image encoding
↓
past_key_values
↓
context_delta
↓
candidate generation
↓
CLIP guidance
↓
loss
↓
context optimization
↓
generation loop
```

Output:

```python
caption = generate_caption(image)
```

## Người 2 — Integration / Benchmark

```text
fixed VAL images
↓
technical tests
↓
benchmark script
↓
timing
↓
CSV / JSON
↓
evaluation
↓
wrapper
```

Hai người gặp nhau ở:

```python
generate_caption(image)
```

---

# 43. Trạng thái code hiện tại

Hiện prototype đã có:

```text
CUDA                         ✅
Load CLIP/GPT-2              ✅
Freeze models                ✅
Load E_img                   ✅
Prompt                       ✅
Original GPT-2 distribution  ✅
Top-K                        ✅
CLIP text encoding           ✅
Cosine similarity            ✅
Language regularization      ✅ prototype
Backward                     ✅ prototype
```

Cần sửa:

```text
1.
context_embeds [1,5,768]
→
past_key_values + context_delta

2.
candidate = "dog"
→
candidate = current caption + "dog"

3.
weighted cosine loss
→
CLIP target distribution + guidance loss

4.
guided token
→ append
→ autoregressive loop
→ full caption
```

---

# 44. Milestone triển khai thực tế

```text
M0
CUDA + CLIP + GPT-2 load/freeze
✅

M1
1 VAL image → E_img
✅

M2
Prompt → past_key_values
↓
create context_delta
↓
context_delta gradient
← NEXT

M3
Top-K
↓
full candidate captions
↓
CLIP scores

M4
CLIP guidance loss
+
language-preservation loss

M5
Optimize context

M6
Generate 1 token

M7
Autoregressive loop
↓
full caption

M8
5 technical tests PASS

M9
5 VAL images runtime benchmark

M10
Choose final test size

M11
Run fixed TEST

M12
BLEU / CIDEr / CLIPScore / runtime

M13
generate_caption(image)
ready for API
```

---

# 45. Interface cuối cùng

Cuối cùng backend không cần biết nội bộ ZeroCap.

Chỉ cần:

```python
zerocap = ZeroCapCaptioner(config)

result = zerocap.generate_caption(image)
```

Output:

```python
{
    "caption": "A dog running through a grassy field.",
    "inference_time_sec": 14.2
}
```

Sau đó:

```text
ZeroCapCaptioner
↓
FastAPI
↓
UI Demo
↓
Docker
↓
Deploy
```

---

# 46. Một câu để nhớ toàn bộ ZeroCap

```text
GPT-2 đề xuất token
        ↓
CLIP nhìn ảnh và chấm candidate
        ↓
ZeroCap tính loss
        ↓
gradient chỉnh context GPT-2
        ↓
KHÔNG chỉnh GPT-2 weights
        ↓
chọn token tốt hơn
        ↓
append
        ↓
lặp
        ↓
caption
```

---

# 47. Checklist hoàn thành ZeroCap end-to-end

```text
[ ] CUDA chạy
[ ] CLIP load + frozen
[ ] GPT-2 load + frozen
[ ] Fixed VAL / TEST rõ ràng
[ ] Image → E_img finite
[ ] Prompt tokenize được
[ ] GPT-2 past_key_values lấy được
[ ] context_delta có đúng structure
[ ] context_delta có gradient
[ ] GPT-2 / CLIP weights không đổi
[ ] Top-K candidate chạy
[ ] Candidate chứa full current caption
[ ] CLIP text embeddings finite
[ ] Cosine similarity finite
[ ] CLIP guidance distribution hợp lệ
[ ] Language preservation loss finite
[ ] Total loss finite
[ ] Context optimization N steps chạy
[ ] Generate được 1 token
[ ] Autoregressive loop chạy
[ ] Full caption không rỗng
[ ] Runtime đo đúng CUDA
[ ] 5 technical tests PASS
[ ] Benchmark 5 VAL images
[ ] Average runtime tính được
[ ] Final test size được chọn
[ ] Fixed TEST chạy hoàn chỉnh
[ ] BLEU / CIDEr / CLIPScore tính được
[ ] `generate_caption(image)` sẵn sàng cho FastAPI
```

---

# 48. Điểm cần làm tiếp ngay bây giờ

Không cần viết lại setup.

Bước tiếp theo chính xác là:

```text
Prompt
↓
GPT-2
↓
past_key_values C
↓
tạo context_delta ΔC
↓
C + ΔC
↓
GPT-2 forward
↓
loss.backward()
↓
xác nhận ΔC có gradient
```

PASS bước này rồi mới tiếp:

```text
Top-K
↓
full candidate captions
↓
CLIP chấm
↓
guidance loss
↓
generate token
```

Đây là đường đi ngắn nhất để đưa prototype hiện tại thành ZeroCap end-to-end hoàn chỉnh.

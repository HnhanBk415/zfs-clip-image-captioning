# ClipCap: Extended Attention Mask và Extended Labels

Tài liệu này mô tả phần ghép **visual prefix** từ Mapping Network với caption token trước khi đưa vào GPT-2. Phạm vi công việc được chia cho hai thành viên:

Mục tiêu là làm rõ ý nghĩa của từng tensor, hợp đồng đầu vào/đầu ra, cách GPT-2 tính causal language-model loss, mã tham khảo, kiểm thử và các lỗi thường gặp.

---

## 1. Vị trí của phần này trong ClipCap

```text
CLIP feature [B, clip_dim]
        |
        v
TransformerMapper
        |
        v
Visual prefix [B, P, D] -------------------------+
                                                     |
input_ids [B, L] -> GPT-2 token embedding [B, L, D] |
                                                     v
                                   Concatenate embeddings
                                       [B, P + L, D]
                                                     |
attention_mask [B, L] -> extended_attention_mask ---+--> GPT-2
input_ids + attention_mask -> extended_labels ------+     |
                                                           v
                                                     loss + logits
```

Ký hiệu:

- `B`: batch size.
- `P`: `prefix_length`, số visual prefix token.
- `L`: độ dài chuỗi caption sau tokenize và padding.
- `D`: embedding dimension của GPT-2.
- `V`: vocabulary size của GPT-2.

Mapping Network hiện tại trả về:

```text
prefix_embeddings: [B, P, D]
```

GPT-2 token embedding trả về:

```text
text_embeddings: [B, L, D]
```

Hai tensor được ghép theo chiều sequence:

```python
inputs_embeds = torch.cat(
    [prefix_embeddings, text_embeddings],
    dim=1,
)
```

Kết quả:

```text
inputs_embeds: [B, P + L, D]
```

Vì sequence đã dài thêm `P` vị trí, `attention_mask` và `labels` cũng phải được mở rộng thành chiều dài `P + L`.

---

## 2. Contract dữ liệu hiện tại của dự án

`ClipCapDataset` đang trả về một batch có cấu trúc:

```python
batch = {
    "image_embed": image_embed,        # [B, clip_dim]
    "input_ids": input_ids,            # [B, L]
    "attention_mask": attention_mask,  # [B, L]
}
```

Dataset không tạo hoặc trả về `labels`. Khi training hoặc validation cần tính loss, tầng model tạo caption labels từ `input_ids` và `attention_mask`:

```python
caption_labels = input_ids.masked_fill(
    attention_mask == 0,
    -100,
)
```

Do đó:

- Dataset chỉ chịu trách nhiệm cung cấp dữ liệu ảnh và caption tokenized.
- Caption token hợp lệ giữ nguyên token ID khi model tạo labels.
- Padding trong caption labels được đổi thành `-100`.
- `input_ids` không bị thay đổi tại chỗ.

Khi generation, model không cần tạo labels. Interface production nên nhận `labels=None`; training và validation truyền `labels=batch["input_ids"]`.

---

## 3. Attention mask là gì?

`attention_mask` trả lời câu hỏi:

> Vị trí nào trong sequence chứa dữ liệu hợp lệ để model sử dụng?

Quy ước thông thường:

```text
1: vị trí hợp lệ
0: vị trí padding
```

Ví dụ caption có 3 token thật và 2 token padding:

```text
input_ids:      [21, 45, 86, 0, 0]
attention_mask: [ 1,  1,  1, 0, 0]
```

Visual prefix là ngữ cảnh hợp lệ do Mapping Network tạo ra. GPT-2 phải được phép sử dụng toàn bộ visual prefix, vì vậy mask cho `P` vị trí prefix phải bằng `1`.

Với `P = 3`:

```text
prefix mask:             [1, 1, 1]
caption attention mask:  [1, 1, 1, 0, 0]
extended attention mask: [1, 1, 1, 1, 1, 1, 0, 0]
```

Shape:

```text
[B, L] -> [B, P + L]
```

### Hàm tham khảo

```python
import torch
from torch import Tensor


def extend_attention_mask(
    attention_mask: Tensor,
    prefix_length: int,
) -> Tensor:
    if attention_mask.ndim != 2:
        raise ValueError(
            "attention_mask must have shape [B, L]"
        )

    if isinstance(prefix_length, bool) or not isinstance(prefix_length, int):
        raise ValueError("prefix_length must be a positive integer")

    if prefix_length <= 0:
        raise ValueError("prefix_length must be a positive integer")

    batch_size = attention_mask.size(0)

    prefix_mask = torch.ones(
        (batch_size, prefix_length),
        dtype=attention_mask.dtype,
        device=attention_mask.device,
    )

    return torch.cat(
        [prefix_mask, attention_mask],
        dim=1,
    )
```

Điểm cần đảm bảo:

- Không hard-code `prefix_length = 10` trong hàm.
- Giữ nguyên `dtype` và `device` của `attention_mask`.
- Không thay đổi tensor đầu vào.
- Đầu ra có shape `[B, P + L]`.

---

## 4. Labels là gì?

`labels` trả lời câu hỏi:

> Model bị chấm loss tại vị trí nào, và token đúng tại vị trí đó là token nào?

Đối với Hugging Face GPT-2:

- Token ID hợp lệ: vị trí đó tham gia tính loss.
- `-100`: vị trí đó bị bỏ qua khi tính loss.

PyTorch cross-entropy dùng `ignore_index=-100` cho bài toán causal language modeling của GPT-2.

### Vì sao visual prefix phải có label `-100`?

Visual prefix là embedding liên tục được Mapping Network sinh ra. Nó không phải token ID trong vocabulary của GPT-2, nên không tồn tại một token đúng để GPT-2 dự đoán tại vị trí prefix.

Ta vẫn muốn GPT-2 đọc visual prefix làm ngữ cảnh, nhưng không muốn tính loss trực tiếp trên các vị trí đó. Vì vậy:

```text
Visual prefix:
attention_mask = 1
label          = -100
```

Đây là hai quyết định khác nhau:

- `attention_mask = 1`: GPT-2 được dùng thông tin tại vị trí prefix.
- `label = -100`: GPT-2 không bị chấm loss tại vị trí prefix.

### Ví dụ extended labels

Giả sử:

```text
P = 3
input_ids:      [21, 45, 86, 0, 0]
attention_mask: [ 1,  1,  1, 0, 0]
```

Caption labels sau khi bỏ qua padding:

```text
[21, 45, 86, -100, -100]
```

Thêm `P` prefix labels ở đầu:

```text
extended_labels:
[-100, -100, -100, 21, 45, 86, -100, -100]
```

Shape:

```text
[B, L] -> [B, P + L]
```

### Hàm tham khảo

```python
import torch
from torch import Tensor


def build_extended_labels(
    input_ids: Tensor,
    attention_mask: Tensor,
    prefix_length: int,
) -> Tensor:
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [B, L]")

    if attention_mask.ndim != 2:
        raise ValueError(
            "attention_mask must have shape [B, L]"
        )

    if input_ids.shape != attention_mask.shape:
        raise ValueError(
            "input_ids and attention_mask must have the same shape"
        )

    if isinstance(prefix_length, bool) or not isinstance(prefix_length, int):
        raise ValueError("prefix_length must be a positive integer")

    if prefix_length <= 0:
        raise ValueError("prefix_length must be a positive integer")

    if input_ids.dtype != torch.long:
        raise TypeError("input_ids must use torch.long dtype")

    caption_labels = input_ids.clone()
    caption_labels.masked_fill_(attention_mask == 0, -100)

    batch_size = input_ids.size(0)

    prefix_labels = torch.full(
        (batch_size, prefix_length),
        fill_value=-100,
        dtype=input_ids.dtype,
        device=input_ids.device,
    )

    return torch.cat(
        [prefix_labels, caption_labels],
        dim=1,
    )
```

Điểm cần đảm bảo:

- Prefix labels luôn bằng `-100`.
- Padding labels luôn bằng `-100`.
- Token ID hợp lệ của caption được giữ nguyên.
- Labels có kiểu `torch.long`, không phải floating point.
- Giữ nguyên device của `input_ids`.
- Clone trước khi sửa để không làm thay đổi `input_ids` do Dataset cung cấp.
- Không hard-code số lượng prefix token.

---

## 5. So sánh attention mask và labels

| Loại vị trí | Attention mask | Label | GPT-2 được đọc? | Có tính loss? |
|---|---:|---:|---:|---:|
| Visual prefix | `1` | `-100` | Có | Không |
| Caption token hợp lệ | `1` | Token ID | Có | Có |
| Padding | `0` | `-100` | Không | Không |

Ghi nhớ ngắn gọn:

```text
attention_mask điều khiển việc sử dụng vị trí làm ngữ cảnh.
labels điều khiển việc chấm loss tại vị trí đó.
```

Không nên dùng hai tensor này thay thế cho nhau.

---

## 6. Causal language modeling và label shifting

GPT-2 dự đoán token kế tiếp dựa trên các vị trí đứng trước nó. Khi truyền `labels` vào GPT-2 của Hugging Face, model tự thực hiện label shifting tương đương:

```python
shift_logits = logits[:, :-1, :]
shift_labels = labels[:, 1:]
```

Ví dụ:

```text
inputs_embeds:
[prefix_1, prefix_2, prefix_3, token_1, token_2, token_3]

labels:
[-100,    -100,    -100,    token_1, token_2, token_3]
```

Sau khi GPT-2 shift nội bộ:

- Output tại `prefix_3` được dùng để dự đoán `token_1`.
- Output tại `token_1` được dùng để dự đoán `token_2`.
- Output tại `token_2` được dùng để dự đoán `token_3`.

Do đó:

> Không tự shift labels trước khi truyền vào GPT-2.

Nếu tự shift bên ngoài rồi GPT-2 tiếp tục shift nội bộ, mục tiêu huấn luyện sẽ bị lệch một vị trí.

Nếu tokenizer có BOS token, token đầu tiên được dự đoán có thể là BOS. Nếu Dataset không thêm BOS, visual prefix cuối sẽ trực tiếp dự đoán token đầu tiên của caption. Hai cách đều có thể dùng, nhưng toàn nhóm phải thống nhất cách tokenize.

---

## 7. Vì sao Mapper vẫn học khi prefix labels bằng `-100`?

`-100` chỉ bỏ qua loss trực tiếp tại các vị trí prefix. Visual prefix vẫn ảnh hưởng đến hidden state và các dự đoán caption đứng sau nó.

Luồng gradient:

```text
caption loss
    |
    v
GPT-2 hidden states
    |
    v
visual prefix embeddings
    |
    v
TransformerMapper parameters
```

Khi caption prediction sai, gradient truyền ngược qua GPT-2 về prefix embedding, rồi tiếp tục truyền về Mapping Network. Nhờ vậy Mapper học cách biến CLIP feature thành ngữ cảnh hữu ích cho GPT-2.

Nếu GPT-2 bị freeze:

- Tham số GPT-2 không được cập nhật.
- Gradient vẫn có thể truyền xuyên qua GPT-2 về Mapper.
- Không được bọc forward của GPT-2 bằng `torch.no_grad()` trong quá trình train Mapper, vì làm vậy sẽ cắt gradient về Mapper.

---

## 8. Ghép hai phần vào GPT-2

```python
prefix_embeddings = mapper(image_embed)
# [B, P, D]

text_embeddings = gpt2.get_input_embeddings()(input_ids)
# [B, L, D]

inputs_embeds = torch.cat(
    [prefix_embeddings, text_embeddings],
    dim=1,
)
# [B, P + L, D]

extended_attention_mask = extend_attention_mask(
    attention_mask=attention_mask,
    prefix_length=prefix_embeddings.size(1),
)
# [B, P + L]

extended_labels = build_extended_labels(
    input_ids=input_ids,
    attention_mask=attention_mask,
    prefix_length=prefix_embeddings.size(1),
)
# [B, P + L]

sequence_length = inputs_embeds.size(1)

if extended_attention_mask.size(1) != sequence_length:
    raise ValueError("inputs_embeds and attention mask length mismatch")

if extended_labels.size(1) != sequence_length:
    raise ValueError("inputs_embeds and labels length mismatch")

outputs = gpt2(
    inputs_embeds=inputs_embeds,
    attention_mask=extended_attention_mask,
    labels=extended_labels,
)

loss = outputs.loss
logits = outputs.logits
```

Shape kỳ vọng:

```text
inputs_embeds:             [B, P + L, D]
extended_attention_mask:   [B, P + L]
extended_labels:           [B, P + L]
logits:                    [B, P + L, V]
loss:                      scalar
```

Nên lấy `prefix_length` từ tensor thực tế:

```python
prefix_length = prefix_embeddings.size(1)
```

Cách này giảm nguy cơ cấu hình và output của Mapper không đồng bộ.

---

## 9. Phân công và hợp đồng tích hợp

### Thành viên làm extended attention mask

Sở hữu:

```python
extend_attention_mask(attention_mask, prefix_length)
```

Cam kết đầu ra:

- Shape `[B, P + L]`.
- `P` vị trí đầu bằng `1`.
- Phần caption giữ nguyên attention mask ban đầu.
- Giữ nguyên dtype và device.
- Không thay đổi tensor đầu vào.

### Thành viên làm extended labels

Sở hữu:

```python
build_extended_labels(input_ids, attention_mask, prefix_length)
```

Cam kết đầu ra:

- Shape `[B, P + L]`.
- `P` vị trí đầu bằng `-100`.
- Padding caption bằng `-100`.
- Token caption hợp lệ giữ nguyên token ID.
- Dtype là `torch.long`.
- Giữ nguyên device.
- Không thay đổi tensor đầu vào.

### Phần hai người cùng kiểm tra

- Chiều sequence của embeddings, mask và labels bằng nhau.
- GPT-2 trả về loss hữu hạn.
- `loss.backward()` truyền gradient về Mapping Network.
- Chạy được với nhiều batch size và caption length.
- Chạy được trên CPU và GPU nếu môi trường có CUDA.

---

## 10. Unit test cho extended attention mask

```python
import torch


def test_extended_attention_mask_values():
    attention_mask = torch.tensor([
        [1, 1, 1, 0, 0],
        [1, 1, 1, 1, 0],
    ])

    result = extend_attention_mask(
        attention_mask=attention_mask,
        prefix_length=3,
    )

    expected = torch.tensor([
        [1, 1, 1, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 0],
    ])

    assert result.shape == (2, 8)
    assert torch.equal(result, expected)
```

```python
def test_attention_mask_input_is_not_modified():
    attention_mask = torch.tensor([[1, 1, 0]])
    original = attention_mask.clone()

    extend_attention_mask(attention_mask, prefix_length=2)

    assert torch.equal(attention_mask, original)
```

---

## 11. Unit test cho extended labels

```python
import torch


def test_extended_labels_values():
    input_ids = torch.tensor([
        [10, 20, 30, 0, 0],
        [11, 21, 31, 41, 0],
    ], dtype=torch.long)

    attention_mask = torch.tensor([
        [1, 1, 1, 0, 0],
        [1, 1, 1, 1, 0],
    ])

    result = build_extended_labels(
        input_ids=input_ids,
        attention_mask=attention_mask,
        prefix_length=3,
    )

    expected = torch.tensor([
        [-100, -100, -100, 10, 20, 30, -100, -100],
        [-100, -100, -100, 11, 21, 31, 41, -100],
    ], dtype=torch.long)

    assert result.shape == (2, 8)
    assert torch.equal(result, expected)
```

```python
def test_input_ids_are_not_modified():
    input_ids = torch.tensor([[10, 20, 0]], dtype=torch.long)
    attention_mask = torch.tensor([[1, 1, 0]])
    original = input_ids.clone()

    build_extended_labels(
        input_ids=input_ids,
        attention_mask=attention_mask,
        prefix_length=2,
    )

    assert torch.equal(input_ids, original)
```

```python
def test_extended_labels_ignore_prefix_and_padding():
    input_ids = torch.tensor([[10, 20, 0]], dtype=torch.long)
    attention_mask = torch.tensor([[1, 1, 0]])

    result = build_extended_labels(
        input_ids=input_ids,
        attention_mask=attention_mask,
        prefix_length=2,
    )

    assert torch.all(result[:, :2] == -100)
    assert result[0, -1].item() == -100
```

---

## 12. Integration test tối thiểu

```python
prefix_embeddings = mapper(image_embed)
text_embeddings = gpt2.get_input_embeddings()(input_ids)

inputs_embeds = torch.cat(
    [prefix_embeddings, text_embeddings],
    dim=1,
)

prefix_length = prefix_embeddings.size(1)

extended_attention_mask = extend_attention_mask(
    attention_mask,
    prefix_length,
)

extended_labels = build_extended_labels(
    input_ids,
    attention_mask,
    prefix_length,
)

assert inputs_embeds.size(0) == extended_attention_mask.size(0)
assert inputs_embeds.size(0) == extended_labels.size(0)
assert inputs_embeds.size(1) == extended_attention_mask.size(1)
assert inputs_embeds.size(1) == extended_labels.size(1)

outputs = gpt2(
    inputs_embeds=inputs_embeds,
    attention_mask=extended_attention_mask,
    labels=extended_labels,
)

assert outputs.loss is not None
assert torch.isfinite(outputs.loss)

outputs.loss.backward()

mapper_has_gradient = any(
    parameter.grad is not None
    for parameter in mapper.parameters()
    if parameter.requires_grad
)

assert mapper_has_gradient
```

---

## 13. Các lỗi thường gặp

### 13.1 Gán label `0` cho prefix

Sai:

```python
prefix_labels = torch.zeros(...)
```

Nếu token ID `0` hợp lệ trong vocabulary, GPT-2 sẽ bị yêu cầu dự đoán token đó tại prefix. Điều này tạo mục tiêu huấn luyện sai.

Đúng:

```python
prefix_labels = torch.full(..., fill_value=-100)
```

### 13.2 Gán attention mask bằng `0` cho prefix

Sai:

```text
prefix attention mask = 0
```

Điều này khiến GPT-2 không sử dụng visual prefix đúng cách.

Đúng:

```text
prefix attention mask = 1
```

### 13.3 Chỉ mở rộng embeddings nhưng không mở rộng mask và labels

Khi `inputs_embeds` có chiều dài `P + L` nhưng mask hoặc labels chỉ có `L`, GPT-2 sẽ gặp lỗi shape hoặc học với dữ liệu không căn chỉnh.

### 13.4 Quên bỏ qua padding trong labels

`attention_mask = 0` không tự động bảo đảm padding bị bỏ qua trong language-model loss. Padding labels vẫn nên được đặt thành `-100`.

### 13.5 Tự shift labels

GPT-2 của Hugging Face tự shift labels khi nhận tham số `labels`. Không shift thủ công thêm lần nữa.

### 13.6 Sửa labels tại chỗ

Sai:

```python
input_ids[attention_mask == 0] = -100
```

nếu `input_ids` còn được dùng để lấy GPT-2 token embeddings.

An toàn hơn:

```python
caption_labels = input_ids.clone()
caption_labels.masked_fill_(attention_mask == 0, -100)
```

### 13.7 Tạo tensor trên sai device

Sai:

```python
prefix_labels = torch.full((batch_size, prefix_length), -100)
```

Tensor mới mặc định nằm trên CPU. Nếu model đang ở GPU, phép concatenate sẽ lỗi.

Đúng:

```python
prefix_labels = torch.full(
    (batch_size, prefix_length),
    -100,
    dtype=input_ids.dtype,
    device=input_ids.device,
)
```

### 13.8 Hard-code `prefix_length`

Không nên viết trực tiếp `10` trong logic ghép tensor. Nên dùng cấu hình hoặc số token thực tế:

```python
prefix_length = prefix_embeddings.size(1)
```

### 13.9 Dùng `torch.no_grad()` quanh GPT-2 khi chỉ train Mapper

Freeze tham số GPT-2 bằng `requires_grad=False` vẫn cho phép gradient truyền về Mapper. Bao toàn bộ GPT-2 forward bằng `torch.no_grad()` sẽ cắt luồng gradient và Mapper không học được.

---

## 14. Training và generation khác nhau thế nào?

### Khi training

Cần cả ba tensor:

```text
inputs_embeds
extended_attention_mask
extended_labels
```

GPT-2 dùng labels để tính loss.

### Khi generation

Không truyền labels vì không tính teacher-forcing loss:

```text
inputs_embeds
attention_mask
```

Trong quá trình sinh từng token, attention mask phải được cập nhật theo sequence mới. Phần generation nên được xử lý riêng, không tái sử dụng nguyên xi logic training nếu API generate yêu cầu định dạng khác.

---

## 15. Definition of Done

Toàn bộ điều kiện sau:

- `inputs_embeds` có shape `[B, P + L, D]`.
- `extended_attention_mask` có shape `[B, P + L]`.
- `extended_labels` có shape `[B, P + L]`.
- Prefix attention mask đều bằng `1`.
- Prefix labels đều bằng `-100`.
- Padding attention mask bằng `0`.
- Padding labels bằng `-100`.
- Caption labels hợp lệ giữ nguyên token ID.
- Không hard-code `P`, `L`, `D` hoặc batch size.
- Tensor mới giữ đúng dtype và device.
- Hàm không thay đổi tensor đầu vào.
- Không shift labels thủ công.
- GPT-2 trả về loss hữu hạn.
- `loss.backward()` tạo gradient cho Mapping Network.
- Unit test chạy được với batch size 1 và batch size lớn hơn 1.
- Có ít nhất một integration test ghép Mapper, GPT-2, mask và labels.

---

## 16. Checklist trước khi merge

### Attention mask

- [ ] Đã kiểm tra input rank `[B, L]`.
- [ ] Đã kiểm tra `prefix_length > 0`.
- [ ] Prefix mask có giá trị `1`.
- [ ] Output có shape `[B, P + L]`.
- [ ] Dtype và device được giữ nguyên.
- [ ] Có unit test giá trị, shape và không sửa input.

### Exteneded labels

- [ ] Đã kiểm tra input IDs và attention mask cùng shape.
- [ ] Đã kiểm tra input IDs dùng `torch.long`.
- [ ] Prefix labels có giá trị `-100`.
- [ ] Padding labels có giá trị `-100`.
- [ ] Output có shape `[B, P + L]`.
- [ ] Không shift labels thủ công.
- [ ] Không sửa input IDs đầu vào.
- [ ] Có unit test giá trị, shape, dtype và padding.

### Cả hai

- [ ] Thống nhất tên tham số và public interface.
- [ ] Thống nhất cách lấy `prefix_length`.
- [ ] Chạy integration test với GPT-2.
- [ ] Kiểm tra loss hữu hạn.
- [ ] Kiểm tra gradient truyền về Mapper.
- [ ] Không thay đổi logic nội bộ của Mapping Network khi tích hợp.

---

## 17. Tóm tắt cần nhớ

```text
Visual prefix:
    attention_mask = 1
    label = -100

Caption token hợp lệ:
    attention_mask = 1
    label = token_id

Padding:
    attention_mask = 0
    label = -100
```

`attention_mask` quyết định vị trí nào GPT-2 được sử dụng làm ngữ cảnh. `labels` quyết định vị trí nào tham gia tính loss. Hai tensor có liên quan nhưng không cùng chức năng.

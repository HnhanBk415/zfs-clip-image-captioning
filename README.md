# ZeroCap & ClipCap Image Captioning
> This project investigates two different approaches for bridging visual representations from **CLIP** with the language generation capabilities of **GPT-2**

## 📖 Overview
A comparative image captioning project implementing **ClipCap** and **ZeroCap** on the **Flickr8k** dataset.

This project investigates two different approaches for bridging visual representations from **CLIP** with the language-generation capabilities of **GPT-2**:

- **ClipCap:** learns a mapping network that transforms CLIP image embeddings into prefix embeddings for GPT-2.
- **ZeroCap:** performs zero-shot image captioning by optimizing the GPT-2 context at inference time under CLIP-based visual-semantic guidance.

## 🖥️ Highlights

- **ClipCap training across multiple data regimes** using 1%, 5%, 10%, 25%, and 100% of the Flickr8k training set.
- **Zero-shot image captioning** with ZeroCap, requiring no task-specific caption training.
- **Fixed test set** shared across experiments for fair and reproducible comparison.
- **Unified evaluation pipeline** using CIDEr, BLEU-4, CLIPScore, and RefCLIPScore.
- **CLIP-based reranking** of generated caption candidates.
- **Google Colab notebooks** for training, inference, benchmarking, and evaluation.
  
## 👥 Core Members

- **Nguyễn Hoàng Nhân** —  **Project Lead** — VietNam National University Ho Chi Minh City University of Technology
- **Vũ Thành Nam** — VietNam National University Ho Chi Minh City University of Technology
- **Nguyễn Hữu Thiện** — VietNam National University Ho Chi Minh City University of Technology
- **Phan Hoài Bảo Khang** — Ho Chi Minh City University of Industry and Trade

## 📚 Dataset
🔗 Dataset: [Flickr8k on Kaggle](https://www.kaggle.com/datasets/adityajn105/flickr8k)
The project uses the **Flickr8k** dataset.

Each image is associated with **five human-written reference captions**.  
The dataset is split at the **image level** using a fixed random seed of `42`.

| Split | Images | Captions |
|---|---:|---:|
| Training | 6,462 | 32,310 |
| Validation | 807 | 4,035 |
| Test | 809 | 4,045 |
| **Total** | **8,078** | **40,390** |

### 🧩 ClipCap Training Subsets

To study the effect of training-data size, ClipCap is trained using multiple subsets of the Flickr8k training split.

| Subset | Images | Captions |
|---|---:|---:|
| **1%** | 64 | 320 |
| **5%** | 323 | 1,615 |
| **10%** | 646 | 3,230 |
| **25%** | 1,615 | 8,075 |
| **100%** | 6,462 | 32,310 |

> **Fair evaluation:** All ClipCap variants and ZeroCap are evaluated on the same fixed test set.

> **Note:** The 1%, 5%, 10%, 25%, and 100% training subsets apply only to ClipCap. ZeroCap remains fully zero-shot.

## 🛠️ Tech Stack

- **Language:** Python 
- **Deep Learning:** PyTorch
- **Vision-Language Model:** CLIP
- **Language Model:** GPT-2
- **Modeling & Tokenization:** Hugging Face Transformers
- **Dataset:** Flickr8k
- **Evaluation:** CIDEr, BLEU-4, CLIPScore, RefCLIPScore
- **Environment:** Google Colab / CUDA-enabled GPU

## 📁 Project Structure

```text
zfs-clip-image-captioning/
│
├── data/
│   └── flickr8k/
│       ├── metadata/
│       ├── splits/
│       ├── features/
│       └── tokenized/
│
├── docs/
│
├── notebook/
│   ├── preprocessing/
│   ├── clipcap/
│   ├── zerocap/
│   ├── evaluation/
│   └── demo/
│
├── scripts/
│   └── run_zerocap.py
│
├── src/
│   ├── clipcap/
│   ├── zerocap/
│   ├── common/
│   └── config/
│
├── app.py
│
├── tests/
│
├── requirements.txt
└── LICENSE
```

## 📊 Evaluation

All model variants are evaluated on the same fixed test set to ensure a fair and reproducible comparison.

The project reports four captioning metrics:

| Metric | Purpose |
|---|---|
| **CIDEr** | Measures agreement with human reference captions using TF-IDF weighted n-grams |
| **BLEU-4** | Measures 4-gram precision against the reference captions |
| **CLIPScore** | Measures semantic similarity between the input image and the generated caption |
| **RefCLIPScore** | Combines image-caption similarity with generated-reference caption similarity |

These metrics provide complementary perspectives on caption quality:

- **CIDEr** and **BLEU-4** focus on agreement with human-written reference captions.
- **CLIPScore** evaluates whether the generated caption is semantically aligned with the image.
- **RefCLIPScore** considers both visual relevance and similarity to reference captions.


## 📬 Contact

For questions, suggestions, or collaboration inquiries, feel free to contact the project members:

- **Nguyễn Hoàng Nhân** — [hoang.nhanbkag@gmail.com](mailto:hoang.nhanbkag@gmail.com)

- **Vũ Thành Nam** — [vthanhnam2006@gmail.com](mailto:vthanhnam2006@gmail.com)

- **Nguyễn Hữu Thiện** — [nguyenhuuthien963@gmail.com](mailto:nguyenhuuthien963@gmail.com)

- **Phan Hoài Bảo Khang** — [phanhoaibaokhanggg@gmail.com](mailto:phanhoaibaokhanggg@gmail.com)


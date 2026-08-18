import torch
from torch.utils.data import Dataset, DataLoader
from config import TOKENIZED_DIR, FEATURE_DIR

class ClipCapDataset(Dataset):
    def __init__(self, tokenized_path, clip_feature_data):
        self.tokenized_data = torch.load(tokenized_path)
        self.image_ids = self.tokenized_data["image_ids"] #list
        self.input_ids = self.tokenized_data["input_ids"]
        self.attention_mask = self.tokenized_data["attention_mask"]
        
        # matrix các vector feature ảnh
        self.clip_features = clip_feature_data["features"]
        
        # hashmap: img_id (name of img) -> idx
        self.img_id_to_idx = {
            img_id: idx 
            for idx, img_id in enumerate(clip_feature_data["image_ids"])
        }

        # validation
        token_image_ids = set(self.image_ids)
        clip_image_ids = set(clip_feature_data["image_ids"])

        assert len(clip_feature_data["image_ids"]) == len(clip_image_ids), \
            "Duplicate image_id in CLIP features"

        missing_clip = token_image_ids - clip_image_ids
        if missing_clip:
            raise ValueError(
                f"Missing CLIP features for {len(missing_clip)} images. "
                f"Examples: {list(missing_clip)[:5]}"
            )

        print(
            f"Alignment PASS: "
            f"{len(token_image_ids)} images ↔ {len(self.image_ids)} caption samples"
        )

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        feature_idx = self.img_id_to_idx[img_id]
        image_embed = self.clip_features[feature_idx]
        input_ids = self.input_ids[idx]
        attention_mask = self.attention_mask[idx]

        labels = input_ids.clone()
        labels[attention_mask == 0] = -100 # gán -100 cho các vị trí đệm để hàm CrossEntropyLoss skip
        
        return {
            "image_embed": image_embed,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

def create_dataloaders(batch_size=32, train_filename="train.pt"):
    clip_feature_path = FEATURE_DIR / "clip_features.pt"
    clip_feature_data = torch.load(clip_feature_path)
    
    dataloaders = {}

    train_path = TOKENIZED_DIR / train_filename
    train_dataset = ClipCapDataset(train_path, clip_feature_data)
    val_dataset = ClipCapDataset(TOKENIZED_DIR / "val.pt", clip_feature_data)
    test_dataset = ClipCapDataset(TOKENIZED_DIR / "test.pt", clip_feature_data)

    # check data leakage (intersection of any two sets must be empty)
    train_ids = set(train_dataset.image_ids)
    val_ids = set(val_dataset.image_ids)
    test_ids = set(test_dataset.image_ids)

    assert train_ids.isdisjoint(val_ids), \
        "Data leakage: Train and Validation share images"
    assert train_ids.isdisjoint(test_ids), \
        "Data leakage: Train and Test share images"
    assert val_ids.isdisjoint(test_ids), \
        "Data leakage: Validation and Test share images"

    dataloaders["train"] = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dataloaders["val"] = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    dataloaders["test"] = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return dataloaders
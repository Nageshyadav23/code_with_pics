# ===========================================
# ✅ Word-level TrOCR fine-tuning (stable)
# Works with older transformers on Colab
# ===========================================


# --- Mount Google Drive ---
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

# --- Imports ---
import os
import numpy as np
import pandas as pd
import torch
from PIL import Image
from datasets import Dataset
import matplotlib.pyplot as plt
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, Seq2SeqTrainer, Seq2SeqTrainingArguments

# --- Paths ---
PAIRS_CSV = "/content/drive/YOUR_CSV_PATH"
MODEL_NAME = "microsoft/trocr-base-handwritten"
OUTPUT_DIR = "/content/drive/YOUR_OUTPUT_PATH"

# --- Load data ---
df = pd.read_csv(PAIRS_CSV)
df = df[df["image"].apply(lambda x: os.path.exists(x))]
print(f"✅ Using {len(df)} image-text pairs")
dataset = Dataset.from_pandas(df)

# --- Device ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"⛑️ Training device: {device}")

# --- Processor & Model ---
processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)
model.to(device)

# --- Fix special tokens ---
if getattr(model.config, "decoder_start_token_id", None) is None:
    model.config.decoder_start_token_id = processor.tokenizer.bos_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.eos_token_id = processor.tokenizer.eos_token_id
try:
    model.config.vocab_size = model.config.decoder.vocab_size
except Exception:
    pass

# --- Word-level tokenizer ---
tokenizer = processor.tokenizer
tokenizer.do_basic_tokenize = False
tokenizer.do_lower_case = False
tokenizer.model_max_length = 64

def word_tokenize(text):
    words = text.strip().split()
    return words if words else [""]

def encode_words(text, max_length=64):
    words = word_tokenize(text)
    ids = [tokenizer.convert_tokens_to_ids(w) if w in tokenizer.vocab else tokenizer.unk_token_id for w in words]
    ids = ids[:max_length]
    ids += [tokenizer.pad_token_id] * (max_length - len(ids))
    return ids

# --- Preprocess function ---
def preprocess(examples):
    pixel_values_list = []
    labels_list = []
    for img_path, text in zip(examples["image"], examples["text"]):
        try:
            img = Image.open(img_path).convert("RGB")
            pixel_values = processor(images=img, return_tensors="pt").pixel_values[0].numpy()
            pixel_values_list.append(pixel_values)
            labels_list.append(encode_words(text))
        except Exception as e:
            print(f"⚠️ Skipping {img_path}: {e}")
            continue
    return {"pixel_values": pixel_values_list, "labels": labels_list}

# --- Split dataset ---
dataset = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

# --- Apply preprocessing ---
train_dataset = train_dataset.map(preprocess, batched=True, remove_columns=["image", "text"])
eval_dataset = eval_dataset.map(preprocess, batched=True, remove_columns=["image", "text"])

# --- Collate function ---
def collate_fn(batch):
    pixel_values = torch.tensor(np.stack([np.array(b["pixel_values"]) for b in batch]), dtype=torch.float32)
    labels = torch.tensor(np.stack([np.array(b["labels"]) for b in batch]), dtype=torch.long)
    return {"pixel_values": pixel_values, "labels": labels}

# --- Training arguments ---
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    num_train_epochs=5,
    logging_steps=100,
    predict_with_generate=True,
    fp16=torch.cuda.is_available(),
    report_to="none",
    save_steps=10**12
)

# --- Trainer ---
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=collate_fn
)

# --- Train ---
train_result = trainer.train()

# --- Save model ---
model.save_pretrained(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)
print(f"✅ Fine-tuned model saved to: {OUTPUT_DIR}")

# --- Plot loss ---
logs = getattr(trainer.state, "log_history", [])
if logs:
    df_logs = pd.DataFrame(logs)
    if "epoch" in df_logs.columns:
        train_losses = df_logs[df_logs["loss"].notnull()].groupby("epoch")["loss"].mean()
        eval_losses = df_logs[df_logs["eval_loss"].notnull()].groupby("epoch")["eval_loss"].mean()
        plt.figure(figsize=(8,5))
        if not train_losses.empty: plt.plot(train_losses.index, train_losses.values, marker="o", label="Training Loss")
        if not eval_losses.empty: plt.plot(eval_losses.index, eval_losses.values, marker="o", label="Eval Loss")
        plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.title("Training vs Eval Loss"); plt.grid(True); plt.show()
    else:
        if "loss" in df_logs.columns: plt.plot(df_logs["loss"].dropna().values, label="Training Loss")
        if "eval_loss" in df_logs.columns: plt.plot(df_logs["eval_loss"].dropna().values, label="Eval Loss")
        plt.legend(); plt.show()
else:
    print("⚠️ No logs to plot.")

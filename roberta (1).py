import os
import random
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    average_precision_score,
    confusion_matrix
)
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer, TrainerCallback

os.environ["WANDB_DISABLED"] = "true"

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed()

# Carregar dataset
columns = ['target', 'ids', 'date', 'flag', 'user', 'text']
df = pd.read_csv('dataset.csv', header=None, names=columns, encoding='latin1')
df['target'] = df['target'].map({0: 0, 4: 1})

train_df, test_df = train_test_split(df, test_size=0.25, stratify=df["target"], random_state=42)

model_name = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_data(texts):
    return tokenizer(texts.tolist(), max_length=128, padding=True, truncation=True)

train_inputs = tokenize_data(train_df['text'])
test_inputs = tokenize_data(test_df['text'])

class NewsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = NewsDataset(train_inputs, train_df['target'].tolist())
test_dataset = NewsDataset(test_inputs, test_df['target'].tolist())

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="macro")
    precision = precision_score(labels, predictions, average="macro")
    recall = recall_score(labels, predictions, average="macro")
    pr_auc = average_precision_score(
        np.eye(len(set(labels)))[labels], logits, average="macro"
    )

    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "pr_auc": pr_auc,
    }

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=df["target"].nunique()
)

# Histórico
train_loss_history = []
eval_loss_history = []
eval_acc_history = []

class LossAccuracyCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            if 'loss' in logs:
                train_loss_history.append(logs['loss'])
            if 'eval_loss' in logs:
                eval_loss_history.append(logs['eval_loss'])
            if 'eval_accuracy' in logs:
                eval_acc_history.append(logs['eval_accuracy'])

training_args = TrainingArguments(
    output_dir="./results_roberta_base",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    report_to=None,
    num_train_epochs=4,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    learning_rate=1e-5,
    warmup_ratio=0.1,
    logging_dir='./logs',
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
    callbacks=[LossAccuracyCallback()],
)

# Treinamento (com retomada automática do último checkpoint se houver)
try:
    trainer.train(resume_from_checkpoint=True)
except KeyboardInterrupt:
    print("\nTreinamento interrompido manualmente. Você pode retomá-lo depois.")

# Avaliação final
eval_results = trainer.evaluate()

print("\nMétricas finais de avaliação:")
print("=============================")
for metric_name in ["eval_accuracy", "eval_f1", "eval_precision", "eval_recall", "eval_pr_auc"]:
    print(f"{metric_name}: {round(eval_results[metric_name], 4)}")

# Matriz de confusão
logits, labels = trainer.predict(test_dataset)[:2]
predictions = np.argmax(logits, axis=-1)
cm = confusion_matrix(labels, predictions)

print("\nMatriz de Confusão:")
print("===================")
print(cm)

# Gráficos em português (limitando a 4 épocas)
epocas = [1, 2, 3, 4]
eval_acc_history = eval_acc_history[:4]
eval_loss_history = eval_loss_history[:4]
train_loss_history = train_loss_history[:4]

# Gráfico de Acurácia
plt.figure(figsize=(8, 6))
plt.plot(epocas, eval_acc_history, label="Acurácia na validação", color="blue", linewidth=2)
plt.title("RoBERTa: Acurácia na validação", fontsize=14)
plt.xlabel("Época", fontsize=12)
plt.ylabel("Acurácia", fontsize=12)
plt.xticks(epocas)
plt.legend(loc="upper left")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("acuracia_por_epoca.pdf")

# Gráfico de Loss
plt.figure(figsize=(8, 6))
plt.plot(epocas, train_loss_history, label="Loss no treino", color="red", linewidth=2)
plt.plot(epocas, eval_loss_history, label="Loss na validação", color="blue", linewidth=2)
plt.title("RoBERTa: Loss no treino e validação", fontsize=14)
plt.xlabel("Época", fontsize=12)
plt.ylabel("Loss", fontsize=12)
plt.xticks(epocas)
plt.legend(loc="upper right")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("loss_por_epoca.pdf")

# Gráfico da Matriz de Confusão
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negativo', 'Positivo'],
            yticklabels=['Negativo', 'Positivo'])
plt.title("Matriz de Confusão", fontsize=14)
plt.xlabel("Classe predita", fontsize=12)
plt.ylabel("Classe real", fontsize=12)
plt.tight_layout()
plt.savefig("matriz_de_confusao.pdf")

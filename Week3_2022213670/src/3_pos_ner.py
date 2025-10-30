import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import Counter

if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print(f"Using device: {device}")


class CRF(nn.Module):
    """条件随机场层"""

    def __init__(self, num_tags, batch_first=True):
        super(CRF, self).__init__()
        self.num_tags = num_tags
        self.batch_first = batch_first

        # 转移矩阵: transitions[i, j] 表示从标签j转移到标签i的分数
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))

        # 开始和结束标签
        self.start_transitions = nn.Parameter(torch.randn(num_tags))
        self.end_transitions = nn.Parameter(torch.randn(num_tags))

    def forward(self, emissions, tags, mask=None):
        """计算负对数似然损失"""
        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.bool)

        numerator = self._compute_score(emissions, tags, mask)

        denominator = self._compute_normalizer(emissions, mask)

        return torch.mean(denominator - numerator)

    def decode(self, emissions, mask=None):
        """使用Viterbi算法解码最优路径"""
        if mask is None:
            batch_size, seq_len, _ = emissions.shape
            mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=emissions.device)

        return self._viterbi_decode(emissions, mask)

    def _compute_score(self, emissions, tags, mask):
        """计算给定标签序列的分数（对数空间）"""
        batch_size, seq_len = tags.shape

        score = self.start_transitions[tags[:, 0]]
        score = score + emissions[:, 0].gather(1, tags[:, 0].unsqueeze(1)).squeeze(1)

        for i in range(1, seq_len):
            score = score + self.transitions[tags[:, i], tags[:, i-1]] * mask[:, i]
            score = score + emissions[:, i].gather(1, tags[:, i].unsqueeze(1)).squeeze(1) * mask[:, i]

        last_tag_indices = mask.sum(1).long() - 1
        last_tags = tags.gather(1, last_tag_indices.unsqueeze(1)).squeeze(1)
        score = score + self.end_transitions[last_tags]

        return score

    def _compute_normalizer(self, emissions, mask):
        """使用前向算法计算配分函数"""
        batch_size, seq_len, num_tags = emissions.shape

        score = self.start_transitions + emissions[:, 0]

        for i in range(1, seq_len):
            broadcast_score = score.unsqueeze(2)
            broadcast_emissions = emissions[:, i].unsqueeze(1)

            next_score = broadcast_score + self.transitions + broadcast_emissions
            next_score = torch.logsumexp(next_score, dim=1)

            score = torch.where(mask[:, i].unsqueeze(1), next_score, score)

        score += self.end_transitions
        return torch.logsumexp(score, dim=1)

    def _viterbi_decode(self, emissions, mask):
        """Viterbi解码"""
        batch_size, seq_len, num_tags = emissions.shape

        score = self.start_transitions + emissions[:, 0]
        history = []

        for i in range(1, seq_len):
            broadcast_score = score.unsqueeze(2)
            broadcast_emissions = emissions[:, i].unsqueeze(1)

            next_score = broadcast_score + self.transitions + broadcast_emissions

            next_score, indices = next_score.max(dim=1)

            score = torch.where(mask[:, i].unsqueeze(1), next_score, score)
            history.append(indices)

        score += self.end_transitions

        best_tags_list = []
        _, best_last_tags = score.max(dim=1)
        best_tags = [best_last_tags.unsqueeze(1)]

        for indices in reversed(history):
            best_last_tags = indices.gather(1, best_tags[-1])
            best_tags.append(best_last_tags)

        best_tags.reverse()
        best_tags = torch.cat(best_tags, dim=1)

        return best_tags


class CharBiLSTMCRF(nn.Module):
    """基于字符的BiLSTM-CRF模型"""

    def __init__(self, char_vocab_size, tag_vocab_size, char_embed_dim=100,
                 hidden_dim=256, num_layers=2, dropout=0.5):
        super(CharBiLSTMCRF, self).__init__()

        self.char_embedding = nn.Embedding(char_vocab_size, char_embed_dim, padding_idx=0)

        self.lstm = nn.LSTM(
            char_embed_dim,
            hidden_dim // 2,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.dropout = nn.Dropout(dropout)
        self.hidden2tag = nn.Linear(hidden_dim, tag_vocab_size)
        self.crf = CRF(tag_vocab_size, batch_first=True)

    def forward(self, chars, tags, mask=None):
        """前向传播（训练时）"""
        # chars: (batch_size, seq_len)
        char_embeds = self.char_embedding(chars)
        char_embeds = self.dropout(char_embeds)

        lstm_out, _ = self.lstm(char_embeds)
        lstm_out = self.dropout(lstm_out)

        emissions = self.hidden2tag(lstm_out)

        loss = self.crf(emissions, tags, mask)

        return loss

    def predict(self, chars, mask=None):
        """预测标签序列"""
        char_embeds = self.char_embedding(chars)
        lstm_out, _ = self.lstm(char_embeds)
        emissions = self.hidden2tag(lstm_out)

        best_tags = self.crf.decode(emissions, mask)

        return best_tags


class POSNERDataset:
    """POS和NER数据集"""

    def __init__(self, data_path, task='pos'):
        """
        task: 'pos' for POS tagging, 'ner' for NER
        """
        self.task = task
        self.data = []

        self.char2idx = {'<PAD>': 0, '<UNK>': 1}

        if task == 'pos':
            self.tag2idx = {'<PAD>': 0}
        else:  # NER
            self.tag2idx = {'<PAD>': 0, 'O': 1}

        self.load_data(data_path)

        self.idx2char = {idx: char for char, idx in self.char2idx.items()}
        self.idx2tag = {idx: tag for tag, idx in self.tag2idx.items()}

        print(f"Task: {task.upper()}")
        print(f"Char vocab size: {len(self.char2idx)}")
        print(f"Tag vocab size: {len(self.tag2idx)}")
        print(f"Data size: {len(self.data)}")

    def load_data(self, data_path):
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                chars = []
                pos_tags = []
                ner_tags = []

                tokens = line.split()
                for token in tokens:
                    if '/' not in token:
                        continue

                    parts = token.rsplit('/', 1)
                    if len(parts) != 2:
                        continue

                    word, pos = parts

                    for i, char in enumerate(word):
                        chars.append(char)
                        pos_tags.append(pos)

                        if pos in ['nr', 'Nr']:
                            ner_tag = 'B-PER' if i == 0 else 'I-PER'
                        elif pos in ['ns', 'Ns']:
                            ner_tag = 'B-LOC' if i == 0 else 'I-LOC'
                        elif pos in ['nt', 'Nt']:
                            ner_tag = 'B-ORG' if i == 0 else 'I-ORG'
                        else:
                            ner_tag = 'O'

                        ner_tags.append(ner_tag)

                if not chars:
                    continue

                for char in chars:
                    if char not in self.char2idx:
                        self.char2idx[char] = len(self.char2idx)

                if self.task == 'pos':
                    for tag in pos_tags:
                        if tag not in self.tag2idx:
                            self.tag2idx[tag] = len(self.tag2idx)
                    self.data.append((chars, pos_tags))
                else:
                    for tag in ner_tags:
                        if tag not in self.tag2idx:
                            self.tag2idx[tag] = len(self.tag2idx)
                    self.data.append((chars, ner_tags))

    def encode(self, chars, tags):
        char_ids = [self.char2idx.get(c, self.char2idx['<UNK>']) for c in chars]
        tag_ids = [self.tag2idx[t] for t in tags]
        return char_ids, tag_ids

    def decode_chars(self, char_ids):
        return ''.join([self.idx2char.get(i, '<UNK>') for i in char_ids if i != 0])

    def decode_tags(self, tag_ids):
        return [self.idx2tag.get(i, '<PAD>') for i in tag_ids if i != 0]

    def get_data_loaders(self, train_ratio=0.9, max_len=150):
        filtered_data = []
        for chars, tags in self.data:
            if len(chars) <= max_len and len(chars) > 0:
                filtered_data.append((chars, tags))

        np.random.shuffle(filtered_data)
        split_idx = int(len(filtered_data) * train_ratio)

        train_data = filtered_data[:split_idx]
        val_data = filtered_data[split_idx:]

        print(f"Train: {len(train_data)}, Val: {len(val_data)}")

        return train_data, val_data


def collate_fn(batch, dataset):
    chars_batch, tags_batch = [], []
    lengths = []

    for chars, tags in batch:
        char_ids, tag_ids = dataset.encode(chars, tags)
        chars_batch.append(char_ids)
        tags_batch.append(tag_ids)
        lengths.append(len(char_ids))

    max_len = max(lengths)

    padded_chars = []
    padded_tags = []
    masks = []

    for char_ids, tag_ids in zip(chars_batch, tags_batch):
        pad_len = max_len - len(char_ids)
        padded_chars.append(char_ids + [0] * pad_len)
        padded_tags.append(tag_ids + [0] * pad_len)
        masks.append([1] * len(char_ids) + [0] * pad_len)

    chars_tensor = torch.tensor(padded_chars, dtype=torch.long)
    tags_tensor = torch.tensor(padded_tags, dtype=torch.long)
    mask_tensor = torch.tensor(masks, dtype=torch.bool)

    return chars_tensor, tags_tensor, mask_tensor


def train_epoch(model, data, dataset, optimizer, batch_size=32, epoch=0, num_epochs=1):
    model.train()
    epoch_loss = 0
    num_batches = 0

    np.random.shuffle(data)

    pbar = tqdm(range(0, len(data), batch_size),
                desc=f"Epoch {epoch+1}/{num_epochs}")

    for i in pbar:
        batch = data[i:i+batch_size]
        chars, tags, mask = collate_fn(batch, dataset)

        chars = chars.to(device)
        tags = tags.to(device)
        mask = mask.to(device)

        optimizer.zero_grad()

        loss = model(chars, tags, mask)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

        optimizer.step()

        epoch_loss += loss.item()
        num_batches += 1

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return epoch_loss / num_batches


def evaluate(model, data, dataset, batch_size=32):
    model.eval()
    epoch_loss = 0
    num_batches = 0

    correct = 0
    total = 0

    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            chars, tags, mask = collate_fn(batch, dataset)

            chars = chars.to(device)
            tags = tags.to(device)
            mask = mask.to(device)

            loss = model(chars, tags, mask)
            epoch_loss += loss.item()
            num_batches += 1

            predictions = model.predict(chars, mask)

            mask_np = mask.cpu().numpy()
            tags_np = tags.cpu().numpy()
            pred_np = predictions.cpu().numpy()

            for j in range(len(batch)):
                seq_len = mask_np[j].sum()
                correct += (pred_np[j][:seq_len] == tags_np[j][:seq_len]).sum()
                total += seq_len

    accuracy = correct / total if total > 0 else 0

    return epoch_loss / num_batches, accuracy


def plot_training_curves(train_losses, val_losses, train_accs, val_accs, task, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss', marker='o')
    ax1.plot(range(1, len(val_losses) + 1), val_losses, label='Val Loss', marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{task.upper()} Training Loss')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(range(1, len(train_accs) + 1), train_accs, label='Train Acc', marker='o')
    ax2.plot(range(1, len(val_accs) + 1), val_accs, label='Val Acc', marker='s')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title(f'{task.upper()} Training Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to {save_path}")


def train_model(task='pos'):
    print(f"\n{'='*50}")
    print(f"Training {task.upper()} Model")
    print('='*50)

    data_path = '../data/result-rmrb.txt'

    dataset = POSNERDataset(data_path, task=task)
    train_data, val_data = dataset.get_data_loaders()

    model = CharBiLSTMCRF(
        char_vocab_size=len(dataset.char2idx),
        tag_vocab_size=len(dataset.tag2idx),
        char_embed_dim=100,
        hidden_dim=256,
        num_layers=2,
        dropout=0.5
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 30
    batch_size = 32

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    best_val_acc = 0

    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_data, dataset, optimizer, batch_size, epoch, num_epochs)
        val_loss, val_acc = evaluate(model, val_data, dataset, batch_size)
        _, train_acc = evaluate(model, train_data[:len(val_data)], dataset, batch_size)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_path = f'../model/{task}_best.pth'
            torch.save({
                'model_state_dict': model.state_dict(),
                'char2idx': dataset.char2idx,
                'tag2idx': dataset.tag2idx,
                'idx2char': dataset.idx2char,
                'idx2tag': dataset.idx2tag
            }, model_path)
            print(f"  → Best model saved (val_acc: {val_acc:.4f})")

    final_model_path = f'../model/{task}_final.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'char2idx': dataset.char2idx,
        'tag2idx': dataset.tag2idx,
        'idx2char': dataset.idx2char,
        'idx2tag': dataset.idx2tag
    }, final_model_path)

    history_path = f'../result/{task}_training_history.json'
    with open(history_path, 'w') as f:
        json.dump({
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accs': train_accs,
            'val_accs': val_accs
        }, f, indent=2)

    curves_path = f'../result/{task}_training_curves.png'
    plot_training_curves(train_losses, val_losses, train_accs, val_accs, task, curves_path)

    print(f"\n{task.upper()} Prediction Examples:")
    print("-" * 50)

    test_examples = val_data[:5]
    predictions = []

    for chars, true_tags in test_examples:
        char_ids, tag_ids = dataset.encode(chars, true_tags)

        chars_tensor = torch.tensor([char_ids], dtype=torch.long).to(device)
        mask = torch.ones_like(chars_tensor, dtype=torch.bool)

        pred_tags = model.predict(chars_tensor, mask)
        pred_tags = pred_tags[0].cpu().tolist()[:len(chars)]

        text = ''.join(chars)
        pred_tag_names = [dataset.idx2tag[t] for t in pred_tags]

        print(f"Text: {text}")
        print(f"True: {true_tags[:20]}...")
        print(f"Pred: {pred_tag_names[:20]}...")
        print()

        predictions.append({
            'text': text,
            'true_tags': true_tags,
            'pred_tags': pred_tag_names
        })

    pred_path = f'../result/{task}_predictions.json'
    with open(pred_path, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    print(f"\n{task.upper()} model training completed!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")

    return model, dataset


def main():
    os.makedirs('../model', exist_ok=True)
    os.makedirs('../result', exist_ok=True)

    train_model('pos')
    train_model('ner')

    print("\n" + "="*50)
    print("All models training completed!")
    print("="*50)


if __name__ == '__main__':
    main()

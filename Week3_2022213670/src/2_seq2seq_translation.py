import torch
import torch.nn as nn
import torch.optim as optim
import json
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pad_sequence

if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print(f"Using device: {device}")


class Encoder(nn.Module):
    """编码器"""

    def __init__(self, vocab_size, embed_dim, hidden_dim, rnn_type='LSTM', num_layers=2, dropout=0.3):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn_type = rnn_type

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        if rnn_type == 'RNN':
            self.rnn = nn.RNN(embed_dim, hidden_dim, num_layers,
                              batch_first=True, dropout=dropout if num_layers > 1 else 0)
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(embed_dim, hidden_dim, num_layers,
                              batch_first=True, dropout=dropout if num_layers > 1 else 0)
        elif rnn_type == 'LSTM':
            self.rnn = nn.LSTM(embed_dim, hidden_dim, num_layers,
                               batch_first=True, dropout=dropout if num_layers > 1 else 0)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_lengths):
        # src: (batch_size, seq_len)
        embedded = self.dropout(self.embedding(src))

        # Pack padded sequence
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, src_lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        if self.rnn_type == 'LSTM':
            outputs, (hidden, cell) = self.rnn(packed)
            # Unpack
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
            return outputs, hidden, cell
        else:
            outputs, hidden = self.rnn(packed)
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
            return outputs, hidden, None


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs, mask):
        # hidden: (batch_size, hidden_dim)
        # encoder_outputs: (batch_size, src_len, hidden_dim)

        src_len = encoder_outputs.shape[1]

        # Repeat hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)

        # Concatenate and compute attention scores
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)

        # Mask attention scores
        attention = attention.masked_fill(mask == 0, -1e10)

        return torch.softmax(attention, dim=1)


class Decoder(nn.Module):
    """解码器（带注意力机制）"""

    def __init__(self, vocab_size, embed_dim, hidden_dim, rnn_type='LSTM', num_layers=2, dropout=0.3):
        super(Decoder, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn_type = rnn_type

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.attention = Attention(hidden_dim)

        if rnn_type == 'RNN':
            self.rnn = nn.RNN(embed_dim + hidden_dim, hidden_dim, num_layers,
                              batch_first=True, dropout=dropout if num_layers > 1 else 0)
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(embed_dim + hidden_dim, hidden_dim, num_layers,
                              batch_first=True, dropout=dropout if num_layers > 1 else 0)
        elif rnn_type == 'LSTM':
            self.rnn = nn.LSTM(embed_dim + hidden_dim, hidden_dim, num_layers,
                               batch_first=True, dropout=dropout if num_layers > 1 else 0)

        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell, encoder_outputs, mask):
        # input: (batch_size)
        # hidden: (num_layers, batch_size, hidden_dim)

        input = input.unsqueeze(1)  # (batch_size, 1)
        embedded = self.dropout(self.embedding(input))

        # Compute attention
        # Use top layer hidden state for attention
        attn_weights = self.attention(hidden[-1], encoder_outputs, mask)
        attn_weights = attn_weights.unsqueeze(1)  # (batch_size, 1, src_len)

        # Weighted sum of encoder outputs
        context = torch.bmm(attn_weights, encoder_outputs)  # (batch_size, 1, hidden_dim)

        # Concatenate embedded input and context
        rnn_input = torch.cat((embedded, context), dim=2)

        if self.rnn_type == 'LSTM':
            output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        else:
            output, hidden = self.rnn(rnn_input, hidden)

        # Prediction
        prediction = self.fc_out(output.squeeze(1))

        return prediction, hidden, cell, attn_weights.squeeze(1)


class Seq2Seq(nn.Module):
    """Seq2Seq模型"""

    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def create_mask(self, src):
        mask = (src != 0).to(device)
        return mask

    def forward(self, src, src_lengths, trg, teacher_forcing_ratio=0.5):
        # src: (batch_size, src_len)
        # trg: (batch_size, trg_len)

        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.vocab_size

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(device)

        encoder_outputs, hidden, cell = self.encoder(src, src_lengths)

        mask = self.create_mask(src)

        input = trg[:, 0]

        for t in range(1, trg_len):
            output, hidden, cell, _ = self.decoder(input, hidden, cell, encoder_outputs, mask)
            outputs[:, t] = output

            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[:, t] if teacher_force else top1

        return outputs


class TranslationDataset:

    def __init__(self, data_dir):
        self.data = []
        self.src_vocab = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}
        self.trg_vocab = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}

        json_files = [
            'compress.json',
            '俗世奇人part1.json',
            '俗世奇人part2.json',
            '熬柿子.json',
            '白蛇传.json'
        ]

        for json_file in json_files:
            file_path = os.path.join(data_dir, json_file)
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                        if isinstance(data, list) and len(data) > 0:
                            self.data.extend(data)
                        elif isinstance(data, dict):
                            dict_pairs = []
                            for tianjin_word, explanation in data.items():
                                if '|' in explanation:
                                    examples = explanation.split('|')[1:]
                                    for example in examples:
                                        if '~' in example:
                                            mandarin = example.replace('~', tianjin_word)
                                            dict_pairs.append({
                                                'source': example,
                                                'translation': mandarin
                                            })
                            if dict_pairs:
                                self.data.extend(dict_pairs)
                except Exception as e:
                    print(f"Error loading {json_file}: {e}")

        print(f"Loaded {len(self.data)} translation pairs")

        self.build_vocab()

    def build_vocab(self):
        for item in self.data:
            src = item.get('source', '')
            trg = item.get('translation', '')

            for char in src:
                if char not in self.src_vocab:
                    self.src_vocab[char] = len(self.src_vocab)

            for char in trg:
                if char not in self.trg_vocab:
                    self.trg_vocab[char] = len(self.trg_vocab)

        self.idx2src = {idx: char for char, idx in self.src_vocab.items()}
        self.idx2trg = {idx: char for char, idx in self.trg_vocab.items()}

        print(f"Source vocab size: {len(self.src_vocab)}")
        print(f"Target vocab size: {len(self.trg_vocab)}")

    def encode_sentence(self, sentence, vocab, max_len=100):
        indices = [vocab.get('<sos>')]
        for char in sentence[:max_len-2]:
            indices.append(vocab.get(char, vocab.get('<unk>')))
        indices.append(vocab.get('<eos>'))
        return indices

    def decode_sentence(self, indices, idx2vocab):
        chars = []
        for idx in indices:
            if idx == self.trg_vocab['<eos>']:
                break
            if idx not in [0, 1, 2]:
                chars.append(idx2vocab.get(idx, '<unk>'))
        return ''.join(chars)

    def get_data_loader(self, batch_size=32, train_ratio=0.9):
        encoded_data = []
        for item in self.data:
            src = self.encode_sentence(item.get('source', ''), self.src_vocab)
            trg = self.encode_sentence(item.get('translation', ''), self.trg_vocab)
            if len(src) > 2 and len(trg) > 2:
                encoded_data.append((src, trg))

        np.random.shuffle(encoded_data)
        split_idx = int(len(encoded_data) * train_ratio)
        train_data = encoded_data[:split_idx]
        val_data = encoded_data[split_idx:]

        print(f"Train size: {len(train_data)}, Val size: {len(val_data)}")

        return train_data, val_data


def collate_fn(batch):
    src_batch, trg_batch = [], []
    src_lengths, trg_lengths = [], []

    for src, trg in batch:
        src_batch.append(torch.tensor(src, dtype=torch.long))
        trg_batch.append(torch.tensor(trg, dtype=torch.long))
        src_lengths.append(len(src))
        trg_lengths.append(len(trg))

    src_batch = pad_sequence(src_batch, batch_first=True, padding_value=0)
    trg_batch = pad_sequence(trg_batch, batch_first=True, padding_value=0)
    src_lengths = torch.tensor(src_lengths, dtype=torch.long)

    return src_batch, trg_batch, src_lengths


def train_epoch(model, data, optimizer, criterion, batch_size=32, epoch=0, num_epochs=1):
    model.train()
    epoch_loss = 0

    if len(data) == 0:
        return 0.0

    np.random.shuffle(data)

    num_batches = (len(data) + batch_size - 1) // batch_size

    pbar = tqdm(range(0, len(data), batch_size),
                desc=f"Epoch {epoch+1}/{num_epochs}")

    for i in pbar:
        batch = data[i:i+batch_size]
        src, trg, src_lengths = collate_fn(batch)

        src = src.to(device)
        trg = trg.to(device)
        src_lengths = src_lengths.to(device)

        optimizer.zero_grad()

        output = model(src, src_lengths, trg, teacher_forcing_ratio=0.5)

        output_dim = output.shape[-1]
        output = output[:, 1:].contiguous().view(-1, output_dim)
        trg = trg[:, 1:].contiguous().view(-1)

        loss = criterion(output, trg)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        epoch_loss += loss.item()

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return epoch_loss / num_batches


def evaluate(model, data, criterion, batch_size=32):
    model.eval()
    epoch_loss = 0

    if len(data) == 0:
        return 0.0

    num_batches = (len(data) + batch_size - 1) // batch_size

    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            src, trg, src_lengths = collate_fn(batch)

            src = src.to(device)
            trg = trg.to(device)
            src_lengths = src_lengths.to(device)

            output = model(src, src_lengths, trg, teacher_forcing_ratio=0)

            output_dim = output.shape[-1]
            output = output[:, 1:].contiguous().view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)

            loss = criterion(output, trg)
            epoch_loss += loss.item()

    return epoch_loss / num_batches


def translate_sentence(model, sentence, dataset, max_len=100):
    model.eval()

    src_indices = dataset.encode_sentence(sentence, dataset.src_vocab, max_len)
    src_tensor = torch.tensor(src_indices, dtype=torch.long).unsqueeze(0).to(device)
    src_lengths = torch.tensor([len(src_indices)], dtype=torch.long).to(device)

    with torch.no_grad():
        encoder_outputs, hidden, cell = model.encoder(src_tensor, src_lengths)

        mask = model.create_mask(src_tensor)

        trg_indices = [dataset.trg_vocab['<sos>']]

        for _ in range(max_len):
            trg_tensor = torch.tensor([trg_indices[-1]], dtype=torch.long).to(device)

            output, hidden, cell, _ = model.decoder(
                trg_tensor, hidden, cell, encoder_outputs, mask
            )

            pred_token = output.argmax(1).item()
            trg_indices.append(pred_token)

            if pred_token == dataset.trg_vocab['<eos>']:
                break

    return dataset.decode_sentence(trg_indices, dataset.idx2trg)


def plot_losses(train_losses, val_losses, model_name, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Val Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{model_name} Training Progress')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Loss curves saved to {save_path}")


def train_model(rnn_type='LSTM'):
    print(f"\n{'='*50}")
    print(f"Training {rnn_type} Model")
    print('='*50)

    data_dir = '../data/Tianjin_dataset-main'
    dataset = TranslationDataset(data_dir)

    if len(dataset.data) == 0:
        print(f"Error: No data loaded! Skipping {rnn_type} model.")
        return None, None

    train_data, val_data = dataset.get_data_loader(batch_size=32)

    embed_dim = 256
    hidden_dim = 512
    num_layers = 2
    dropout = 0.3

    encoder = Encoder(len(dataset.src_vocab), embed_dim, hidden_dim, rnn_type, num_layers, dropout)
    decoder = Decoder(len(dataset.trg_vocab), embed_dim, hidden_dim, rnn_type, num_layers, dropout)
    model = Seq2Seq(encoder, decoder).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    num_epochs = 30
    batch_size = 32

    train_losses = []
    val_losses = []

    best_val_loss = float('inf')

    print(f"\n{'='*50}")
    print(f"Training {rnn_type} Model")
    print(f"{'='*50}")

    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_data, optimizer, criterion, batch_size, epoch, num_epochs)
        val_loss = evaluate(model, val_data, criterion, batch_size)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = f'../model/seq2seq_{rnn_type.lower()}_best.pth'
            torch.save(model.state_dict(), model_path)
            print(f"  → Best model saved (val_loss: {val_loss:.4f})")

    final_model_path = f'../model/seq2seq_{rnn_type.lower()}_final.pth'
    torch.save(model.state_dict(), final_model_path)

    history_path = f'../result/seq2seq_{rnn_type.lower()}_history.json'
    with open(history_path, 'w') as f:
        json.dump({
            'train_losses': train_losses,
            'val_losses': val_losses
        }, f, indent=2)

    loss_plot_path = f'../result/seq2seq_{rnn_type.lower()}_loss.png'
    plot_losses(train_losses, val_losses, rnn_type, loss_plot_path)

    print(f"\n{rnn_type} Translation Examples:")
    print("-" * 50)

    test_sentences = [
        "苏大夫本名苏金散",
        "民国初年在小白楼一带",
        "正骨拿环",
        "天津卫挂头牌"
    ]

    translations = []
    for sent in test_sentences:
        translation = translate_sentence(model, sent, dataset)
        print(f"Source: {sent}")
        print(f"Translation: {translation}")
        print()
        translations.append({'source': sent, 'translation': translation})

    trans_path = f'../result/seq2seq_{rnn_type.lower()}_translations.json'
    with open(trans_path, 'w', encoding='utf-8') as f:
        json.dump(translations, f, ensure_ascii=False, indent=2)

    return model, dataset


def main():
    os.makedirs('../model', exist_ok=True)
    os.makedirs('../result', exist_ok=True)

    for rnn_type in ['RNN', 'GRU', 'LSTM']:
        train_model(rnn_type)

    print("\n" + "="*50)
    print("All Seq2Seq models training completed!")
    print("="*50)


if __name__ == '__main__':
    main()

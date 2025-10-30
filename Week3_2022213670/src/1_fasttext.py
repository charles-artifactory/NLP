"""
FastText词向量训练
使用Skip-gram with Negative Sampling和subword information
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import os
from collections import Counter
from tqdm import tqdm
import matplotlib.pyplot as plt

if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print(f"Using device: {device}")


class FastTextModel(nn.Module):
    """FastText模型：Skip-gram + Subword"""

    def __init__(self, vocab_size, ngram_vocab_size, embedding_dim):
        super(FastTextModel, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.ngram_embeddings = nn.Embedding(ngram_vocab_size, embedding_dim)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, center_word, ngrams, context_words, neg_words):
        # center word embedding: (batch_size, embed_dim)
        center_embed = self.word_embeddings(center_word)

        # ngram embeddings: (batch_size, num_ngrams, embed_dim)
        if ngrams.size(1) > 0:
            ngram_embed = self.ngram_embeddings(ngrams).mean(dim=1)
            center_embed = center_embed + ngram_embed

        # positive context: (batch_size, embed_dim)
        pos_embed = self.context_embeddings(context_words)

        # negative samples: (batch_size, num_neg, embed_dim)
        neg_embed = self.context_embeddings(neg_words)

        # positive score
        pos_score = torch.sum(center_embed * pos_embed, dim=1)
        pos_loss = -torch.log(torch.sigmoid(pos_score) + 1e-10)

        # negative score
        neg_score = torch.bmm(neg_embed, center_embed.unsqueeze(2)).squeeze(2)
        neg_loss = -torch.sum(torch.log(torch.sigmoid(-neg_score) + 1e-10), dim=1)

        return (pos_loss + neg_loss).mean()

    def get_word_embedding(self, word_idx, ngram_indices):
        """获取词向量（词向量+子词向量的平均）"""
        word_embed = self.word_embeddings(word_idx)
        if len(ngram_indices) > 0:
            ngram_embed = self.ngram_embeddings(torch.tensor(ngram_indices, device=device)).mean(dim=0)
            return word_embed + ngram_embed
        return word_embed


class FastTextTrainer:
    def __init__(self, data_path, embedding_dim=100, window_size=5,
                 min_count=5, num_neg=5, min_n=3, max_n=6):
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.min_count = min_count
        self.num_neg = num_neg
        self.min_n = min_n
        self.max_n = max_n

        self.sentences = self.load_data(data_path)

        self.build_vocab()

        self.build_ngram_vocab()

        self.model = FastTextModel(
            len(self.word2idx),
            len(self.ngram2idx),
            embedding_dim
        ).to(device)

        self.neg_sampling_probs = self.get_neg_sampling_probs()

    def load_data(self, data_path):
        """加载数据"""
        sentences = []

        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                words = [word.split('/')[0] for word in line.split() if '/' in word]
                if words:
                    sentences.append(words)

        print(f"Loaded {len(sentences)} sentences")
        return sentences

    def build_vocab(self):
        """构建词汇表"""
        word_counts = Counter()
        for sentence in self.sentences:
            word_counts.update(sentence)

        vocab = [word for word, count in word_counts.items() if count >= self.min_count]

        self.word2idx = {word: idx for idx, word in enumerate(vocab)}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        self.word_counts = {word: word_counts[word] for word in vocab}

        print(f"Vocabulary size: {len(self.word2idx)}")

    def get_ngrams(self, word):
        """获取词的n-gram子词"""
        word = f"<{word}>"
        ngrams = []
        for n in range(self.min_n, self.max_n + 1):
            for i in range(len(word) - n + 1):
                ngrams.append(word[i:i+n])
        return ngrams

    def build_ngram_vocab(self):
        """构建n-gram词汇表"""
        ngram_counts = Counter()
        for word in self.word2idx.keys():
            ngrams = self.get_ngrams(word)
            ngram_counts.update(ngrams)

        ngram_vocab = [ng for ng, count in ngram_counts.items() if count >= 5]

        self.ngram2idx = {ng: idx for idx, ng in enumerate(ngram_vocab)}

        print(f"N-gram vocabulary size: {len(self.ngram2idx)}")

    def get_neg_sampling_probs(self):
        """计算负采样概率分布"""
        freq = np.array([self.word_counts.get(self.idx2word[i], 1)
                        for i in range(len(self.word2idx))])
        probs = np.power(freq, 0.75)
        probs = probs / probs.sum()
        return probs

    def generate_training_data(self):
        """生成训练数据"""
        training_data = []

        for sentence in self.sentences:
            word_indices = [self.word2idx[w] for w in sentence if w in self.word2idx]

            for i, center_word in enumerate(word_indices):
                start = max(0, i - self.window_size)
                end = min(len(word_indices), i + self.window_size + 1)

                for j in range(start, end):
                    if i != j:
                        context_word = word_indices[j]

                        center_word_str = self.idx2word[center_word]
                        ngrams = self.get_ngrams(center_word_str)
                        ngram_indices = [self.ngram2idx[ng] for ng in ngrams if ng in self.ngram2idx]

                        training_data.append((center_word, ngram_indices, context_word))

        return training_data

    def train(self, epochs=5, batch_size=512, lr=0.025):
        """训练模型"""
        print("Generating training data...")
        training_data = self.generate_training_data()
        print(f"Training samples: {len(training_data)}")

        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        losses = []

        for epoch in range(epochs):
            epoch_loss = 0
            num_batches = 0

            np.random.shuffle(training_data)

            pbar = tqdm(range(0, len(training_data), batch_size),
                        desc=f"Epoch {epoch+1}/{epochs}")

            for batch_start in pbar:
                batch_end = min(batch_start + batch_size, len(training_data))
                batch = training_data[batch_start:batch_end]

                center_words = []
                ngrams_list = []
                context_words = []

                max_ngram_len = 0
                for center, ngrams, context in batch:
                    center_words.append(center)
                    ngrams_list.append(ngrams)
                    context_words.append(context)
                    max_ngram_len = max(max_ngram_len, len(ngrams))

                if max_ngram_len == 0:
                    max_ngram_len = 1

                padded_ngrams = []
                for ngrams in ngrams_list:
                    if len(ngrams) == 0:
                        padded_ngrams.append([0] * max_ngram_len)
                    else:
                        padded = ngrams + [0] * (max_ngram_len - len(ngrams))
                        padded_ngrams.append(padded)

                neg_samples = np.random.choice(
                    len(self.word2idx),
                    size=(len(batch), self.num_neg),
                    p=self.neg_sampling_probs
                )

                center_words_tensor = torch.tensor(center_words, dtype=torch.long, device=device)
                ngrams_tensor = torch.tensor(padded_ngrams, dtype=torch.long, device=device)
                context_words_tensor = torch.tensor(context_words, dtype=torch.long, device=device)
                neg_samples_tensor = torch.tensor(neg_samples, dtype=torch.long, device=device)

                optimizer.zero_grad()
                loss = self.model(center_words_tensor, ngrams_tensor,
                                  context_words_tensor, neg_samples_tensor)

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            avg_loss = epoch_loss / num_batches
            losses.append(avg_loss)
            print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")

        return losses

    def save_embeddings(self, save_dir):
        """保存词向量"""
        os.makedirs(save_dir, exist_ok=True)

        embeddings = {}
        for word, idx in self.word2idx.items():
            word_idx = torch.tensor([idx], device=device)
            ngrams = self.get_ngrams(word)
            ngram_indices = [self.ngram2idx[ng] for ng in ngrams if ng in self.ngram2idx]

            embedding = self.model.get_word_embedding(word_idx, ngram_indices)
            embeddings[word] = embedding.cpu().detach().numpy().tolist()

        with open(os.path.join(save_dir, 'fasttext_embeddings.json'), 'w', encoding='utf-8') as f:
            json.dump(embeddings, f, ensure_ascii=False, indent=2)

        torch.save(self.model.state_dict(), os.path.join(save_dir, 'fasttext_model.pth'))

        vocab_info = {
            'word2idx': self.word2idx,
            'ngram2idx': self.ngram2idx
        }
        with open(os.path.join(save_dir, 'vocab.json'), 'w', encoding='utf-8') as f:
            json.dump(vocab_info, f, ensure_ascii=False, indent=2)

        print(f"Embeddings saved to {save_dir}")


def plot_training_loss(losses, save_path):
    """绘制训练损失曲线"""
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(losses) + 1), losses, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('FastText Training Loss')
    plt.grid(True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Loss curve saved to {save_path}")


def main():
    data_path = '../data/result-rmrb.txt'
    model_dir = '../model'
    result_dir = '../result'

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    print("=" * 50)
    print("Training FastText Model")
    print("=" * 50)

    trainer = FastTextTrainer(
        data_path=data_path,
        embedding_dim=100,
        window_size=5,
        min_count=5,
        num_neg=5,
        min_n=3,
        max_n=6
    )

    losses = trainer.train(epochs=10, batch_size=512, lr=0.001)

    trainer.save_embeddings(model_dir)

    plot_training_loss(losses, os.path.join(result_dir, 'fasttext_loss.png'))

    with open(os.path.join(result_dir, 'fasttext_training_history.json'), 'w') as f:
        json.dump({'losses': losses}, f, indent=2)

    print("\nFastText training completed!")

    print("\n" + "=" * 50)
    print("Testing word similarities")
    print("=" * 50)

    test_words = ['中国', '人民', '经济', '发展', '建设']
    for word in test_words:
        if word in trainer.word2idx:
            idx = trainer.word2idx[word]
            word_idx = torch.tensor([idx], device=device)
            ngrams = trainer.get_ngrams(word)
            ngram_indices = [trainer.ngram2idx[ng] for ng in ngrams if ng in trainer.ngram2idx]

            embedding = trainer.model.get_word_embedding(word_idx, ngram_indices)
            print(f"{word}: {embedding.shape}")


if __name__ == '__main__':
    main()

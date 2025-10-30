# Week3 NLP 作业 - 2022213670

本项目实现了三个 NLP 任务：

1. FastText 词向量训练
2. Seq2Seq 天津话到普通话翻译（支持 RNN/GRU/LSTM）
3. Character-level 的 POS 和 NER 标注

## 项目结构

```
Week3_2022213670/
├── data/                           # 数据集
│   ├── result-rmrb.txt            # 人民日报语料（POS/NER）
│   └── Tianjin_dataset-main/      # 天津话翻译数据集
├── src/                            # 源代码
│   ├── main.py                    # 主运行脚本
│   ├── 1_fasttext.py              # FastText实现
│   ├── 2_seq2seq_translation.py   # Seq2Seq翻译
│   └── 3_pos_ner.py               # POS和NER标注
├── model/                          # 训练后的模型
└── result/                         # 运行结果
```

## 环境要求

```bash
torch >= 1.8.0
numpy
matplotlib
tqdm
```

## 使用方法

### 方式 1: 运行所有任务

```bash
cd Week3_2022213670/src
python main.py
```

### 方式 2: 单独运行任务

#### 任务 1: FastText 词向量

```bash
cd Week3_2022213670/src
python 1_fasttext.py
```

**输出：**

- `model/fasttext_model.pth` - FastText 模型
- `model/fasttext_embeddings.json` - 词向量文件
- `model/vocab.json` - 词汇表
- `result/fasttext_loss.png` - 训练损失曲线
- `result/fasttext_training_history.json` - 训练历史

#### 任务 2: Seq2Seq 翻译

```bash
cd Week3_2022213670/src
python 2_seq2seq_translation.py
```

**输出：**

- `model/seq2seq_rnn_best.pth` - RNN 最佳模型
- `model/seq2seq_gru_best.pth` - GRU 最佳模型
- `model/seq2seq_lstm_best.pth` - LSTM 最佳模型
- `model/seq2seq_*_final.pth` - 最终模型
- `result/seq2seq_*_loss.png` - 训练损失曲线
- `result/seq2seq_*_history.json` - 训练历史
- `result/seq2seq_*_translations.json` - 翻译示例

#### 任务 3: POS 和 NER 标注

```bash
cd Week3_2022213670/src
python 3_pos_ner.py
```

**输出：**

- `model/pos_best.pth` - POS 最佳模型
- `model/ner_best.pth` - NER 最佳模型
- `model/pos_final.pth` - POS 最终模型
- `model/ner_final.pth` - NER 最终模型
- `result/pos_training_curves.png` - POS 训练曲线
- `result/ner_training_curves.png` - NER 训练曲线
- `result/pos_training_history.json` - POS 训练历史
- `result/ner_training_history.json` - NER 训练历史
- `result/pos_predictions.json` - POS 预测示例
- `result/ner_predictions.json` - NER 预测示例

## 任务详情

### 任务 1: FastText 词向量

**实现细节：**

- 使用 Skip-gram 架构
- 负采样（Negative Sampling）
- Subword 信息（3-6 gram）
- 词向量维度：100
- 窗口大小：5
- 负样本数：5

**特点：**

- 支持 OOV（Out-of-Vocabulary）词汇
- 使用字符级 n-gram 捕捉形态学信息
- 适合中文词向量学习

### 任务 2: Seq2Seq 翻译

**实现细节：**

- 支持三种 RNN 架构：RNN、GRU、LSTM
- 编码器-解码器架构
- 注意力机制（Attention）
- 字符级别处理
- Teacher Forcing 训练策略

**模型参数：**

- 嵌入维度：256
- 隐藏层维度：512
- 层数：2
- Dropout：0.3

### 任务 3: POS 和 NER 标注

**实现细节：**

- BiLSTM-CRF 架构
- 字符级别表示
- 条件随机场（CRF）层
- Viterbi 解码

**模型参数：**

- 字符嵌入维度：100
- 隐藏层维度：256
- BiLSTM 层数：2
- Dropout：0.5

**标注任务：**

1. **POS 标注**: 词性标注（基于人民日报语料的词性标签）
2. **NER 标注**: 命名实体识别（人名、地名、机构名）

## 设备支持

代码自动检测并使用可用的计算设备：

- Apple Silicon (MPS)
- NVIDIA GPU (CUDA)
- CPU

## 数据集说明

### 1. 人民日报语料 (result-rmrb.txt)

- 用于 FastText 词向量训练
- 用于 POS 和 NER 标注训练
- 格式：词/词性

### 2. 天津话翻译数据集 (Tianjin_dataset-main/)

- 包含多个 JSON 文件
- 格式：{"source": "天津话", "translation": "普通话"}
- 来源：俗世奇人等文学作品

## 注意事项

1. 训练时间较长，建议使用 GPU 加速
2. 确保数据集路径正确
3. 模型和结果会自动保存到相应目录
4. 可以根据需要调整超参数

## 作者

学号：2022213670

## 许可证

本项目仅用于学习和研究目的。

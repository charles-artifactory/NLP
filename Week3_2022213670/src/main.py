import sys


def run_fasttext():
    print("\n" + "="*80)
    print("TASK 1: FastText词向量训练")
    print("="*80 + "\n")

    import importlib
    fasttext_module = importlib.import_module('1_fasttext')
    fasttext_module.main()


def run_seq2seq():
    print("\n" + "="*80)
    print("TASK 2: Seq2Seq天津话到普通话翻译")
    print("="*80 + "\n")

    import importlib
    seq2seq_module = importlib.import_module('2_seq2seq_translation')
    seq2seq_module.main()


def run_pos_ner():
    print("\n" + "="*80)
    print("TASK 3: Character-level POS和NER标注")
    print("="*80 + "\n")

    import importlib
    pos_ner_module = importlib.import_module('3_pos_ner')
    pos_ner_module.main()


def main():
    print("="*80)
    print("Week3 NLP作业 - 2022213670")
    print("="*80)
    print("\n任务列表：")
    print("1. FastText词向量训练")
    print("2. Seq2Seq天津话到普通话翻译（RNN/GRU/LSTM）")
    print("3. Character-level POS和NER标注")
    print("\n" + "="*80)

    try:
        run_fasttext()
        run_seq2seq()
        run_pos_ner()

        print("\n" + "="*80)
        print("所有任务已完成！")
        print("="*80)
        print("\n结果保存位置：")
        print("  - 模型: ../model/")
        print("  - 结果: ../result/")
        print("="*80 + "\n")

    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

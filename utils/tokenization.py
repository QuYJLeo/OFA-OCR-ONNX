import os
import numpy as np


class TokenizerZH:
    def __init__(self):
        # 特殊 token
        self.bos_token = '<s>'
        self.eos_token = '</s>'
        self.sep_token = '</s>'
        self.cls_token = '<s>'
        self.unk_token = '<unk>'
        self.pad_token = '<pad>'
        self.mask_token = '<mask>'

        self.vocab = self.load_vocab(os.path.join(os.getcwd(), "../onxw", "vocab.txt"))

        # 构建 ids_to_tokens 映射
        self.ids_to_tokens = {v: k for k, v in self.vocab.items()}

    def load_vocab(self, vocab_file):
        vocab = dict()
        with open(vocab_file, 'r', encoding='utf-8') as reader:
            tokens = reader.readlines()
        for index, token in enumerate(tokens):
            token = token.rstrip('\n')
            vocab[token] = index
        return vocab

    def id_to_token(self, index):
        if index in self.ids_to_tokens:
            return self.ids_to_tokens[index]
        return self.unk_token

    def decode(self, token_ids):
        tokens = [self.id_to_token(tid) for tid in token_ids]
        tokens = [t for t in tokens if
                  t not in {self.bos_token, self.eos_token, self.sep_token, self.cls_token, self.pad_token,
                            self.mask_token}]
        return ''.join(tokens)

    def bdecode(self, batch_token_ids):
        return [self.decode(ids) for ids in batch_token_ids]


if __name__ == '__main__':
    tokenizer = TokenizerZH()

    # 测试解码
    result = tokenizer.bdecode([np.array([1159, 775, 5842, 2])])
    print(result)  # 输出解码结果

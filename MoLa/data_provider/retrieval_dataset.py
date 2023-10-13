import torch
from torch_geometric.data import Dataset
import os
from transformers import BertTokenizer


class RetrievalDataset(Dataset):
    def __init__(self, root, args):
        super(RetrievalDataset, self).__init__(root)
        self.root = root
        self.graph_aug = args.graph_aug
        self.text_max_len = args.text_max_len
        self.graph_name_list = os.listdir(root+'graph/')
        self.graph_name_list.sort()
        self.text_name_list = os.listdir(root+'text/')
        self.text_name_list.sort()
        self.tokenizer = self.extend_tokenizer()

    def __len__(self):
        return len(self.graph_name_list)

    def __getitem__(self, index):
        graph_name, text_name = self.graph_name_list[index], self.text_name_list[index]
        # load and process graph
        graph_path = os.path.join(self.root, 'graph', graph_name)
        data_graph = torch.load(graph_path)

        # load and process text
        text_path = os.path.join(self.root, 'text', text_name)
        text = ''
        count = 0
        for line in open(text_path, 'r', encoding='utf-8'):
            count += 1
            line.strip('\n')
            text += line
            if count > 100:
                break
        text, mask = self.tokenizer_text(text)

        return data_graph, text.squeeze(0), mask.squeeze(0)

    def tokenizer_text(self, text):
        sentence_token = self.tokenizer(text=text,
                                        truncation=True,
                                        padding='max_length',
                                        add_special_tokens=False,
                                        max_length=self.text_max_len,
                                        return_tensors='pt',
                                        return_attention_mask=True)
        input_ids = sentence_token['input_ids']
        attention_mask = sentence_token['attention_mask']
        return input_ids, attention_mask

    def extend_tokenizer(self):
        tokenizer = BertTokenizer.from_pretrained('checkpoints/bert_pretrained/')
        tokenizer.add_special_tokens({'bos_token': '[DEC]'})
        tokenizer.add_special_tokens({'additional_special_tokens': ['[ENC]', '[SMI]']})
        tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]
        tokenizer.smi_token_id = tokenizer.additional_special_tokens_ids[1]
        return tokenizer

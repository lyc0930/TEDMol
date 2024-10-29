import torch
from torch_geometric.data import Dataset
import os

class GINPretrainDataset(Dataset):
    def __init__(self, root, text_max_len):
        super(GINPretrainDataset, self).__init__(root)
        self.root = root
        self.text_max_len = text_max_len
        self.graph_name_list = os.listdir(root+'graph/')
        self.graph_name_list.sort()
        self.text_name_list = os.listdir(root+'text/')
        self.text_name_list.sort()
        self.smiles_name_list = os.listdir(root + 'smiles/')
        self.smiles_name_list.sort()
        self.tokenizer = None

    def __len__(self):
        return len(self.graph_name_list)

    def __getitem__(self, index):
        graph_name, text_name, smiles_name = self.graph_name_list[index], self.text_name_list[index], self.smiles_name_list[index]
        # load and process smiles
        smiles_path = os.path.join(self.root, 'smiles', smiles_name)
        raw_smiles = ''
        for line in open(smiles_path, 'r', encoding='utf-8'):
            line.strip('\n')
            raw_smiles += line
        smiles, smiles_mask = self.tokenizer_text(raw_smiles)

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
            if count > 10:
                break
        text, mask = self.tokenizer_text(text)

        return data_graph, text.squeeze(0), mask.squeeze(0), smiles.squeeze(0), smiles_mask.squeeze(0)

    def tokenizer_text(self, text):
        sentence_token = self.tokenizer(text=text,
                                        truncation=True,
                                        padding='max_length',
                                        add_special_tokens=True,
                                        max_length=self.text_max_len,
                                        return_tensors='pt',
                                        return_attention_mask=True)
        input_ids = sentence_token['input_ids']
        attention_mask = sentence_token['attention_mask']
        return input_ids, attention_mask
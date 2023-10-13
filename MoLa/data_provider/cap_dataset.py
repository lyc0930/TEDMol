from transformers import BertTokenizer
from ogb.utils import smiles2graph
from torch_geometric.data import Data, Dataset
import torch
import os.path as osp
import csv


class CapDataset(Dataset):
    def __init__(self, data_path, split):
        self.data_path = data_path
        self.tokenizer = self.extend_tokenizer()
        self.tokenizer.padding_side = 'left'
        self.cids = []
        self.descriptions = {}
        self.cids_to_smiles = {}
        self.smiles = {}
        self.text_max_len = 128
        self.prompt = 'The molecule\t'

        # load data
        with open(osp.join(data_path, split + '.txt')) as f:
            reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
            for n, line in enumerate(reader):
                self.descriptions[line['CID']] = line['description']
                self.cids_to_smiles[line['CID']] = line['SMILES']
                self.cids.append(line['CID'])

    def __len__(self):
        return len(self.cids)

    def __getitem__(self, idx):
        cid = self.cids[idx]

        smiles = self.cids_to_smiles[cid]
        description = self.descriptions[cid]
        ori_graph = smiles2graph(smiles)
        x = torch.from_numpy(ori_graph['node_feat']).to(torch.int64)
        edge_index = torch.from_numpy(ori_graph['edge_index']).to(torch.int64)
        edge_attr = torch.from_numpy(ori_graph['edge_feat']).to(torch.int64)
        num_nodes = int(ori_graph['num_nodes'])
        graph = Data(x, edge_index, edge_attr, num_nodes=num_nodes)

        smiles_id, smiles_mask = self.tokenizer_text(smiles)
        prompt_id, prompt_mask = self.tokenizer_text(self.prompt)
        description_id, description_mask = self.tokenizer_text(description)

        return graph, smiles_id.squeeze(0), smiles_mask.squeeze(0), prompt_id.squeeze(0), description_id.squeeze(0), description_mask.squeeze(0)

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

    def extend_tokenizer(self):
        tokenizer = BertTokenizer.from_pretrained('checkpoints/bert_pretrained/')
        tokenizer.add_special_tokens({'bos_token': '[DEC]'})
        tokenizer.add_special_tokens({'additional_special_tokens': ['[ENC]', '[SMI]']})
        tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]
        tokenizer.smi_token_id = tokenizer.additional_special_tokens_ids[1]
        return tokenizer

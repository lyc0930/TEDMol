import json

# import pubchempy as pcp
import torch
from ogb.utils import smiles2graph
from torch_geometric.data import Data

import shutil
import requests


def smiles2data(smiles):

    graph = smiles2graph(smiles)

    x = torch.from_numpy(graph['node_feat'])
    edge_index = torch.from_numpy(graph['edge_index'], )
    edge_attr = torch.from_numpy(graph['edge_feat'])
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return data


# def cid2smiles(cid):
#     try:
#         compound = pcp.Compound.from_cid(cid)
#         return compound.isomeric_smiles
#     except:
#         try:
#             url = f'https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON/'
#
#             request = requests.get(url)
#             proper_json = json.loads(request.text)
#             canonical_smiles = None
#             isomeric_smiles = None
#             for item in proper_json['Record']['Section']:
#                 if item['TOCHeading'] == 'Names and Identifiers':
#                     for item in item['Section']:
#                         if item['TOCHeading'] == 'Computed Descriptors':
#                             for item in item['Section']:
#                                 if item['TOCHeading'] == 'Canonical SMILES':
#                                     canonical_smiles = item['Information'][0]['Value']['StringWithMarkup'][0]['String']
#                                 if item['TOCHeading'] == 'Isomeric SMILES':
#                                     isomeric_smiles = item['Information'][0]['Value']['StringWithMarkup'][0]['String']
#                             break
#                     break
#             if isomeric_smiles:
#                 return isomeric_smiles
#             else:
#                 return canonical_smiles
#         except:
#             raise Exception(f'Cannot get smiles for cid {cid}')
#
#
# def stamp(cid):
#     url = f'https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON?heading=Create+Date'
#     # url = f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/dates/JSON'
#     req = requests.get(url)
#     raw_json_data = json.loads(req.text)
#     date = raw_json_data["Record"]["Section"][0]["Section"][0]["Information"][0]["Value"]["DateISO8601"][0]
#     # creation_date = raw_json_data["InformationList"]["Information"][0]["CreationDate"]
#     # date=f"{creation_date['Year']}-{creation_date['Month']:02}-{creation_date['Day']:02}"
#     return date


def pair_copy(path, new_path, cid):
    shutil.copy(f'{path}/graph/graph_{cid}.pt', f'{new_path}/graph/graph_{cid}.pt')
    shutil.copy(f'{path}/text/text_{cid}.txt', f'{new_path}/text/text_{cid}.txt')


def tuple_copy(path, new_path, cid):
    shutil.copy(f'{path}/all/graph/graph_{cid}.pt', f'{new_path}/graph/graph_{cid}.pt')
    shutil.copy(f'{path}/all/text/text_{cid}.txt', f'{new_path}/text/text_{cid}.txt')
    shutil.copy(f'{path}/all/smiles/smiles_{cid}.txt', f'{new_path}/smiles/smiles_{cid}.txt')

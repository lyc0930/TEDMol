import concurrent.futures
import os
import random
import shutil
import sys
import yaml
from datetime import datetime
from tqdm import tqdm
from utils import pair_copy, tuple_copy


def prune_short_description(path, threshold):
    print(f'Prune short description with threshold {threshold}...')
    total = 0
    os.makedirs(f'{path}-{threshold}/graph')
    os.makedirs(f'{path}-{threshold}/text')
    _, _, file_list = next(os.walk(os.path.join(path, 'text')))
    for file_name in tqdm(file_list):
        try:
            with open(os.path.join(path, 'text', file_name), 'r', encoding='utf-8') as f:
                count = len([word for line in f for word in line.split()])
                if count >= threshold:
                    cid = file_name.split('_')[1].split('.')[0]
                    tuple_copy(path, f'{path}-{threshold}', cid)
                    total += 1
        except:
            continue
    return total


def split(path, train_ratio=0.8, valid_ratio=0.1, test_ratio=0.1, method='random', index=None):
    os.makedirs(os.path.join(path, 'pretrain/graph'))
    os.makedirs(os.path.join(path, 'pretrain/text'))
    os.makedirs(os.path.join(path, 'pretrain/smiles'))
    os.makedirs(os.path.join(path, 'train/graph'))
    os.makedirs(os.path.join(path, 'train/text'))
    os.makedirs(os.path.join(path, 'train/smiles'))
    os.makedirs(os.path.join(path, 'valid/graph'))
    os.makedirs(os.path.join(path, 'valid/text'))
    os.makedirs(os.path.join(path, 'valid/smiles'))
    os.makedirs(os.path.join(path, 'test/graph'))
    os.makedirs(os.path.join(path, 'test/text'))
    os.makedirs(os.path.join(path, 'test/smiles'))
    # print(f'Split dataset into train: {train_ratio}, valid: {valid_ratio}, test: {test_ratio} with method {method}...')
    _, _, file_list = next(os.walk(os.path.join(path, 'text')))
    cids = [file_name.split('_')[1].split('.')[0] for file_name in file_list]

    total = len(cids)
    # train_number = int(total * train_ratio)
    # valid_number = int(total * valid_ratio)
    train_number = total - 20000
    valid_number = 10000

    if method == 'random':
        random.shuffle(cids)
    elif method.startswith('date'):
        with open("./PugView/History/date.yaml", 'r', encoding='utf-8') as f:
            date_dict = yaml.load(f, Loader=yaml.BaseLoader)
            cids = sorted(cids, key = lambda cid : date_dict[cid])

        print("The created date of compound in train set starts from", date_dict[cids[0]], "to", date_dict[cids[train_number - 1]])

        if method == 'date+':
            cids[train_number:] = random.sample(cids[train_number:], len(cids[train_number:]))
            print("The created date of compound in valid/test set starts from", date_dict[cids[train_number]], "to", date_dict[cids[-1]])
        elif method == 'date':
            print("The created date of compound in valid set starts from", date_dict[cids[train_number]], "to", date_dict[cids[train_number + valid_number - 1]])
            print("The created date of compound in test set starts from", date_dict[cids[train_number + valid_number]], "to", date_dict[cids[-1]])

    train_cids = cids[:train_number]
    valid_cids = cids[train_number:train_number + valid_number]
    test_cids = cids[train_number + valid_number:]

    if index!=None:
        with open(f"{path}/split/train_cids", "w", encoding='utf-8') as f:
            f.write("\n".join(train_cids))
        with open(f"{path}/split/valid_cids", "w", encoding='utf-8') as f:
            f.write("\n".join(valid_cids))
        with open(f"{path}/split/test_cids", "w", encoding='utf-8') as f:
            f.write("\n".join(test_cids))

    if index!="only":
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            list(tqdm(executor.map(lambda cid : tuple_copy(path, f'{path}/split/train/', cid), train_cids), total=len(train_cids)))
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            list(tqdm(executor.map(lambda cid : tuple_copy(path, f'{path}/split/valid/', cid), valid_cids), total=len(valid_cids)))
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            list(tqdm(executor.map(lambda cid : tuple_copy(path, f'{path}/split/test/', cid), test_cids), total=len(test_cids)))


def split_with_index(path):
    os.makedirs(os.path.join(path, 'split/train/graph'))
    os.makedirs(os.path.join(path, 'split/train/text'))
    os.makedirs(os.path.join(path, 'split/train/smiles'))
    os.makedirs(os.path.join(path, 'split/valid/graph'))
    os.makedirs(os.path.join(path, 'split/valid/text'))
    os.makedirs(os.path.join(path, 'split/valid/smiles'))
    os.makedirs(os.path.join(path, 'split/test/graph'))
    os.makedirs(os.path.join(path, 'split/test/text'))
    os.makedirs(os.path.join(path, 'split/test/smiles'))

    with open(f"{path}/split/train_cids", "r", encoding='utf-8') as f:
        train_cids = f.read().split('\n')
    with open(f"{path}/split/test_cids", "r", encoding='utf-8') as f:
        test_cids = f.read().split('\n')
    with open(f"{path}/split/valid_cids", "r", encoding='utf-8') as f:
        valid_cids = f.read().split('\n')

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            list(tqdm(executor.map(lambda cid : tuple_copy(path, f'{path}/split/train/', cid), train_cids), total=len(train_cids)))
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            list(tqdm(executor.map(lambda cid : tuple_copy(path, f'{path}/split/valid/', cid), valid_cids), total=len(valid_cids)))
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            list(tqdm(executor.map(lambda cid : tuple_copy(path, f'{path}/split/test/', cid), test_cids), total=len(test_cids)))


def main():
    path = './GraphTextPretrain/our_data/PubChemDataset_v4'

    # os.makedirs(os.path.join(path, 'pretrain/graph'))
    # os.makedirs(os.path.join(path, 'pretrain/text'))
    # os.makedirs(os.path.join(path, 'pretrain/smiles'))
    # os.makedirs(os.path.join(path, 'train/graph'))
    # os.makedirs(os.path.join(path, 'train/text'))
    # os.makedirs(os.path.join(path, 'train/smiles'))
    # os.makedirs(os.path.join(path, 'valid/graph'))
    # os.makedirs(os.path.join(path, 'valid/text'))
    # os.makedirs(os.path.join(path, 'valid/smiles'))
    # os.makedirs(os.path.join(path, 'test/graph'))
    # os.makedirs(os.path.join(path, 'test/text'))
    # os.makedirs(os.path.join(path, 'test/smiles'))


    train_num = 12000
    valid_num = 1000
    test_num = 2000

    word_threshhold = 20

    _, _, file_list = next(os.walk(f'{path}/all/text'))
    random.shuffle(file_list)
    for file_name in tqdm(file_list):
        # try:
        #     with open(os.path.join(path, 'all/text', file_name), 'r', encoding='utf-8') as f:
        #         count = len([word for line in f for word in line.split()])
        #         cid = file_name.split('_')[1].split('.')[0]
        #         if count >= word_threshhold:
        #             if train_num > 0:
        #                 tuple_copy(path, f'{path}/train/', cid)
        #                 train_num = train_num - 1
        #             elif valid_num > 0:
        #                 tuple_copy(path, f'{path}/valid/', cid)
        #                 valid_num = valid_num - 1
        #             elif test_num > 0:
        #                 tuple_copy(path, f'{path}/test/', cid)
        #                 test_num = test_num - 1
        #             else:
        #                 tuple_copy(path, f'{path}/pretrain/', cid)
        #         else:
        #             tuple_copy(path, f'{path}/pretrain/', cid)
        # except:
        #     print("Error Alert")
        #     continue
        with open(os.path.join(path, 'all/text', file_name), 'r', encoding='utf-8') as f:
            count = len([word for line in f for word in line.split()])
            cid = file_name.split('_')[1].split('.')[0]
            if count >= word_threshhold:
                if train_num > 0:
                    tuple_copy(path, f'{path}/train/', cid)
                    train_num = train_num - 1
                elif valid_num > 0:
                    tuple_copy(path, f'{path}/valid/', cid)
                    valid_num = valid_num - 1
                elif test_num > 0:
                    tuple_copy(path, f'{path}/test/', cid)
                    test_num = test_num - 1
                else:
                    tuple_copy(path, f'{path}/pretrain/', cid)
            else:
                tuple_copy(path, f'{path}/pretrain/', cid)


if '__main__' == __name__:
    main()

import torch


def compute_mean_mad(dataloaders, properties, dataset_name):
    if dataset_name == 'qm9':
        return compute_mean_mad_from_dataloader(dataloaders['train'], properties)
    elif dataset_name == 'qm9_second_half' or dataset_name == 'qm9_second_half':
        return compute_mean_mad_from_dataloader(dataloaders['valid'], properties)
    else:
        raise Exception('Wrong dataset name')


def compute_mean_mad_from_dataloader(dataloader, properties):
    property_norms = {}
    for property_key in properties:
        values = dataloader.dataset.data[property_key]
        mean = torch.mean(values)
        ma = torch.abs(values - mean)
        mad = torch.mean(ma)
        property_norms[property_key] = {}
        property_norms[property_key]['mean'] = mean
        property_norms[property_key]['mad'] = mad
    return property_norms

edges_dic = {}
def get_adj_matrix(n_nodes, batch_size, device):
    if n_nodes in edges_dic:
        edges_dic_b = edges_dic[n_nodes]
        if batch_size in edges_dic_b:
            return edges_dic_b[batch_size]
        else:
            # get edges for a single sample
            rows, cols = [], []
            for batch_idx in range(batch_size):
                for i in range(n_nodes):
                    for j in range(n_nodes):
                        rows.append(i + batch_idx*n_nodes)
                        cols.append(j + batch_idx*n_nodes)

    else:
        edges_dic[n_nodes] = {}
        return get_adj_matrix(n_nodes, batch_size, device)


    edges = [torch.LongTensor(rows).to(device), torch.LongTensor(cols).to(device)]
    return edges

def preprocess_input(one_hot, charges, charge_power, charge_scale, device):
    charge_tensor = (charges.unsqueeze(-1) / charge_scale).pow(
        torch.arange(charge_power + 1., device=device, dtype=torch.float32))
    charge_tensor = charge_tensor.view(charges.shape + (1, charge_power + 1))
    atom_scalars = (one_hot.unsqueeze(-1) * charge_tensor).view(charges.shape[:2] + (-1,))
    return atom_scalars
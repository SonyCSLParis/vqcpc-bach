import importlib
import os
import pickle as pkl
from itertools import product

import torch

from VQCPCB.getters import get_encoder


def build_index_cluster_maps(dataloader_generator, config, overfitted_encoders):
    print('Building cluster to index mappings...')
    # Get dataloaders. Batch_size does not matter, it does not modify the indexing
    #  However, the train, val, test split DOES matter, the exact same as during training need to be use
    neg_gens = dataloader_generator.dataset_negative.data_loaders(
        batch_size=1,
        num_workers=0,
        indexed_dataloaders=True
    )
    pos_gens = dataloader_generator.dataset.data_loaders(
        batch_size=1,
        num_workers=0,
        indexed_dataloaders=True
    )

    index2cluster = process_positive_generator(config,
                                               dataloader_generator=dataloader_generator,
                                               pos_gens=pos_gens,
                                               early_stopped_encoders=not overfitted_encoders)
    cluster2index = process_negative_generator(config,
                                               dataloader_generator=dataloader_generator,
                                               neg_gens=neg_gens,
                                               early_stopped_encoders=not overfitted_encoders)

    return dict(index2cluster=index2cluster,
                cluster2index=cluster2index)


def process_negative_generator(config, dataloader_generator, neg_gens, early_stopped_encoders):
    cluster2index_product = {
        'train': {},
        'val': {},
        'test': {}
    }
    cluster2index_hierarchy = {
        'train': {},
        'val': {},
        'test': {}
    }

    encoders_config = config['dataloader_generator_kwargs']['previous_encoders']
    mixing_method = config['dataloader_generator_kwargs']['mixing_method']

    #  Deal with empty list of encoders (i.e. step 0 in the progressive learning)
    if len(encoders_config) == 0:
        cluster2index_product = {
            'train': {},
            'val': {},
            'test': {}
        }
        neg_gen_train, neg_gen_val, neg_gen_test = neg_gens
        for split_name, gen in [('train', neg_gen_train), ('val', neg_gen_val), ('test', neg_gen_test)]:
            neg_indices = list(range(len(gen)))
            cluster2index_product[split_name] = {(0,): neg_indices}
        return cluster2index_product

    for hierarchy_level, config_encoder_path in enumerate(encoders_config):
        # ==== Load encoder ====
        config_encoder_module_name = os.path.splitext(config_encoder_path)[0].replace('/', '.')
        config_encoder = importlib.import_module(config_encoder_module_name).config
        config_encoder['quantizer_kwargs']['initialize'] = False
        model_dir_encoder = os.path.dirname(config_encoder_path)
        assert config['dataset'] == config_encoder['dataset']
        encoder = get_encoder(model_dir=model_dir_encoder,
                              dataloader_generator=dataloader_generator,
                              config=config_encoder
                              )
        encoder.load(early_stopped=early_stopped_encoders)
        encoder.to('cuda')
        encoder.eval()

        # ==== Cluster to index on negatives ====
        cluster2index_path = f'{model_dir_encoder}/cluster2index_negative.pkl'
        if os.path.isfile(cluster2index_path):
            with open(cluster2index_path, 'rb') as ff:
                cluster2index = pkl.load(ff)
        else:
            cluster2index = {
                'train': {},
                'val': {},
                'test': {},
            }
            neg_gen_train, neg_gen_val, neg_gen_test = neg_gens
            # get mapping for each split
            for split_name, gen in [('train', neg_gen_train), ('val', neg_gen_val), ('test', neg_gen_test)]:
                with torch.no_grad():
                    for n_gen in gen:
                        n, index = n_gen
                        n = n[0].transpose(1, 2)
                        # n: (batch_size, num_events_per_block, num_channels)
                        index = int(index)
                        _, encoding_indices, _ = encoder(n)
                        cluster_index = int(encoding_indices[0].detach().cpu().numpy())
                        if cluster_index in cluster2index[split_name].keys():
                            cluster2index[split_name][cluster_index].append(index)
                        else:
                            cluster2index[split_name][cluster_index] = [index]
            with open(cluster2index_path, 'wb') as ff:
                pkl.dump(cluster2index, ff)

        for split_name, value in cluster2index.items():
            cluster2index_hierarchy[split_name][hierarchy_level] = value

    for split_name, cluster2index_hierarchy_split in cluster2index_hierarchy.items():
        clusters_sizes = [list(e.keys()) for e in cluster2index_hierarchy_split.values()]
        product_clusters = product(*clusters_sizes)
        for cluster_tuple in product_clusters:
            # Compute set of negative indices associated to each product of cluster
            negative_indices = set()
            for hierarchy_level in range(len(cluster_tuple)):
                hierarchy_cluster = cluster_tuple[hierarchy_level]
                hierarchy_indices = set(cluster2index_hierarchy_split[hierarchy_level][hierarchy_cluster])
                if mixing_method == 'intersection':
                    if len(negative_indices) == 0:
                        negative_indices = hierarchy_indices
                    else:
                        negative_indices = negative_indices.intersection(hierarchy_indices)
                elif mixing_method == 'union':
                    negative_indices = negative_indices.union(hierarchy_indices)
                else:
                    raise ValueError(f'{mixing_method} is not a valid mixing method for hierarchies of codes')

            # Add product of cluster -> indices in dict
            if cluster_tuple in cluster2index_product[split_name].keys():
                assert cluster2index_product[split_name][cluster_tuple] == negative_indices
                continue
            else:
                cluster2index_product[split_name][cluster_tuple] = list(negative_indices)

    return cluster2index_product


def process_positive_generator(config, dataloader_generator, pos_gens, early_stopped_encoders):
    index2cluster_product = {
        'train': {},
        'val': {},
        'test': {}
    }
    index2cluster_hierarchy = {
        'train': {},
        'val': {},
        'test': {}
    }

    encoders_config = config['dataloader_generator_kwargs']['previous_encoders']

    num_blocks_right = config['dataloader_generator_kwargs']['num_blocks_right']
    num_blocks_left = config['dataloader_generator_kwargs']['num_blocks_left']
    positions = range(num_blocks_left, num_blocks_left + num_blocks_right)

    #  Deal with empty list of encoders (i.e. step 0 in the progressive learning)
    if len(encoders_config) == 0:
        index2cluster_product = {
            'train': {},
            'val': {},
            'test': {}
        }
        pos_gen_train, pos_gen_val, pos_gen_test = pos_gens
        for split_name, gen in [('train', pos_gen_train), ('val', pos_gen_val), ('test', pos_gen_test)]:
            gen_len = len(gen)
            for index, position in product(range(gen_len), positions):
                index2cluster_product[split_name][(index, position)] = (0,)
            # for p_gen in gen:
            #     p, index = p_gen
            #     #  p: (batch_size, num_events_per_block, num_channels)
            #     index = int(index)
            #     for position in positions:
            #         index2cluster_product[split_name][(index, position)] = (0,)
        return index2cluster_product

    for hierarchy_level, config_encoder_path in enumerate(encoders_config):
        # ==== Load encoder ====
        config_encoder_module_name = os.path.splitext(config_encoder_path)[0].replace('/', '.')
        config_encoder = importlib.import_module(config_encoder_module_name).config
        config_encoder['quantizer_kwargs']['initialize'] = False
        model_dir_encoder = os.path.dirname(config_encoder_path)
        assert config['dataset'] == config_encoder['dataset']
        encoder = get_encoder(model_dir=model_dir_encoder,
                              dataloader_generator=dataloader_generator,
                              config=config_encoder
                              )
        encoder.load(early_stopped=early_stopped_encoders)
        encoder.to('cuda')
        encoder.eval()

        # ==== Index to cluster on positives ====
        index2cluster_path = f'{model_dir_encoder}/index2cluster_positive.pkl'
        if os.path.isfile(index2cluster_path):
            with open(index2cluster_path, 'rb') as ff:
                index2cluster = pkl.load(ff)
        else:
            index2cluster = {
                'train': {},
                'val': {},
                'test': {}
            }
            pos_gen_train, pos_gen_val, pos_gen_test = pos_gens
            for split_name, gen in [('train', pos_gen_train), ('val', pos_gen_val), ('test', pos_gen_test)]:
                with torch.no_grad():
                    for p_gen in gen:
                        p, index = p_gen
                        p = p[0].transpose(1, 2)
                        #  p: (batch_size, num_events_per_block, num_channels)
                        index = int(index)
                        _, encoding_indices, _ = encoder(p)
                        index2cluster[split_name][index] = encoding_indices[0].detach().cpu().numpy()
            with open(index2cluster_path, 'wb') as ff:
                pkl.dump(index2cluster, ff)

        for split_name, value in index2cluster.items():
            index2cluster_hierarchy[split_name][hierarchy_level] = value

    for split_name, index2cluster_hierarchy_split in index2cluster_hierarchy.items():
        num_positive_indices = None
        for level, index2cluster in index2cluster_hierarchy_split.items():
            if num_positive_indices is None:
                num_positive_indices = len(index2cluster)
            else:
                assert num_positive_indices == len(index2cluster)

        #  These need to be recomputed for each hierarchy of codes, so we don't pickle.save them somewhere
        for positive_index in range(num_positive_indices):
            for position in positions:
                # not efficient but easier to write :)
                cluster_tuple = []
                for _, index2cluster in index2cluster_hierarchy_split.items():
                    cluster_tuple.append(index2cluster[positive_index][position])
                index2cluster_product[split_name][(positive_index, position)] = tuple(cluster_tuple)

    return index2cluster_product

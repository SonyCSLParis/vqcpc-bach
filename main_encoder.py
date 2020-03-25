"""
@author: Gaetan Hadjeres
"""
import shutil
from datetime import datetime
import importlib
import os

import click
import torch

from VQCPCB.encoder import EncoderTrainer
from VQCPCB.getters import get_dataloader_generator, get_encoder, get_encoder_trainer


@click.command()
@click.option('-t', '--train', is_flag=True)
@click.option('-l', '--load', is_flag=True)
@click.option('-c', '--config', type=click.Path(exists=True))
@click.option('--num_workers', type=int, default=0)
def main(train,
         load,
         num_workers,
         config,
         ):
    # Use all gpus available
    gpu_ids = [int(gpu) for gpu in range(torch.cuda.device_count())]
    print(f'Using GPUs {gpu_ids}')
    if len(gpu_ids) == 0:
        device = 'cpu'
    else:
        device = 'cuda'

    # Load config
    config_path = config
    config_module_name = os.path.splitext(config)[0].replace('/', '.')
    config = importlib.import_module(config_module_name).config
    # compute time stamp
    if config['timestamp'] is not None:
        timestamp = config['timestamp']
    else:
        timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        config['timestamp'] = timestamp
    if load:
        model_dir = os.path.dirname(config_path)
    else:
        model_dir = f'models/{config["savename"]}_{timestamp}'
    # Add dynamic parameters to corresponding dicts
    config['quantizer_kwargs']['initialize'] = not load

    dataloader_generator = get_dataloader_generator(
        dataset=config['dataset'],
        training_method=config['training_method'],
        dataloader_generator_kwargs=config['dataloader_generator_kwargs'],
    )

    # Append previous encoders
    encoder = get_encoder(model_dir=model_dir,
                          dataloader_generator=dataloader_generator,
                          config=config
                          )

    encoder_trainer: EncoderTrainer = get_encoder_trainer(
        model_dir=model_dir,
        dataloader_generator=dataloader_generator,
        training_method=config['training_method'],
        encoder=encoder,
        auxiliary_networks_kwargs=config['auxiliary_networks_kwargs'],
    )

    if load:
        encoder_trainer.load(early_stopped=False)

    encoder_trainer.to(device)

    if train:
        # Copy .py config file in the save directory before training
        if not load:
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            shutil.copy(config_path, f'{model_dir}/config.py')
        encoder_trainer.train_model(
            batch_size=config['batch_size'],
            device=device,
            num_batches=config['num_batches'],
            num_epochs=config['num_epochs'],
            lr=config['lr'],
            corrupt_labels=config['quantizer_regularization']['corrupt_labels'],
            plot=True,
            num_workers=num_workers,
        )

    # Generate clusters
    dataloader_generator_clusters = get_dataloader_generator(
        dataset=config['dataset'],
        training_method='decoder',
        dataloader_generator_kwargs=config['dataloader_generator_kwargs']
    )

    if hasattr(encoder, 'encode_score') and config['dataset'] == 'harpsichord':
        scores = [
            'classic_piano_dataset/beethoven_hammerklavier_1_format0.mid',
            'classic_piano_dataset/alb_esp1_format0.mid',
            'ecomp_piano_dataset/Bach03.mid'
        ]
        db_dir = f'{os.path.expanduser("~")}/Data/databases/Piano'
        num_block_to_write = 50
        write_dir = f'{encoder.model_dir}/encoded_scores/'
        if not os.path.isdir(write_dir):
            os.mkdir(write_dir)
        for score in scores:
            abs_path = f'{db_dir}/{score}'
            codes = encoder.encode_score(dataloader_generator_clusters, abs_path, num_blocks=num_block_to_write)
            write_name_midi = re.sub('/', '__', score)
            write_name_txt = re.sub('.mid', '.txt', write_name_midi)
            save_path = f'{write_dir}/{write_name_txt}'
            codes_str = [str(e) for e in codes]
            codes_str = ' - '.join(codes_str)
            with open(save_path, 'w') as ff:
                ff.write(codes_str)
            shutil.copyfile(abs_path, f'{write_dir}/{write_name_midi}')

    num_batches_clusters = 512
    encoder.plot_clusters(dataloader_generator_clusters, split='train', num_batches=num_batches_clusters)
    encoder.plot_clusters(dataloader_generator_clusters, split='val')

    if hasattr(encoder, 'show_NN_clusters'):
        encoder.show_NN_clusters()

    if hasattr(encoder, 'scatterplot_cluster_3D'):
        if encoder.quantizer.codebook_dim == 3:
            encoder.scatterplot_clusters_3D()


if __name__ == '__main__':
    main()

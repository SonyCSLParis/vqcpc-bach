"""
@author: Gaetan Hadjeres
"""
import importlib
import os
import shutil
from datetime import datetime
import numpy as np
import click
import torch

from VQCPCB.autoencoder import Autoencoder
from VQCPCB.data.data_processor import DataProcessor
from VQCPCB.encoder import Encoder
from VQCPCB.getters import get_dataloader_generator, get_data_processor, get_downscaler, get_upscaler, get_decoder
from VQCPCB.quantizer.vector_quantizer import ProductVectorQuantizer


def get_quantizer():
    pass


@click.command()
@click.option('-t', '--train', is_flag=True)
@click.option('-l', '--load', is_flag=True)
@click.option('-o', '--overfitted', is_flag=True,
              help='Load over-fitted weights for the decoder instead of early-stopped.'
                   'Only used with -l')
@click.option('-c', '--config', type=click.Path(exists=True))
@click.option('-r', '--reharmonization', is_flag=True)
@click.option('--code_juxtaposition', is_flag=True)
@click.option('-n', '--num_workers', type=int, default=0)
def main(train,
         load,
         overfitted,
         config,
         reharmonization,
         code_juxtaposition,
         num_workers
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
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        config['timestamp'] = timestamp

    if load:
        model_dir = os.path.dirname(config_path)
    else:
        model_dir = f'models/{config["savename"]}_{timestamp}'

    config['quantizer_kwargs']['initialize'] = not load

    # === Autoencoder ====
    dataloader_generator = get_dataloader_generator(
        training_method=config['training_method'],
        dataloader_generator_kwargs=config['dataloader_generator_kwargs']
    )

    data_processor: DataProcessor = get_data_processor(
        dataloader_generator=dataloader_generator,
        data_processor_kwargs=config['data_processor_kwargs']
    )

    downscaler_kwargs = config['downscaler_kwargs']
    quantizer_kwargs = config['quantizer_kwargs']
    downscaler_kwargs['input_dim'] = data_processor.embedding_size
    downscaler_kwargs['output_dim'] = quantizer_kwargs['codebook_dim']
    downscaler_kwargs['num_tokens'] = data_processor.num_tokens
    downscaler_kwargs['num_channels'] = data_processor.num_channels
    downscaler = get_downscaler(
        downscaler_type='lstm_downscaler',
        downscaler_kwargs=downscaler_kwargs
    )

    quantizer = ProductVectorQuantizer(
        codebook_size=quantizer_kwargs['codebook_size'],
        codebook_dim=quantizer_kwargs['codebook_dim'],
        initialize=quantizer_kwargs['initialize'],
        squared_l2_norm=quantizer_kwargs['squared_l2_norm'],
        use_batch_norm=quantizer_kwargs['use_batch_norm'],
        commitment_cost=quantizer_kwargs['commitment_cost']
    )

    upscaler = get_upscaler(
        upscaler_type=None,
        upscaler_kwargs={}
    )

    encoder = Encoder(
        model_dir=model_dir,
        data_processor=data_processor,
        downscaler=downscaler,
        quantizer=quantizer,
        upscaler=upscaler
    )

    decoder_kwargs = config['decoder_kwargs']
    num_channels_decoder = data_processor.num_channels
    num_events_decoder = data_processor.num_events
    num_channels_encoder = 1
    num_events_encoder = int((num_events_decoder * num_channels_decoder) // \
                             (np.prod(encoder.downscaler.downscale_factors) *
                              num_channels_encoder)
                             )
    decoder = get_decoder(
        model_dir=model_dir,
        dataloader_generator=dataloader_generator,
        data_processor=data_processor,
        encoder=encoder,
        freeze_encoder=False,
        decoder_type='transformer_relative_diagonal',
        decoder_kwargs=decoder_kwargs,
        num_channels_decoder=num_channels_decoder,
        num_events_decoder=num_events_decoder,
        num_channels_encoder=num_channels_encoder,
        num_events_encoder=num_events_encoder,
        re_embed_source=False
    )
    autoencoder = Autoencoder(
        model_dir=model_dir,
        dataloader_generator=dataloader_generator,
        data_processor=data_processor,
        encoder=encoder,
        decoder=decoder
    )

    if load:
        if overfitted:
            autoencoder.load(early_stopped=False, device=device)
        else:
            autoencoder.load(early_stopped=True, device=device)
        autoencoder.to(device)

    if train:
        # Copy .py config file in the save directory before training
        if not load:
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            shutil.copy(config_path, f'{model_dir}/config.py')
        autoencoder.to(device)
        autoencoder.train_model(
            batch_size=config['batch_size'],
            num_batches=config['num_batches'],
            num_epochs=config['num_epochs'],
            lr=config['lr'],
            schedule_lr=config['schedule_lr'],
            plot=True,
            num_workers=num_workers
        )

    num_examples = 3
    for _ in range(num_examples):
        if code_juxtaposition:
            scores = autoencoder.generate(
                temperature=1.0,
                top_p=0.8,
                top_k=0,
                batch_size=3,
                seed_set='val',
                plot_attentions=False,
                code_juxtaposition=True
            )

        scores = autoencoder.generate(temperature=1.0,
                                      top_p=0.95,
                                      top_k=0,
                                      batch_size=3,
                                      seed_set='val',
                                      plot_attentions=False,
                                      code_juxtaposition=False)
        # for score in scores:
        #     score.show()

    if reharmonization:
        scores = autoencoder.generate_reharmonisation(
            temperature=0.9,
            top_p=0.95,
            top_k=0,
            num_reharmonisations=3)
    # for score in scores:
    #     score.show()


if __name__ == '__main__':
    main()

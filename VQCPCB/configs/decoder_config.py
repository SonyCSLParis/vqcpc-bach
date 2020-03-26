from pathlib import Path

config = {
    'config_encoder':              'models/encoder_sameSeq_config_2020-03-26_15-20-28/config.py',
    'training_method':             'decoder',
    'dataset':                     'bach',

    # --- Dataloader ---
    'dataloader_generator_kwargs': dict(
        sequences_size=12
    ),  # Can be different from the encoder's data loader

    # --- DataProcessor ---
    'data_processor_type':         'bach',  # can be used to filter out some channels
    'data_processor_kwargs':       dict(
        embedding_size=32
    ),  # Can be different from the encoder's data processor

    # --- Decoder ---
    # 'transformer' or 'transformer_custom' or 'weak_transformer' or 'transformer_relative'
    'decoder_type':                'transformer_relative',
    'decoder_kwargs':              dict(
        d_model=512,
        n_head=4,
        num_encoder_layers=3,
        num_decoder_layers=3,
        dim_feedforward=1024,
        positional_embedding_size=8,
        dropout=0.1,
    ),
    # ======== Training ========
    'lr':                          1e-4,
    'batch_size':                  64,
    'num_batches':                 None,
    'num_epochs':                  2000,

    # ======== model ID ========
    'timestamp':                   None,
    'savename':                    Path(__file__).stem,
}

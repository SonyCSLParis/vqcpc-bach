from pathlib import Path


config = {
    'config_encoder':              'models/encoder_sameSeq_config_2020-03-27_11-34-57/config.py',
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
    'decoder_type':                'transformer_relative',
    # 'transformer', 'transformer_relative' or 'transformer_relative_diagonal'
    # transformer_relative_diagonal = cross-attention mask is the identity
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
    'batch_size':                  2,
    'num_batches':                 4,
    'num_epochs':                  3,

    # ======== model ID ========
    'timestamp':                   None,
    'savename':                    Path(__file__).stem,
}

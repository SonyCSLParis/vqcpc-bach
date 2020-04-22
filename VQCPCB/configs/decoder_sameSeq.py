from pathlib import Path


config = {
    'config_encoder':              'models/encoder_sameSeq_2020-04-22_18-06-21/config.py',
    'training_method':             'decoder',

    # --- Dataloader ---
    'dataloader_generator_kwargs': dict(
        sequences_size=24
    ),  # Can be different from the encoder's data loader

    # --- DataProcessor ---
    'data_processor_kwargs':       dict(
        embedding_size=32
    ),  # Can be different from the encoder's data processor

    # --- Decoder ---
    'decoder_type':                'transformer_relative_diagonal',
    # See get_decoder in VQCPCB/getters.py to see the different types of transformers
    'decoder_kwargs':              dict(
        d_model=512,
        n_head=8,
        num_encoder_layers=3,
        num_decoder_layers=3,
        dim_feedforward=1024,
        positional_embedding_size=8,
        dropout=0.2,
    ),
    # ======== Training ========
    'lr':                          1e-4,
    'schedule_lr':                 True,
    'batch_size':                  4,
    'num_batches':                 2,
    'num_epochs':                  20000,

    # ======== model ID ========
    'timestamp':                   None,
    'savename':                    Path(__file__).stem,
}


from pathlib import Path


subdivision = 4
num_beats = 1
num_voices = 4
num_tokens_per_block = num_beats * subdivision * num_voices
sequences_size = 24

config = {
    'training_method':             'autoencoder',

    # --- Dataloader ---
    'dataloader_generator_kwargs': dict(
        sequences_size=24
    ),  # Can be different from the encoder's data loader

    # --- DataProcessor ---
    'data_processor_kwargs':       dict(
        embedding_size=32
    ),  # Can be different from the encoder's data processor

    # --- Encoder ---
    # 'downscaler_type': 'lstm_downscaler',
    'downscaler_kwargs': dict(
        # DCPC uses a Transformer
        downscale_factors=[num_tokens_per_block],
        hidden_size=512,
        num_layers=2,
        dropout=0.1,
        bidirectional=True
    ),
    # --- Quantizer ---
    # 'quantizer_type': 'commitment',
    'quantizer_kwargs': dict(
        codebook_size=16,
        codebook_dim=32,
        commitment_cost=0.25,
        use_batch_norm=False,
        squared_l2_norm=True
        # add corrupt indices
    ),
    # --- Upscaler ---
    # 'upscaler_type': None,  # mlp_upscaler
    # 'upscaler_kwargs': dict(
    #     output_dim=32,
    #     hidden_size=512,
    #     dropout=0.1
    # ),

    # --- Decoder ---
    # 'decoder_type':                'transformer_relative_diagonal',
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
    'batch_size':                  32,
    'num_batches':                 512,
    'num_epochs':                  20000,

    # ======== model ID ========
    'timestamp':                   None,
    'savename':                    Path(__file__).stem,
}

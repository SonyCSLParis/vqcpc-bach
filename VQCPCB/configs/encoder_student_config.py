from pathlib import Path

subdivision = 4

config = {
    # ======== Model ===========
    'training_method':             'Student',  # DCPC or Student
    'dataset':                     'bach',  # bach or bach_small

    # ======== Dataloader ======
    'dataloader_generator_kwargs': dict(
        sequences_size=12
    ),
    'subdivision':                 subdivision,  # Number of frame per quarter note

    # ======== Encoder =========
    # --- DataProcessor ---
    'data_processor_type':         'bach',  # can be used to filter out some channels
    'data_processor_kwargs':       dict(
        embedding_size=32
    ),
    # --- Downscaler ---
    'downscaler_type':             'relative_transformer_downscaler',
    'downscaler_kwargs':           dict(
        # DCPC uses a Transformer
        # downscale_factors=[4, 8],
        downscale_factors=[4, 4],
        d_model=512,
        n_head=8,
        # list_of_num_layers=[3, 5],
        list_of_num_layers=[4, 4],
        dim_feedforward=2048,
        attention_masking_type='block',  # None or 'block'
        # both are bidirectional
        attention_bias_type='relative_attention',
        dropout=0.1
    ),
    # --- Quantizer ---
    'quantizer_type': 'commitment',
    'quantizer_kwargs': dict(
        num_codebooks=1,
        codebook_size=32,
        codebook_dim=3,
        commitment_cost=0.25,
        use_batch_norm=False,
        squared_l2_norm=True
        # add corrupt indices
    ),

    # --- Upscaler ---
    'upscaler_type': None,
    # 'upscaler_kwargs': dict(),

    # ======== AuxiliaryNetworks =====
    'auxiliary_networks_kwargs':   {
        # multiplicative term in front of the quantization loss
        'quantization_weighting':   1,
        'num_events_masked':        4,
        'teacher_type':             'relative',
        # relative or absolute
        'teacher_kwargs':           dict(
            # teacher must have its own data_processor
            data_processor_config=dict(
                data_processor_type='bach',
                data_processor_kwargs=dict(
                    embedding_size=32
                )
            ),
            num_layers=8,
            positional_embedding_size=8,
            d_model=512,
            dim_feedforward=2048,
            n_head=8,
            dropout=0.1,
        ),

        'auxiliary_decoder_type':   'relative',
        # relative or absolute
        'auxiliary_decoder_kwargs': dict(
            positional_embedding_size=8,
            d_model=512,
            dim_feedforward=2048,
            n_head=8,
            dropout=0.1,
            list_of_num_layers=[4, 4]
        )  # the decoder mirrors the encoder

    },

    # ======== Training ========
    'lr': 1e-4,
    'batch_size': 16,
    'num_batches': 512,
    'num_epochs': 20000,
    'quantizer_regularization': dict(
        corrupt_labels=False
    ),

    # ======== model ID ========
    'timestamp': None,
    'savename': Path(__file__).stem,
}

import numpy as np

from VQCPCB.auxiliary_decoders.auxiliary_decoder import AuxiliaryDecoder
from VQCPCB.auxiliary_decoders.auxiliary_decoder_relative import AuxiliaryDecoderRelative
from VQCPCB.data_processor.bach_cpc_data_processor import BachCPCDataProcessor
from VQCPCB.data_processor.bach_data_processor import BachDataProcessor
from VQCPCB.data_processor.data_processor import DataProcessor
from VQCPCB.dataloaders.bach_cpc_dataloader import BachCPCDataloaderGenerator
from VQCPCB.dataloaders.bach_dataloader import BachDataloaderGenerator
from VQCPCB.decoders.decoder import Decoder
from VQCPCB.downscalers.lstm_downscaler import LstmDownscaler
from VQCPCB.downscalers.relative_transformer_downscaler import RelativeTransformerDownscaler
from VQCPCB.downscalers.relative_transformer_downscaler_linear import \
    RelativeTransformerDownscalerLinear
from VQCPCB.encoder import Encoder
from VQCPCB.priors.prior_relative import PriorRelative
from VQCPCB.quantizer.vector_quantizer import ProductVectorQuantizer, NoQuantization
from VQCPCB.student_encoder_trainer import StudentEncoderTrainer
from VQCPCB.teachers.teacher_relative import TeacherRelative
from VQCPCB.upscalers.mlp_upscaler import MlpUpscaler
from VQCPCB.vqcpc_encoder_trainer import VQCPCEncoderTrainer


def get_dataloader_generator(
        dataset,
        training_method,
        dataloader_generator_kwargs):
    if dataset.lower() == 'bach':
        if training_method.lower() == 'vqcpc':
            return BachCPCDataloaderGenerator(
                num_tokens_per_block=dataloader_generator_kwargs['num_tokens_per_block'],
                num_blocks_left=dataloader_generator_kwargs['num_blocks_left'],
                num_blocks_right=dataloader_generator_kwargs['num_blocks_right'],
                negative_sampling_method=dataloader_generator_kwargs['negative_sampling_method'],
                num_negative_samples=dataloader_generator_kwargs['num_negative_samples']
            )
        elif (training_method.lower() == 'student'
              or training_method.lower() == 'decoder'
              or training_method.lower() == 'prior'):
            return BachDataloaderGenerator(
                sequences_size=dataloader_generator_kwargs['sequences_size']
            )
    else:
        raise NotImplementedError("If you want to use your own datasets, you need to implement a"
                                  " datasets, data_processor and dataloader")


def get_downscaler(downscaler_type,
                   downscaler_kwargs):
    if downscaler_type == 'relative_transformer_downscaler':
        return RelativeTransformerDownscaler(
            input_dim=downscaler_kwargs['input_dim'],
            output_dim=downscaler_kwargs['output_dim'],
            downscale_factors=downscaler_kwargs['downscale_factors'],
            num_channels=downscaler_kwargs['num_channels'],
            d_model=downscaler_kwargs['d_model'],
            n_head=downscaler_kwargs['n_head'],
            list_of_num_layers=downscaler_kwargs['list_of_num_layers'],
            dim_feedforward=downscaler_kwargs['dim_feedforward'],
            dropout=downscaler_kwargs['dropout'],
        )
    if downscaler_type == 'relative_transformer_downscaler_linear':
        return RelativeTransformerDownscalerLinear(
            input_dim=downscaler_kwargs['input_dim'],
            output_dim=downscaler_kwargs['output_dim'],
            downscale_factors=downscaler_kwargs['downscale_factors'],
            num_channels=downscaler_kwargs['num_channels'],
            d_model=downscaler_kwargs['d_model'],
            n_head=downscaler_kwargs['n_head'],
            list_of_num_layers=downscaler_kwargs['list_of_num_layers'],
            dim_feedforward=downscaler_kwargs['dim_feedforward'],
            dropout=downscaler_kwargs['dropout'],
        )
    elif downscaler_type == 'lstm_downscaler':
        return LstmDownscaler(
            input_dim=downscaler_kwargs['input_dim'],
            output_dim=downscaler_kwargs['output_dim'],
            num_channels=downscaler_kwargs['num_channels'],
            downscale_factors=downscaler_kwargs['downscale_factors'],
            # additional params
            hidden_size=downscaler_kwargs['hidden_size'],
            num_layers=downscaler_kwargs['num_layers'],
            dropout=downscaler_kwargs['dropout'],
            bidirectional=downscaler_kwargs['bidirectional']
        )
    elif downscaler_type == 'mlp_downscaler':
        return MlpDownscaler(
            input_dim=downscaler_kwargs['input_dim'],
            output_dim=downscaler_kwargs['output_dim'],
            downscale_factors=downscaler_kwargs['downscale_factors'],
            # additional params
            hidden_size=downscaler_kwargs['hidden_size'],
            num_layers=downscaler_kwargs['num_layers'],
            dropout=downscaler_kwargs['dropout'],
        )

    else:
        raise NotImplementedError


def get_upscaler(upscaler_type, upscaler_kwargs):
    """

    :param downscaler_type:
    :param downscaler_kwargs:
    :return:
    """
    if upscaler_type == 'mlp_upscaler':
        return MlpUpscaler(
            input_dim=upscaler_kwargs['input_dim'],
            output_dim=upscaler_kwargs['output_dim'],
            hidden_size=upscaler_kwargs['hidden_size'],
            dropout=upscaler_kwargs['dropout']
        )
    elif upscaler_type == None:
        return None
    else:
        raise NotImplementedError


def get_encoder(model_dir,
                dataloader_generator,
                config
                ):
    training_method = config['training_method']
    quantizer_kwargs = config['quantizer_kwargs']
    downscaler_kwargs = config['downscaler_kwargs']
    if config['upscaler_type'] is not None:
        upscaler_kwargs = config['upscaler_kwargs']

    if training_method.lower() == 'vqcpc':
        data_processor: DataProcessor = get_data_processor(
            dataloader_generator=dataloader_generator,
            data_processor_type=config['data_processor_type'],
            data_processor_kwargs=config['data_processor_kwargs'],
        )

        downscaler_kwargs['input_dim'] = data_processor.embedding_size
        downscaler_kwargs['output_dim'] = quantizer_kwargs['codebook_dim']
        downscaler_kwargs['num_tokens'] = data_processor.num_events * data_processor.num_channels
        downscaler_kwargs['num_channels'] = data_processor.num_channels
        downscaler = get_downscaler(
            downscaler_type=config['downscaler_type'],
            downscaler_kwargs=downscaler_kwargs
        )

        if config['quantizer_type'] == 'commitment':
            quantizer = ProductVectorQuantizer(
                codebook_size=quantizer_kwargs['codebook_size'],
                codebook_dim=quantizer_kwargs['codebook_dim'],
                initialize=quantizer_kwargs['initialize'],
                squared_l2_norm=quantizer_kwargs['squared_l2_norm'],
                use_batch_norm=quantizer_kwargs['use_batch_norm'],
                commitment_cost=quantizer_kwargs['commitment_cost']
            )
        elif config['quantizer_type'] is None:
            quantizer = NoQuantization(codebook_dim=quantizer_kwargs['codebook_dim'])

        if config['upscaler_type'] is not None:
            upscaler_kwargs['input_dim'] = quantizer_kwargs['codebook_dim']
            upscaler = get_upscaler(upscaler_type=config['upscaler_type'],
                                    upscaler_kwargs=upscaler_kwargs)
        else:
            upscaler = None

        encoder = Encoder(
            model_dir=model_dir,
            data_processor=data_processor,
            downscaler=downscaler,
            quantizer=quantizer,
            upscaler=upscaler
        )

        return encoder

    elif training_method.lower() == 'student':
        data_processor: DataProcessor = get_data_processor(
            dataloader_generator=dataloader_generator,
            data_processor_type=config['data_processor_type'],
            data_processor_kwargs=config['data_processor_kwargs']
        )

        # add required parameters to downscaler_kwargs
        downscaler_kwargs['input_dim'] = data_processor.embedding_size
        downscaler_kwargs['output_dim'] = quantizer_kwargs['codebook_dim']
        downscaler_kwargs['num_tokens'] = data_processor.num_tokens
        downscaler_kwargs['num_channels'] = data_processor.num_channels
        downscaler = get_downscaler(downscaler_type=config['downscaler_type'],
                                    downscaler_kwargs=downscaler_kwargs)

        quantizer = ProductVectorQuantizer(
            codebook_size=quantizer_kwargs['codebook_size'],
            num_codebooks=quantizer_kwargs['num_codebooks'],
            codebook_dim=quantizer_kwargs['codebook_dim'],
            initialize=quantizer_kwargs['initialize'],
            squared_l2_norm=quantizer_kwargs['squared_l2_norm'],
            use_batch_norm=quantizer_kwargs['use_batch_norm'],
            commitment_cost=quantizer_kwargs['commitment_cost']
        )

        if config['upscaler_type'] is not None:
            upscaler_kwargs['input_dim'] = quantizer_kwargs['codebook_dim']
            upscaler = get_upscaler(upscaler_type=config['upscaler_type'],
                                    upscaler_kwargs=upscaler_kwargs)
        else:
            upscaler = None

        encoder = Encoder(
            model_dir=model_dir,
            data_processor=data_processor,
            downscaler=downscaler,
            quantizer=quantizer,
            upscaler=upscaler
        )
        return encoder
    else:
        raise NotImplementedError


def get_teacher(teacher_type,
                teacher_kwargs,
                dataloader_generator):
    data_processor_config = teacher_kwargs['data_processor_config']
    data_processor = get_data_processor(dataloader_generator=dataloader_generator,
                                        data_processor_type=data_processor_config[
                                            'data_processor_type'],
                                        data_processor_kwargs=data_processor_config[
                                            'data_processor_kwargs']
                                        )
    return TeacherRelative(
        num_layers=teacher_kwargs['num_layers'],
        num_tokens_per_channel=teacher_kwargs['num_tokens_per_channel'],
        d_model=teacher_kwargs['d_model'],
        positional_embedding_size=teacher_kwargs['positional_embedding_size'],
        dim_feedforward=teacher_kwargs['dim_feedforward'],
        n_head=teacher_kwargs['n_head'],
        dropout=teacher_kwargs['dropout'],
        num_tokens=teacher_kwargs['num_tokens'],
        data_processor=data_processor
    )


def get_auxiliary_decoder(auxiliary_decoder_type,
                          auxiliary_decoder_kwargs):
    if auxiliary_decoder_type == 'absolute':
        return AuxiliaryDecoder(
            num_tokens_per_channel=auxiliary_decoder_kwargs['num_tokens_per_channel'],
            codebook_dim=auxiliary_decoder_kwargs['codebook_dim'],
            upscale_factors=auxiliary_decoder_kwargs['upscale_factors'],
            n_head=auxiliary_decoder_kwargs['n_head'],
            dim_feedforward=auxiliary_decoder_kwargs['dim_feedforward'],
            list_of_num_layers=auxiliary_decoder_kwargs['list_of_num_layers'],
            d_model=auxiliary_decoder_kwargs['d_model'],
            num_tokens_bottleneck=auxiliary_decoder_kwargs['num_tokens_bottleneck'],
            dropout=auxiliary_decoder_kwargs['dropout']
        )
    elif auxiliary_decoder_type == 'relative':
        return AuxiliaryDecoderRelative(
            num_tokens_per_channel=auxiliary_decoder_kwargs['num_tokens_per_channel'],
            codebook_dim=auxiliary_decoder_kwargs['codebook_dim'],
            upscale_factors=auxiliary_decoder_kwargs['upscale_factors'],
            n_head=auxiliary_decoder_kwargs['n_head'],
            dim_feedforward=auxiliary_decoder_kwargs['dim_feedforward'],
            list_of_num_layers=auxiliary_decoder_kwargs['list_of_num_layers'],
            d_model=auxiliary_decoder_kwargs['d_model'],
            num_tokens_bottleneck=auxiliary_decoder_kwargs['num_tokens_bottleneck'],
            dropout=auxiliary_decoder_kwargs['dropout']
        )
    else:
        raise NotImplementedError


def get_decoder(model_dir,
                dataloader_generator,
                data_processor,
                encoder,
                decoder_type,
                decoder_kwargs):
    num_channels_decoder = data_processor.num_channels
    num_events_decoder = data_processor.num_events
    num_channels_encoder = 1
    num_events_encoder = int((num_events_decoder * num_channels_decoder) // \
                             (np.prod(encoder.downscaler.downscale_factors) *
                              num_channels_encoder)
                             )

    if decoder_type == 'transformer':
        decoder = Decoder(
            model_dir=model_dir,
            dataloader_generator=dataloader_generator,
            data_processor=data_processor,
            encoder=encoder,
            transformer_type='absolute',
            encoder_attention_type='anticausal',  # anticausal, causal, diagonal or full
            cross_attention_type='full',  # anticausal, causal, diagonal or full
            d_model=decoder_kwargs['d_model'],
            num_encoder_layers=decoder_kwargs['num_encoder_layers'],
            num_decoder_layers=decoder_kwargs['num_decoder_layers'],
            n_head=decoder_kwargs['n_head'],
            dim_feedforward=decoder_kwargs['dim_feedforward'],
            dropout=decoder_kwargs['dropout'],
            positional_embedding_size=decoder_kwargs['positional_embedding_size'],
            num_channels_encoder=num_channels_encoder,
            num_events_encoder=num_events_encoder,
            num_channels_decoder=num_channels_decoder,
            num_events_decoder=num_events_decoder,
        )
    elif decoder_type == 'transformer_relative':
        decoder = Decoder(
            model_dir=model_dir,
            dataloader_generator=dataloader_generator,
            data_processor=data_processor,
            encoder=encoder,
            transformer_type='relative',
            encoder_attention_type='anticausal',
            cross_attention_type='anticausal',
            d_model=decoder_kwargs['d_model'],
            num_encoder_layers=decoder_kwargs['num_encoder_layers'],
            num_decoder_layers=decoder_kwargs['num_decoder_layers'],
            n_head=decoder_kwargs['n_head'],
            dim_feedforward=decoder_kwargs['dim_feedforward'],
            dropout=decoder_kwargs['dropout'],
            positional_embedding_size=decoder_kwargs['positional_embedding_size'],
            num_channels_encoder=num_channels_encoder,
            num_events_encoder=num_events_encoder,
            num_channels_decoder=num_channels_decoder,
            num_events_decoder=num_events_decoder,
        )
    elif decoder_type == 'transformer_relative_fullCross':
        decoder = Decoder(
            model_dir=model_dir,
            dataloader_generator=dataloader_generator,
            data_processor=data_processor,
            encoder=encoder,
            transformer_type='relative',
            encoder_attention_type='anticausal',
            cross_attention_type='full',
            d_model=decoder_kwargs['d_model'],
            num_encoder_layers=decoder_kwargs['num_encoder_layers'],
            num_decoder_layers=decoder_kwargs['num_decoder_layers'],
            n_head=decoder_kwargs['n_head'],
            dim_feedforward=decoder_kwargs['dim_feedforward'],
            dropout=decoder_kwargs['dropout'],
            positional_embedding_size=decoder_kwargs['positional_embedding_size'],
            num_channels_encoder=num_channels_encoder,
            num_events_encoder=num_events_encoder,
            num_channels_decoder=num_channels_decoder,
            num_events_decoder=num_events_decoder,
        )
    elif decoder_type == 'transformer_relative_diagonal':
        decoder = Decoder(
            model_dir=model_dir,
            dataloader_generator=dataloader_generator,
            data_processor=data_processor,
            encoder=encoder,
            transformer_type='relative',
            encoder_attention_type='anticausal',
            cross_attention_type='diagonal',
            d_model=decoder_kwargs['d_model'],
            num_encoder_layers=decoder_kwargs['num_encoder_layers'],
            num_decoder_layers=decoder_kwargs['num_decoder_layers'],
            n_head=decoder_kwargs['n_head'],
            dim_feedforward=decoder_kwargs['dim_feedforward'],
            dropout=decoder_kwargs['dropout'],
            positional_embedding_size=decoder_kwargs['positional_embedding_size'],
            num_channels_encoder=num_channels_encoder,
            num_events_encoder=num_events_encoder,
            num_channels_decoder=num_channels_decoder,
            num_events_decoder=num_events_decoder,
        )
    elif decoder_type == 'transformer_relative_full':
        decoder = Decoder(
            model_dir=model_dir,
            dataloader_generator=dataloader_generator,
            data_processor=data_processor,
            encoder=encoder,
            transformer_type='relative',
            encoder_attention_type='full',
            cross_attention_type='full',
            d_model=decoder_kwargs['d_model'],
            num_encoder_layers=decoder_kwargs['num_encoder_layers'],
            num_decoder_layers=decoder_kwargs['num_decoder_layers'],
            n_head=decoder_kwargs['n_head'],
            dim_feedforward=decoder_kwargs['dim_feedforward'],
            dropout=decoder_kwargs['dropout'],
            positional_embedding_size=decoder_kwargs['positional_embedding_size'],
            num_channels_encoder=num_channels_encoder,
            num_events_encoder=num_events_encoder,
            num_channels_decoder=num_channels_decoder,
            num_events_decoder=num_events_decoder,
        )
    else:
        raise NotImplementedError
    return decoder


def get_prior(model_dir,
              dataloader_generator,
              encoder,
              prior_type,
              prior_kwargs
              ):
    if prior_type == 'transformer_relative':
        num_channels = 1
        data_processor = encoder.data_processor
        num_events = int(
            (data_processor.num_events * data_processor.num_channels) // \
            (
                    np.prod(encoder.downscaler.downscale_factors) * num_channels)
        )

        prior = PriorRelative(
            model_dir,
            dataloader_generator=dataloader_generator,
            encoder=encoder,
            d_model=prior_kwargs['d_model'],
            num_layers=prior_kwargs['num_layers'],
            n_head=prior_kwargs['n_head'],
            dim_feedforward=prior_kwargs['dim_feedforward'],
            embedding_size=prior_kwargs['embedding_size'],
            dropout=prior_kwargs['dropout'],
            num_channels=num_channels,
            num_events=num_events
        )
    else:
        raise NotImplementedError
    return prior


def get_encoder_trainer(model_dir,
                        dataloader_generator,
                        training_method,
                        encoder,
                        auxiliary_networks_kwargs):
    if training_method.lower() == 'vqcpc':
        return VQCPCEncoderTrainer(
            model_dir=model_dir,
            dataloader_generator=dataloader_generator,
            encoder=encoder,
            c_net_kwargs=auxiliary_networks_kwargs['c_net_kwargs'],
            quantization_weighting=auxiliary_networks_kwargs['quantization_weighting']
        )
    elif training_method.lower() == 'student':
        # --- teacher
        # add required kwargs
        teacher_kwargs = auxiliary_networks_kwargs['teacher_kwargs']
        teacher_kwargs['num_tokens_per_channel'] = encoder.data_processor.num_tokens_per_channel
        teacher_kwargs['num_tokens'] = encoder.data_processor.num_tokens

        teacher = get_teacher(
            teacher_type=auxiliary_networks_kwargs['teacher_type'],
            teacher_kwargs=teacher_kwargs,
            dataloader_generator=dataloader_generator
        )

        # --- auxiliary decoder
        # add required kwargs
        auxiliary_decoder_kwargs = auxiliary_networks_kwargs['auxiliary_decoder_kwargs']
        auxiliary_decoder_kwargs[
            'num_tokens_per_channel'] = encoder.data_processor.num_tokens_per_channel
        auxiliary_decoder_kwargs['codebook_dim'] = encoder.quantizer.codebook_dim
        auxiliary_decoder_kwargs['upscale_factors'] = list(reversed(
            encoder.downscaler.downscale_factors
        ))
        auxiliary_decoder_kwargs['num_tokens_bottleneck'] = (
                encoder.data_processor.num_tokens // np.prod(encoder.downscaler.downscale_factors)
        )
        auxiliary_decoder = get_auxiliary_decoder(
            auxiliary_decoder_type=auxiliary_networks_kwargs['auxiliary_decoder_type'],
            auxiliary_decoder_kwargs=auxiliary_decoder_kwargs
        )

        return StudentEncoderTrainer(
            model_dir=model_dir,
            dataloader_generator=dataloader_generator,
            encoder=encoder,
            teacher=teacher,
            auxiliary_decoder=auxiliary_decoder,
            quantization_weighting=auxiliary_networks_kwargs['quantization_weighting'],
            num_events_masked=auxiliary_networks_kwargs['num_events_masked']
        )
    else:
        raise NotImplementedError


def get_data_processor(dataloader_generator,
                       data_processor_type,
                       data_processor_kwargs):
    if data_processor_type == 'bach':
        # compute num_events num_tokens_per_channel
        dataset = dataloader_generator.dataset
        num_events = dataset.sequences_size * dataset.subdivision
        num_tokens_per_channel = [len(d) for d in dataset.index2note_dicts]
        data_processor = BachDataProcessor(embedding_size=data_processor_kwargs['embedding_size'],
                                           num_events=num_events,
                                           num_tokens_per_channel=num_tokens_per_channel)
        return data_processor

    elif data_processor_type == 'bach_cpc':
        # compute num_events num_tokens_per_channel
        dataset = dataloader_generator.dataset_positive
        num_tokens_per_block = dataloader_generator.num_tokens_per_block
        num_channels = dataloader_generator.num_channels
        num_events = dataset.sequences_size * dataset.subdivision
        num_tokens_per_channel = [len(d) for d in dataset.index2note_dicts]
        data_processor = BachCPCDataProcessor(
            embedding_size=data_processor_kwargs['embedding_size'],
            num_events=num_events,
            num_channels=num_channels,
            num_tokens_per_channel=num_tokens_per_channel,
            num_tokens_per_block=num_tokens_per_block)
        assert dataloader_generator.num_channels == data_processor.num_channels
        return data_processor
    else:
        raise NotImplementedError

import os
import random

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from VQCPCB.utils import dict_pretty_print, flatten
import matplotlib.pyplot as plt


class Encoder(nn.Module):
    """
    Base class for BachEncoder

    Encoder is composed of
    - a data_processor comprised of
      ___ preprocessing
        from ? to (batch_size, num_events, num_tokens_per_event)
      ___ embedding
        from (batch_size, num_events, num_tokens_per_event)
        to (batch_size, num_events * num_tokens_per_event, embedding_dim)
    - a downscaler
      from (batch_size, num_events * num_tokens_per_event, embedding_dim)
      to (batch_size, num_events * num_tokens_per_event // downscale_factor, codebook_dim)
    - a quantizer
      from (batch_size, num_events * num_tokens_per_event // downscale_factor, codebook_dim)
      to (batch_size, num_events * num_tokens_per_event // downscale_factor, num_codebooks)
    """

    def __init__(self,
                 model_dir,
                 data_processor,
                 downscaler,
                 quantizer,
                 upscaler
                 ):
        super(Encoder, self).__init__(
        )
        # TODO put in DataParallel
        self.data_processor = data_processor
        self.downscaler = downscaler
        self.quantizer = quantizer
        self.upscaler = upscaler
        self.model_dir = model_dir

    def save(self, early_stopped):
        if early_stopped:
            model_dir = f'{self.model_dir}/early_stopped'
        else:
            model_dir = f'{self.model_dir}/overfitted'
        torch.save(self.data_processor.state_dict(), f'{model_dir}/data_processor')
        torch.save(self.downscaler.state_dict(), f'{model_dir}/downscaler')
        torch.save(self.quantizer.state_dict(), f'{model_dir}/quantizer')
        if self.upscaler is not None:
            torch.save(self.upscaler.state_dict(), f'{model_dir}/upscaler')
        # print(f'Model {self.__repr__()} saved')

    def load(self, early_stopped):
        print(f'Loading models {self.__repr__()}')
        if early_stopped:
            model_dir = f'{self.model_dir}/early_stopped'
        else:
            model_dir = f'{self.model_dir}/overfitted'

        #  Deal with older configs
        if not os.path.exists(model_dir):
            model_dir = self.model_dir

        self.data_processor.load_state_dict(torch.load(f'{model_dir}/data_processor'))
        self.downscaler.load_state_dict(torch.load(f'{model_dir}/downscaler'))
        self.quantizer.load_state_dict(torch.load(f'{model_dir}/quantizer'))
        if self.upscaler:
            self.upscaler.load_state_dict(torch.load(f'{model_dir}/upscaler'))

    def forward(self, x, corrupt_labels=False):
        """

        :param x: x comes from the dataloader
        :param corrupt_labels: if true, assign with probability 5% a different label than the computed centroid
        :return: z_quantized, encoding_indices, quantization_loss
        """
        x_proc = self.data_processor.preprocess(x)
        x_embed = self.data_processor.embed(x_proc)
        x_flat = flatten(x_embed)
        z = self.downscaler.forward(x_flat)
        z_quantized, encoding_indices, quantization_loss = self.quantizer.forward(
            z,
            corrupt_labels=corrupt_labels
        )

        if self.upscaler is not None:
            z_quantized = self.upscaler(z_quantized)

        return z_quantized, encoding_indices, quantization_loss

    def plot_clusters(self, dataloader_generator, split_name, batch_size=32, num_batches=64):
        """
        Visualize elements belonging to the same cluster
        Elements belong to the training set
        :param split_name: name of the dataset split (train, val, test)
        :param dataloader_generator:
        :param batch_size:
        :param num_batches:
        :return:
        """
        (generator_train,
         generator_val,
         generator_test) = dataloader_generator.dataloaders(
            batch_size=batch_size,
            num_workers=0)

        if split_name == 'train':
            generator = generator_train
        elif split_name == 'val':
            generator = generator_val
        elif split_name == 'test':
            generator = generator_test
        else:
            raise ValueError(f'{split_name} is not a valid split value. Choose between train, val or test')

        self.eval()
        d = {}
        for k, tensor_dict in enumerate(generator):
            with torch.no_grad():
                original_x = tensor_dict['x']
                z_quantized, encoding_indices, quantization_loss = self.forward(original_x)

                num_events_for_one_index = int(np.product(self.downscaler.downscale_factors) // \
                                               len(self.data_processor.num_tokens_per_channel)
                                               )
                for batch_index, (encoding_index_of_batch, corresponding_x_of_batch) in enumerate(
                        zip(encoding_indices, original_x)
                ):
                    for cluster_index, corresponding_x in zip(encoding_index_of_batch,
                                                              corresponding_x_of_batch.split(
                                                                  num_events_for_one_index, dim=0)):
                        cluster_index = cluster_index.item()
                        if cluster_index not in d:
                            d[cluster_index] = []
                        d[cluster_index].append(corresponding_x)

                if k > num_batches:
                    break

        #  Write scores
        if not os.path.exists(f'{self.model_dir}/clusters_{split_name}/'):
            os.mkdir(f'{self.model_dir}/clusters_{split_name}/')

        for unit_index, list_elements in d.items():
            # element is [num_events, num_channels]
            save_path = f'{self.model_dir}/clusters_{split_name}/{unit_index}'

            random.shuffle(list_elements)
            # keep only a limited number of examples
            list_elements = list_elements[:50]
            # to score
            tensor_score = self.data_processor.postprocess(list_elements)
            dataloader_generator.write(tensor_score, save_path)

            #
            # tensor_score = self.data_processor.list_to_tensor(list_elements)
            # score = dataloader_generator.to_score(tensor_score)
            # self.data_processor.write(score, save_path)
            # print(f'File {save_path} saved')
        return

    def show_nn_clusters(self, k=3):
        self.eval()
        clusters = self.quantizer.embeddings._parameters['0'].data.cpu()
        dists = torch.norm(clusters[None, :, :] - clusters[:, None, :], p=2, dim=2)
        print('Nearest neighbours list:')
        for i in range(dists.size(0)):
            res = torch.topk(dists[i], k=k + 1, largest=False)[1][1:].numpy()
            print(f'{i}: {res}')

    def scatterplot_clusters_3d(self):
        """
        Plot clusters. Only works for embedding dimensions < 4
        :return:
        """
        self.eval()
        clusters = self.quantizer.embeddings._parameters['0'].data.cpu().numpy()
        x = clusters[:, 0]
        y = clusters[:, 1]
        z = clusters[:, 2]
        # x = y = z = [1, 2, 3]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for i in range(len(x)):  # plot each point + it's index as text above
            ax.scatter(x[i], y[i], z[i], color='b')
            ax.text(x[i], y[i], z[i], '%s' % (str(i)), size=12, zorder=1, color='k')

        ## Hovering stuff, think it works, but need tkinter fucking shit which does not work on my computer
        # # now try to get the display coordinates of the first point
        #
        # x2, y2, _ = proj3d.proj_transform(1, 1, 1, ax.get_proj())
        #
        # label = plt.annotate(
        #     "this",
        #     xy=(x2, y2), xytext=(-20, 20),
        #     textcoords='offset points', ha='right', va='bottom',
        #     bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
        #     arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        #
        # def update_position(e):
        #     x2, y2, _ = proj3d.proj_transform(1, 1, 1, ax.get_proj())
        #     label.xy = x2, y2
        #     label.update_positions(fig.canvas.renderer)
        #     fig.canvas.draw()
        #
        # fig.canvas.mpl_connect('button_release_event', update_position)

        savepath = f'{self.model_dir}/clusters_scatter.pdf'
        plt.savefig(savepath)

        return


class EncoderTrainer(nn.Module):
    """
    Base class for DCPCTrainer and StudentEncoderTrainer
    """

    def __init__(self, dataloader_generator):
        """
        Create appropriate auxiliary networks
        :param downscaler:
        """
        super(EncoderTrainer, self).__init__()
        self.dataloader_generator = dataloader_generator

    def train_model(self,
                    batch_size,
                    num_batches=None,
                    num_epochs=10,
                    lr=1e-3,
                    corrupt_labels=False,
                    plot=False,
                    num_workers=0,
                    **kwargs
                    ):

        if plot:
            self.writer = SummaryWriter(f'{self.model_dir}')

        best_val = 1e8
        self.init_optimizers(lr=lr)
        for epoch_id in range(num_epochs):
            (generator_train,
             generator_val,
             generator_test) = self.dataloader_generator.dataloaders(
                batch_size=batch_size,
                num_workers=num_workers)

            monitored_quantities_train = self.epoch(
                data_loader=generator_train,
                train=True,
                num_batches=num_batches,
                corrupt_labels=corrupt_labels,
            )

            del generator_train
            monitored_quantities_val = self.epoch(
                data_loader=generator_val,
                train=False,
                num_batches=num_batches // 2 if num_batches is not None else None,
                corrupt_labels=corrupt_labels,
            )
            del generator_val

            print(f'======= Epoch {epoch_id} =======')
            print(f'---Train---')
            dict_pretty_print(monitored_quantities_train, endstr=' ' * 5)
            print()
            print(f'---Val---')
            dict_pretty_print(monitored_quantities_val, endstr=' ' * 5)
            print('\n')

            # What criterion for early stopping ?? loss_contrastive ? loss_monitor ?
            self.save(early_stopped=False)
            valid_loss = monitored_quantities_val['loss_monitor']
            if valid_loss < best_val:
                self.save(early_stopped=True)
                best_val = valid_loss

            if plot:
                self.plot(epoch_id,
                          monitored_quantities_train,
                          monitored_quantities_val)

    def plot(self, epoch_id, monitored_quantities_train,
             monitored_quantities_val, index_encoder=None):
        if index_encoder is not None:
            suffix = f'_{index_encoder})'
        else:
            suffix = ''

        if monitored_quantities_train is not None:
            for k, v in monitored_quantities_train.items():
                if type(v) == list:
                    for ind, elem in enumerate(v):
                        self.writer.add_scalar(f'{k}_{ind}{suffix}/train', elem, epoch_id)
                else:
                    self.writer.add_scalar(f'{k}{suffix}/train', v, epoch_id)

        if monitored_quantities_val is not None:
            for k, v in monitored_quantities_val.items():
                if type(v) == list:
                    for ind, elem in enumerate(v):
                        self.writer.add_scalar(f'{k}_{ind}{suffix}/val', elem, epoch_id)
                else:
                    self.writer.add_scalar(f'{k}{suffix}/val', v, epoch_id)

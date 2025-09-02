from pathlib import Path
from typing import List, Tuple, Callable, Optional, Literal

import pandas as pd

import torch
import torch_geometric
import pytorch_lightning as pl
from torch_geometric.nn.dense.linear import Linear
import torch.nn as nn
import torch.nn.functional as F

from vqniche.modules.mlp import MLP as MLP_AdjacencyDecoder
from ..decoders.mlp_softmax import MLPSoftmax
from vqniche.loss import (
    cross_entropy_loss,
    mse_attribute_reconstruction_loss,
    nb_attribute_reconstruction_loss,
    mse_adjacency_reconstruction_loss,
    bce_adjacency_reconstruction_loss,
    mse_joint_code_commit_loss,
    mse_commit_loss,
    mse_code_loss,
    l2_codebook_orthogonal_regularization_loss,
    mask_token_regularization
)
from vqniche.utils.type_conversions import inference_data_dict_to_adata
from vqniche.metrics import compute_benchmarking_metrics


class BaseModel(pl.LightningModule):
    def __init__(
            self,
            model_name: str,
            encoder_name: str,
            attribute_decoder_name: str,
            adjacency_decoder_name: str,
            predictor_name: str,
            in_channels: int = None,
            out_channels: int = None,
            train_metrics_list: List[str] = [],
            optimizer_name: str = 'adam',
            lr: float = 0.01,
            weight_decay: float = 0.0,
            mask_lr_scale: float = 1.0,
            loss_names: List[str] = ['cross_entropy'],
            loss_kwargs: dict = {'reduction': 'mean'},
        ) -> None:
        """
        Initialize the BaseModel class.

        Parameters
        ----------
        - model_name: str
            The name of the model.
        - encoder_name: str
            The encoder name.
        - attribute_decoder_name: str
            The name of the attribute decoder module.
        - adjacency_decoder_name: str
            The name of the adjacency decoder module.
        - predictor_name: str
            The name of the predictor module.

        - in_channels: int
            The number of input channels.
        - out_channels: int
            The number of output channels.

        - train_metrics_list: List[str]
            The list of metrics to compute during training.

        - optimizer_name: str
            The optimizer name.
        - lr: float
            The learning rate.
        - weight_decay: float
            The weight decay.
        - mask_lr_scale: float
            The learning rate scale for the learnable mask.

        - loss_names: List[str]
            The loss function names.
        - loss_kwargs: dict
            The loss function keyword arguments.
        """
        # names of the model components
        self.model_name = model_name
        self.encoder_name = encoder_name
        self.attribute_decoder_name = attribute_decoder_name
        self.adjacency_decoder_name = adjacency_decoder_name
        self.predictor_name = predictor_name

        super().__init__()

        # Data parameters
        self.in_channels = in_channels
        self.out_channels = out_channels

        # metrics to compute during training
        self.train_metrics_list = train_metrics_list

        self.on_train_epoch_end_logs_df = pd.DataFrame()

        # Optimizer parameters
        self.optimizer_name = optimizer_name
        self.lr = lr
        self.weight_decay = weight_decay
        self.mask_lr_scale = mask_lr_scale

        # Loss parameters
        self.loss_kwargs = loss_kwargs
        self.dispersion = torch.nn.Parameter(torch.randn(self.in_channels))
        self.loss_fn_tuples = self.set_loss_fn_tuples(loss_names, loss_kwargs)

        self.save_hyperparameters()


    def set_loss_fn_tuples(
            self,
            loss_fn_names: List[str],
            loss_kwargs: dict = {}
        ) -> List[Tuple[str, Callable, List[str], dict]]:
        """
        Set the loss functions for the model.

        Parameters
        ----------
        - loss_names: List[str]
            The loss function names.
        - loss_kwargs: dict
            Additional keyword arguments.

        Returns
        -------
        loss_fn_tuples: List
            One tuple per loss name in loss_names comprising of loss function name (str), loss function (callable), list of data related key strings required to be passed to the loss function, and a dictionary of additional keyword arguments for the loss function.

        Notes
        -----
        Use this method to set the loss functions for the encoder. The loss functions should be defined in the utils.loss module.
        """
        # initialize a list to store the loss function tuples
        loss_fn_tuples = []

        print("Setting the following loss terms as criterion for training:")
        for loss_fn_name in loss_fn_names:
            print(f"Loss function: {loss_fn_name}")
            loss_fn_params = {}

            if loss_fn_name == 'cross_entropy_loss':
                # set the cross-entropy loss function
                loss_fn = cross_entropy_loss

                # set key names for data required to compute cross-entropy loss
                loss_fn_data_keys = ['logits', 'labels']

                # set keyword parameters for cross-entropy loss
                wt_cross_entropy = loss_kwargs.get('wt_cross_entropy')
                if wt_cross_entropy is not None:
                    loss_fn_params['wt_cross_entropy'] = wt_cross_entropy

            elif loss_fn_name == 'mse_attribute_reconstruction_loss':
                loss_fn = mse_attribute_reconstruction_loss

                loss_fn_data_keys = ['pred_attr', 'target_attr']

                wt_attr_reconstr = loss_kwargs.get('wt_attr_reconstr')
                if wt_attr_reconstr is not None:
                    loss_fn_params['wt_attr_reconstr'] = wt_attr_reconstr

            elif loss_fn_name == 'nb_attribute_reconstruction_loss':
                loss_fn = nb_attribute_reconstruction_loss

                loss_fn_data_keys = ['pred_attr', 'target_attr', 'edge_index', 'batch_size', 'dispersion']

                k_hop_nb_loss = loss_kwargs.get('k_hop_nb_loss')
                if k_hop_nb_loss is not None:
                    loss_fn_params['k_hop_nb_loss'] = k_hop_nb_loss

                wt_attr_reconstr = loss_kwargs.get('wt_attr_reconstr')
                if wt_attr_reconstr is not None:
                    loss_fn_params['wt_attr_reconstr'] = wt_attr_reconstr

            elif loss_fn_name == 'mse_adjacency_reconstruction_loss':
                loss_fn = mse_adjacency_reconstruction_loss

                loss_fn_data_keys = ['batch_size', 'h_adj', 'batch_edge_index']

                estimate_adj_kwargs = loss_kwargs.get('estimate_adj_kwargs')
                if estimate_adj_kwargs is not None:
                    loss_fn_params['estimate_adj_kwargs'] = estimate_adj_kwargs

                wt_adj_reconstr = loss_kwargs.get('wt_adj_reconstr')
                if wt_adj_reconstr is not None:
                    loss_fn_params['wt_adj_reconstr'] = wt_adj_reconstr

            elif loss_fn_name == 'bce_adjacency_reconstruction_loss':
                loss_fn = bce_adjacency_reconstruction_loss

                loss_fn_data_keys = ['batch_size', 'h_adj', 'batch_edge_index']
                
                edge_sampling_ratio = loss_kwargs.get('edge_sampling_ratio')
                if edge_sampling_ratio is not None:
                    loss_fn_params['edge_sampling_ratio'] = edge_sampling_ratio

                use_pos_weight = loss_kwargs.get('use_pos_weight')
                if use_pos_weight is not None:
                    loss_fn_params['use_pos_weight'] = use_pos_weight

                estimate_adj_kwargs = loss_kwargs.get('estimate_adj_kwargs')
                if estimate_adj_kwargs is not None:
                    loss_fn_params['estimate_adj_kwargs'] = estimate_adj_kwargs

                wt_adj_reconstr = loss_kwargs.get('wt_adj_reconstr')
                if wt_adj_reconstr is not None:
                    loss_fn_params['wt_adj_reconstr'] = wt_adj_reconstr

            elif loss_fn_name == 'mse_joint_code_commit_loss':
                loss_fn = mse_joint_code_commit_loss

                loss_fn_data_keys = ['quantizer_input', 'quantizer_output']

                wt_joint_code_commit = loss_kwargs.get('wt_joint_code_commit')
                if wt_joint_code_commit is not None:
                    loss_fn_params['wt_joint_code_commit'] = wt_joint_code_commit

            elif loss_fn_name == 'mse_commit_loss':
                loss_fn = mse_commit_loss

                loss_fn_data_keys = ['quantizer_input', 'quantizer_output']

                wt_commit = loss_kwargs.get('wt_commit')
                if wt_commit is not None:
                    loss_fn_params['wt_commit'] = wt_commit

            elif loss_fn_name == 'mse_code_loss':
                loss_fn = mse_code_loss

                loss_fn_data_keys = ['quantizer_input', 'quantizer_output']

                wt_code = loss_kwargs.get('wt_code')
                if wt_code is not None:
                    loss_fn_params['wt_code'] = wt_code

            elif loss_fn_name == 'l2_codebook_orthogonal_regularization_loss':
                loss_fn = l2_codebook_orthogonal_regularization_loss

                loss_fn_data_keys = ['codebook_embeddings']

                wt_codebook_orthogonal_regularization = loss_kwargs.get('wt_codebook_orthogonal_regularization')
                if wt_codebook_orthogonal_regularization is not None:
                    loss_fn_params['wt_codebook_orthogonal_regularization'] = wt_codebook_orthogonal_regularization

                codebook_reg_active_codes_only = loss_kwargs.get('codebook_reg_active_codes_only')
                if codebook_reg_active_codes_only is not None:
                    loss_fn_params['codebook_reg_active_codes_only'] = codebook_reg_active_codes_only

                codebook_reg_max_codes = loss_kwargs.get('codebook_reg_max_codes')
                if codebook_reg_max_codes is not None:
                    loss_fn_params['codebook_reg_max_codes'] = codebook_reg_max_codes

            elif loss_fn_name == 'mask_token_regularization':
                loss_fn = mask_token_regularization

                loss_fn_data_keys = ['mask_token']

                wt_mask_token_regularization = loss_kwargs.get('wt_mask_token_regularization')
                if wt_mask_token_regularization is not None:
                    loss_fn_params['wt_mask_token_regularization'] = wt_mask_token_regularization

            else:
                raise NotImplementedError(f'{loss_fn_name} Loss not implemented')

            loss_fn_tuple = (loss_fn_name, loss_fn, loss_fn_data_keys, loss_fn_params)
            loss_fn_tuples.append(loss_fn_tuple)

        return loss_fn_tuples


    def criterion(
            self,
            loss_data: dict,
            curr_batch_size: Optional[int] = None,
            mode: Literal['train', 'val'] = 'train',
        ) -> torch.Tensor:
        """
        Compute the loss for the model.

        Parameters
        ----------
        - loss_data: dict
            A collection of data objects required to compute the loss.
        - curr_batch_size: int
            The number of samples in the current batch. Required for logging.
        - mode: Literal['train', 'val']
            The mode of the model (train, val).

        Returns
        -------
        - total_loss: torch.Tensor
            The total computed loss across all loss terms for the current batch.
        """
        assert len(self.loss_fn_tuples) > 0, 'No loss functions defined'

        # initialize total_loss = 0.0 with requires_grad=True so that the loss can be backpropagated
        # total_loss will be computed as the sum of all the loss terms from self.loss_names
        total_loss = torch.tensor(0.0, requires_grad=True, dtype=torch.float32).to(self.device)

        # during model initialization, self.loss_fn_tuples is set to a list of tuples
        # one tuple per loss name in self.loss_names
        # each tuple contains the loss function name, the callable loss function, the data keys required to compute the loss, and the loss function parameters
        for loss_fn_name, loss_fn, loss_fn_data_keys, loss_fn_params in self.loss_fn_tuples:
            # extract from loss_data, the data required to compute the current loss function
            _loss_fn_data = {key: loss_data[key] for key in loss_fn_data_keys}

            # pass the extracted data to the loss function along with the loss function related kwargs
            loss_fn_value = loss_fn(**_loss_fn_data, **loss_fn_params)
            # add the computed loss to the total_loss
            total_loss = torch.add(total_loss, loss_fn_value)

            # log each computed loss term
            self.log(
                    name=f"{mode}_{loss_fn_name}",
                    value=loss_fn_value,
                    prog_bar=False,
                    on_step=False,
                    on_epoch=True,
                    batch_size=curr_batch_size,
                    sync_dist=True,
                    )

            # free up memory by deleting the intermediate loss function data
            del _loss_fn_data

        return total_loss


    def _init_attribute_decoder(
            self,
            in_channels: int,
            out_channels: int,
            attribute_decoder_name: Literal['MLPSoftmax'] = 'MLPSoftmax',
            attribute_decoder_params: dict = {},
        ) -> pl.LightningModule:
        """
        Initialize the attribute decoder module.

        Parameters
        ----------
        - in_channels: int
            The input dimension of the attribute decoder module.
        - out_channels: int
            The output dimension of the attribute decoder module.
        - attribute_decoder_name: Literal['MLPSoftmax']
            The name of the attribute decoder module.
        - attribute_decoder_params: dict
            The parameters for the attribute decoder module.

        Returns
        -------
        - attribute_decoder: pl.LightningModule
            The attribute decoder module.
        """
        if attribute_decoder_name == 'MLPSoftmax':
            return MLPSoftmax(
                in_channels=in_channels,
                out_channels=out_channels,
                **attribute_decoder_params,
            )


    def _init_adjacency_decoder(
            self,
            in_channels: int,
            out_channels: int = 600,
            adjacency_decoder_name: Literal['MLP_AdjacencyDecoder'] = 'MLP_AdjacencyDecoder',
            mlp_params: dict = {},
        ) -> torch.nn.Module:
        """
        Initialize the adjacency decoder module.

        Parameters
        ----------
        - in_channels: int
            The input dimension of the adjacency decoder module.
        - adjacency_decoder_name: Literal['MLP_AdjacencyDecoder']
            The name of the adjacency decoder module.
        - mlp_params: dict
            The parameters for the MLP module.
        - conditioning_params: dict
            The parameters for the conditioning module.

        Returns
        -------
        - adjacency_decoder: torch.nn.Module
            The adjacency decoder module.
        """
        if adjacency_decoder_name == 'MLP_AdjacencyDecoder':
            return MLP_AdjacencyDecoder(
                in_channels=in_channels,
                out_channels=out_channels,
                **mlp_params,
            )


    def _init_predictor(
            self,
            predictor_name: Literal['Linear'] = 'Linear',
            in_channels: int = None,
            out_channels: int = None,
            init_method: str = 'kaiming_uniform'
        ) -> pl.LightningModule:
        """
        Initialize the predictor module.

        Parameters
        ----------
        - predictor_name: str
            The name of the predictor module.
        - in_channels: int
            The input dimension of the predictor module.
        - out_channels: int
            The output dimension of the predictor module.
        - init_method: str
            The initialization method for the predictor module.

        Returns
        -------
        - predictor: pl.LightningModule
            The predictor module.
        """
        if predictor_name == 'Linear':
            return Linear(
                in_channels=in_channels,
                out_channels=out_channels,
                weight_initializer=init_method
            )


    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Configure the optimizer for the model.

        Returns
        -------
        - torch.optim.Optimizer
            The configured optimizer.
        """
        # TODO: Add support for multiple optimizers
        if self.optimizer_name == 'adam':
            # mask_params, other_params = [], []
            # for name, param in self.named_parameters():
            #     (mask_params if name.endswith("learnable_mask") else other_params).append(param)
            
            # return torch.optim.Adam(
            #     [
            #         {"params": other_params, "lr": self.lr, "weight_decay": self.weight_decay},
            #         {"params": mask_params, "lr": self.lr * self.mask_lr_scale, "weight_decay": 0.0},
            #     ]
            # )
            return torch.optim.Adam(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        else:
            raise NotImplementedError(f'Optimizer {self.optimizer_name} not implemented')


    def common_step(
            self,
            batch_loss_data: dict,
            batch_size: int,
            mode: Literal['train', 'val'] = 'train',
        ) -> torch.Tensor:
        """
        Compute and log the loss for a model for a given batch during training or validation.

        Parameters
        ----------
        - batch_loss_data: dict
            The data required to compute the loss.
        - batch_size: int
            The size of the batch.
        - mode: Literal['train', 'val']
            The mode of the fit process (train, val).

        Returns
        -------
        - torch.Tensor
            The computed loss for the current batch.
        """
        loss_value = self.criterion(
            loss_data=batch_loss_data,
            curr_batch_size=batch_size,
        )

        self.log(
            name=f'{mode}_loss',
            value=loss_value,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
            sync_dist=True,
        )

        return loss_value


    def forward(
            self,
            batch_x: torch.Tensor,
            batch_edge_index: torch.Tensor
        ) -> torch.Tensor:
        """
        Forward pass of the model. The batch of nodes may be the entire set of nodes in the graph or a subset of nodes.

        Parameters
        ----------
        - batch_x: torch.Tensor
            The input features of the batch of nodes.
        - batch_edge_index: torch.Tensor
            The edge index tensor of the batch of nodes.

        Returns
        -------
        - unnormalized_logits: torch.Tensor
            The unnormalized logits of the model.
        """
        raise NotImplementedError('Forward pass should be implemented by the subclass')


    def _print_epoch_stats(self) -> None:
        """
        Print the epoch stats for the current epoch.
        """
        # print logged values during the current epoch segregated by loss terms and metrics
        logged_value_types = {
            'train': [],
            'val': [],
            'other': [],
            }
        for logged_value_name in self.trainer.callback_metrics.keys():
            if logged_value_name.startswith('train'):
                logged_value_types['train'].append(logged_value_name)
            elif logged_value_name.startswith('val'):
                logged_value_types['val'].append(logged_value_name)
            else:
                logged_value_types['other'].append(logged_value_name)

        print(f"----------------Train-----------------")        
        for loss_term_name in logged_value_types['train']:
                print(f"{loss_term_name}: {self.trainer.callback_metrics[loss_term_name]}")
        print("--------------------------------------------\n")
        
        print(f"----------------Validation-----------------")        
        for loss_term_name in logged_value_types['val']:
                print(f"{loss_term_name}: {self.trainer.callback_metrics[loss_term_name]}")
        print("--------------------------------------------\n")
        
        if len(logged_value_types['other']) > 0:
            print(f"----------------Other-----------------")        
            for loss_term_name in logged_value_types['other']:
                print(f"{loss_term_name}: {self.trainer.callback_metrics[loss_term_name]}")
            print("--------------------------------------------\n")
        
        print(f"--------------------------------End of Epoch {self.current_epoch}--------------------------------------\n\n")
        
        return


    def _update_on_train_epoch_end_logs_df(self) -> None:
        """
        Update the on_train_epoch_end_logs_df dataframe with the logged values for the current epoch.
        """
        logged_values_keys = list(self.trainer.callback_metrics.keys())
        logged_values_values = [value.item() for value in self.trainer.callback_metrics.values()]
        logged_values_dict = dict(zip(logged_values_keys, logged_values_values))
        logged_values_dict['epoch'] = self.current_epoch
        self.on_train_epoch_end_logs_df = pd.concat(
            [
                self.on_train_epoch_end_logs_df,
                pd.DataFrame(
                    [logged_values_dict],
                    columns=list(logged_values_dict.keys())
                )
            ],
            ignore_index=True
        )
        return


    def compute_metrics(
        self,
        mode: Literal['train', 'val', 'test'] = 'val',
    ) -> None:
        """
        Compute metrics for a given mode. The mode determines the dataloader and inference data cache used to compute the metrics.

        Parameters
        ----------
        - mode: Literal['train', 'val', 'test']
            The mode of the fit process (train, val, test).
        """
        # 1) Get the dataloader for the given mode
        dataloader = getattr(self.trainer.datamodule, f'{mode}_dataloader')()

        # 2) Get the data and model outputs cached during the steps of the given mode
        inference_data_cache = getattr(self, f'{mode}_inference_data_cache')

        # self.train_metrics_list is a list of metrics specified in the train config file
        # this is used to track the model performance at the end of each epoch
        # currently, the trainer.validate() and trainer.test() methods also use this list
        # a fuller list of metrics must be manually computed separately
        # TODO: Add support for computing this full list
        if len(self.train_metrics_list) > 0:
            # 3) Concatenate the inference data cache
            for key in self.cache_keys:
                inference_data_cache[key] = torch.cat(inference_data_cache[key], dim=0)
            inference_data_cache['edge_index'] = dataloader.data.edge_index

            # 4) Convert the inference data to an AnnData object
            adata = inference_data_dict_to_adata(
                inference_data=inference_data_cache,
                label_categories_dict=None,
            )

            # 5) Compute the benchmarking metrics
            metrics_dict = compute_benchmarking_metrics(
                adata=adata,
                metrics=self.train_metrics_list,
                **self.loss_kwargs['estimate_adj_kwargs'],
            )
            
            # 6) Clear the inference data cache
            for key in self.cache_keys:
                inference_data_cache[key] = []

            return metrics_dict


    def on_train_epoch_start(self) -> None:
        """
        Pytorch Lightning hook that is executed at the start of each training epoch.

        Notes
        -----
        - We use this hook to print the start of the current training epoch.
        """        
        print(f"--------------------------------Start of Epoch {self.current_epoch}--------------------------------------")

        # call the parent class method to complete default behavior
        return super().on_train_epoch_start()


    def training_step(
            self,
            train_batch: torch_geometric.data.Data,
            batch_idx: Optional[int] = None,
        ) -> torch.Tensor:
        """
        Pytorch Lightning hook that is executed for each training batch after the on_train_epoch_start() hook but before the on_validation_epoch_start() hook.

        Parameters
        ----------
        - train_batch: torch_geometric.data.Data
            The training batch.
        - batch_idx: int
            The batch index.

        Returns
        -------
        - torch.Tensor
            The computed loss.

        Notes
        -----
        - We use this hook to define the training step for the model. This is expected to be implemented by the child class.
        """
        raise NotImplementedError('Training step not implemented')


    def on_validation_epoch_start(self) -> None:
        """
        Pytorch Lightning hook that is executed at the start of each validation epoch. During training, this hook is executed within the on_train_epoch_start() and on_train_epoch_end() hooks, but after the training_step() hook is executed for each training batch.

        Notes
        -----
        - We do not use this hook in the base class. It is included for readability.
        """
        return super().on_validation_epoch_start()


    def validation_step(
            self,
            val_batch: torch_geometric.data.Data,
            batch_idx: Optional[int] = None,
        ) -> torch.Tensor:
        """
        Pytorch Lightning hook that is executed for each validation batch after the on_validation_epoch_start() hook but before the on_validation_epoch_end() hook.

        Parameters
        ----------
        - val_batch: torch_geometric.data.Data
            The validation batch.
        - batch_idx: int
            The batch index.

        Returns
        -------
        - torch.Tensor
            The computed validation loss.

        Notes
        -----
        - We use this hook to define the validation step for the model. This is expected to be implemented by the child class.
        """
        raise NotImplementedError('Validation step not implemented')


    def on_validation_epoch_end(self) -> None:
        """
        Pytorch Lightning hook that is executed at the end of each validation epoch. During training, this hook is executed within the on_train_epoch_start() and on_train_epoch_end() hooks, but after the validation_step() hook is executed for each validation batch.

        Notes
        -----
        - We use this hook to compute and log metrics for the nodes in the validation set.
        - This 
        """
        # compute metrics for the nodes in the validation set based on cached data
        metrics_dict = self.compute_metrics(mode='val')
        
        # log the metrics for the nodes in the validation set
        for key, value in metrics_dict.items():
            self.log(
                name=f'val_{key}',
                value=value,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )

        # call the parent class method to complete default behavior
        return super().on_validation_epoch_end()


    def on_train_epoch_end(self) -> None:
        """
        Pytorch Lightning hook that is executed at the end of each training epoch. During training, this hook is executed after the training_step() hook is executed for each training batch and after the on_validation_epoch_start(), validation_step() for each validation batch, and on_validation_epoch_end() hooks are executed.

        Notes
        -----
        - We use this hook to print the loss terms and metrics for the current epoch.
        - We also use this hook to update the on_train_epoch_end_logs_df dataframe which is used to store the loss terms and metrics for all epochs.
        """
        metrics_dict = self.compute_metrics(mode='train')
        for key, value in metrics_dict.items():
            self.log(
                name=f'train_{key}',
                value=value,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )

        # print the epoch stats to the console
        self._print_epoch_stats()
        
        # write the logged values (loss terms + metrics) to the epoch metrics dataframe
        self._update_on_train_epoch_end_logs_df()

        # call the parent class method to complete default behavior
        return super().on_train_epoch_end()


    def on_fit_end(self) -> None:
        """
        Pytorch Lightning hook that is executed at the end of the training process.

        Notes
        -----
        - We use this hook to save the on_train_epoch_end_logs_df dataframe to a CSV file.
        """
        # save epoch-wise logged loss terms and metrics to a CSV file
        on_train_epoch_end_logs_fname = Path(self.logger.experiment.dir) / 'on_train_epoch_end_logs.csv'
        self.on_train_epoch_end_logs_df.to_csv(
            path_or_buf=on_train_epoch_end_logs_fname,
            sep=',',
            index=False
        )

        return super().on_fit_end()
    
    
    def on_test_model_eval(self) -> None:
        """
        Pytorch Lightning hook that is executed before the test steps are called.
        """        
        return super().on_test_model_eval()


    def test_step(
            self,
            test_batch: torch_geometric.data.Data,
            batch_idx: Optional[int] = None,
        ) -> torch.Tensor:
        """
        Pytorch Lightning hook that is executed for each test batch after the on_test_epoch_start() hook but before the on_test_epoch_end() hook. This hook is outside the training loop.

        Parameters
        ----------
        - test_batch: torch_geometric.data.Data
            The test batch.
        - batch_idx: int
            The batch index.

        Returns
        -------
        - torch.Tensor
            The computed test accuracy.

        Notes
        -----
        - We use this hook to define the test step for the model. This is expected to be implemented by the child class.
        """
        raise NotImplementedError('Test step not implemented')
    
    
    def on_test_model_train(self) -> None:
        """
        Pytorch Lightning hook that is executed after all test steps are completed.
        """
        metrics_dict = self.compute_metrics(mode='test')
        
        print("--------------------------------Test Metrics--------------------------------")
        for key, value in metrics_dict.items():
            print(f"{key}: {value}")
        print("----------------------------------------------------------------")
        
        return super().on_test_model_train()
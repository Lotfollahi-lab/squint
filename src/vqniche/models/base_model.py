from pathlib import Path
from typing import List, Tuple, Callable, Optional, Literal

import pandas as pd
import networkx as nx

import torch
import torch_geometric
import pytorch_lightning as pl
from torch_geometric.nn.dense.linear import Linear

from vqniche import metrics
from ..utils.loss import *
from ..modules.mlp import MLP as MLP_AdjacencyDecoder
from ..decoders.mlp_softmax import MLPSoftmax
from ..utils.loss_utils import aggregate_1hop_neighbor_features
from ..utils.type_conversions import edge_index_to_adjacency_tensor
from ..utils.adjacency_reconstruction import reconstruct_adjacency_matrix as construct_binary_adjacency_matrix


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
            log_metrics_during_training: bool = False,
            log_similarity: bool = False,
            log_attribute_imputation: bool = False,
            log_graph_imputation: bool = False,
            log_codebook_utilization: Optional[bool] = None,
            optimizer_name: str = 'adam',
            lr: float = 0.01,
            weight_decay: float = 0.0,
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

        - log_metrics_during_training: bool
            Whether to log the metrics during training.
        - log_similarity: bool
            Whether to log the similarity statistics.
        - log_attribute_imputation: bool
            Whether to log metrics for attribute imputation.
        - log_graph_imputation: bool
            Whether to log metrics for graph imputation.
        - log_codebook_utilization: bool
            Whether to log the codebook utilization.

        - optimizer_name: str
            The optimizer name.
        - lr: float
            The learning rate.
        - weight_decay: float
            The weight decay.

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

        # log flags
        self.log_metrics_during_training = log_metrics_during_training
        self.log_similarity = log_similarity
        self.log_attribute_imputation = log_attribute_imputation
        self.log_graph_imputation = log_graph_imputation
        if log_codebook_utilization is not None:
            self.log_codebook_utilization = log_codebook_utilization

        self.train_val_epoch_metrics = pd.DataFrame()

        # Optimizer parameters
        self.optimizer_name = optimizer_name
        self.lr = lr
        self.weight_decay = weight_decay

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

            if loss_fn_name == 'cross_entropy':
                # set the cross-entropy loss function
                loss_fn = cross_entropy

                # set key names for data required to compute cross-entropy loss
                loss_fn_data_keys = ['logits', 'labels']

                # set keyword parameters for cross-entropy loss
                wt_cross_entropy = loss_kwargs.get('wt_cross_entropy')
                if wt_cross_entropy is not None:
                    loss_fn_params['wt_cross_entropy'] = wt_cross_entropy

            elif loss_fn_name == 'mse_attribute_reconstruction':
                loss_fn = mse_attribute_reconstruction

                loss_fn_data_keys = ['pred_attr', 'target_attr']

                wt_attr_reconstr = loss_kwargs.get('wt_attr_reconstr')
                if wt_attr_reconstr is not None:
                    loss_fn_params['wt_attr_reconstr'] = wt_attr_reconstr

            elif loss_fn_name == 'nb_attribute_reconstruction':
                loss_fn = nb_attribute_reconstruction

                loss_fn_data_keys = ['pred_attr', 'target_attr', 'edge_index', 'batch_size', 'dispersion']

                k_hop_nb_loss = loss_kwargs.get('k_hop_nb_loss')
                if k_hop_nb_loss is not None:
                    loss_fn_params['k_hop_nb_loss'] = k_hop_nb_loss

                wt_attr_reconstr = loss_kwargs.get('wt_attr_reconstr')
                if wt_attr_reconstr is not None:
                    loss_fn_params['wt_attr_reconstr'] = wt_attr_reconstr

            elif loss_fn_name == 'mse_adjacency_reconstruction':
                loss_fn = mse_adjacency_reconstruction

                loss_fn_data_keys = ['batch_size', 'h_adj', 'batch_edge_index']

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
            in_channels: int = None,
            out_channels: int = None,
            attribute_decoder_name: Literal['MLPSoftmax'] = 'MLPSoftmax',
            attribute_decoder_params: dict = {}
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
                name=attribute_decoder_name,
                use_xy_coordinates=attribute_decoder_params['use_xy_coordinates'],
                mlp_params=attribute_decoder_params['mlp_params'],
            )


    def _init_adjacency_decoder(
            self,
            in_channels: int,
            adjacency_decoder_name: Literal['MLP_AdjacencyDecoder'] = 'MLP_AdjacencyDecoder',
            adjacency_decoder_params: dict = {}
        ) -> torch.nn.Module:
        """
        Initialize the adjacency decoder module.

        Parameters
        ----------
        - in_channels: int
            The input dimension of the adjacency decoder module.
        - adjacency_decoder_name: Literal['MLP_AdjacencyDecoder']
            The name of the adjacency decoder module.
        - adjacency_decoder_params: dict
            The parameters for the adjacency decoder module.

        Returns
        -------
        - adjacency_decoder: torch.nn.Module
            The adjacency decoder module.
        """
        if adjacency_decoder_name == 'MLP_AdjacencyDecoder':
            if 'out_channels' not in adjacency_decoder_params:
                adjacency_decoder_params['out_channels'] = in_channels
            return MLP_AdjacencyDecoder(
                in_channels=in_channels,
                out_channels=adjacency_decoder_params['out_channels'],
                mlp_params=adjacency_decoder_params['mlp_params'],
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
            return torch.optim.Adam(
                params=self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay
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


    def training_step(
            self,
            train_batch: torch_geometric.data.Data,
            batch_idx: Optional[int] = None,
        ) -> torch.Tensor:
        """
        Training step for the model.

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
        """
        raise NotImplementedError('Training step not implemented')


    def on_train_epoch_end(self) -> None:
        """
        Callback function to be executed at the end of each training epoch.

        Notes
        -----
        - We use this hook to log embedding similarity, imputation, and codebook utilization metrics if enabled, as well as the loss term values and accuracies at the end of every training epoch.
        """
        # log model evaluation metrics at the end of each training epoch if enabled
        if self.log_metrics_during_training:    
            metrics_dict = {}
        
            # collect the raw input data and model embeddings for the infer_dataloader()
            # infer_dataloader() returns a dataloader for the entire dataset
            # child classes should implement this method
            inference_data = self.collect_inference_data(
                                self.trainer.datamodule.infer_dataloader()
                            )
            
            if self.log_similarity:
                metrics_dict.update(
                    self.compute_similarity_metrics(                
                            data_dict=inference_data
                    )
                )
            if self.log_attribute_imputation:
                metrics_dict.update(
                    self.compute_attribute_imputation_metrics(
                        data_dict=inference_data
                    )
                )
            if self.log_graph_imputation:
                metrics_dict.update(
                    self.compute_graph_imputation_metrics(
                        data_dict=inference_data
                    )
                )
                
            if self.model_name == 'VQNiche':
                if self.log_codebook_utilization:
                    num_active_codes = len(set(inference_data['Indices'].cpu().numpy()))
                    total_codes = 1.0 * self.encoder.vq.codebook.shape[0]
                    metrics_dict['codebook_utilization'] = num_active_codes / total_codes

            # log the training epoch end stats
            for key, value in metrics_dict.items():
                self.log(
                    name=key,
                    value=value,
                    prog_bar=False,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )
        
        # print to stdout all logged values during the training epoch
        metric_names = ['epoch'] + list(self.trainer.callback_metrics.keys())
        metrics_values = [self.current_epoch] + [value.item() for value in self.trainer.callback_metrics.values()]
        print("--------------------------------")
        for metric_name, metric_value in zip(metric_names, metrics_values):
            print(f"{metric_name}: {metric_value}")
        print("--------------------------------\n")

        # update the training and validation epoch metrics dataframe with metrics logged during the training epoch
        self.train_val_epoch_metrics = pd.concat(
            [
                self.train_val_epoch_metrics,
                pd.DataFrame(
                    [metrics_values],
                    columns=metric_names
                )
            ],
            ignore_index=True
        )

        # call the parent class method to complete default behavior
        return super().on_train_epoch_end()


    def compute_similarity_metrics(
            self,
            data_dict: dict,
        ) -> dict:
        """
        Compute cosine similarity between the raw input data and the model embeddings.

        Parameters
        ----------
        - data_dict: dict
            Dictionary containing the inference data with keys: X, Y_cell_type, Y_niche_type, edge_index, H_latent, X_hat, H_adj, and H_quantized and Indices (if VQNiche)

        Returns
        -------
        - dict
            Dictionary containing the computed cosine similarity between the raw input data and the model embeddings.

        Notes
        -----
        - This method may be used to check if any of the embeddings have collapsed.
        """
        similarity_metrics = {}

        similarity_metrics.update(
            metrics.cosine_similarity(data_dict['X'], 'X')
        )
        similarity_metrics.update(
            metrics.cosine_similarity(data_dict['H_latent'], 'H_latent')
        )
        similarity_metrics.update(
            metrics.cosine_similarity(data_dict['X_hat'], 'X_hat')
        )
        similarity_metrics.update(
            metrics.cosine_similarity(data_dict['H_adj'], 'H_adj')
        )
        if self.model_name == 'VQNiche':
            similarity_metrics.update(
                metrics.cosine_similarity(data_dict['H_quantized'], 'H_quantized')
            )

        return similarity_metrics


    def compute_attribute_imputation_metrics(
            self,
            data_dict: dict,
        ) -> dict:
        """
        Compute metrics to measure the quality of imputation of the original and estimated attributes (here, gene expression).

        Parameters
        ----------
        - data_dict: dict
            Dictionary containing the inference data with keys: X, Y_cell_type, Y_niche_type, edge_index, H_latent, X_hat, H_adj, and H_quantized and Indices (if VQNiche)

        Returns
        -------
        - dict
            Dictionary containing the computed metrics for attribute imputation.
        """
        attribute_imputation_metrics = {}
        
        # Pearson correlation between the original and estimated gene expression of a cell averaged across cells (row-wise)
        pearson_cell_wise = metrics.pearson_correlation(
            data_dict['X'].cpu().numpy(),
            data_dict['X_hat'].cpu().numpy(),
            compare_genes=False,
            mean=True,
        )
        attribute_imputation_metrics['pearson_cell_wise'] = pearson_cell_wise

        # Pearson correlation between the original and estimated gene expression averaged across genes (column-wise)
        pearson_gene_wise = metrics.pearson_correlation(
                    data_dict['X'].cpu().numpy(),
                    data_dict['X_hat'].cpu().numpy(),
                    compare_genes=True,
                    mean=True,
                )
        attribute_imputation_metrics['pearson_gene_wise'] = pearson_gene_wise
        
        # Pearson correlation between the original and estimated gene expression of 1-hop neighborhood of a cell averaged across cells
        X_nbr = aggregate_1hop_neighbor_features(
                    X=data_dict['X'].cpu(),
                    edge_index=data_dict['edge_index'].cpu(),
                    return_mean=True,
                )
        X_hat_nbr = aggregate_1hop_neighbor_features(
                    X=data_dict['X_hat'].cpu(),
                    edge_index=data_dict['edge_index'].cpu(),
                    return_mean=True,
                )
        pearson_1hop_nbr = metrics.pearson_correlation(
                    X_nbr,
                    X_hat_nbr,
                    compare_genes=False,
                    mean=True,
                )
        attribute_imputation_metrics['pearson_1hop_nbr'] = pearson_1hop_nbr

        return attribute_imputation_metrics
        

    def compute_graph_imputation_metrics(
            self,
            data_dict: dict,
        ) -> dict:
        """
        Compute metrics to measure the quality of imputation of the original and estimated graph structure.

        Parameters
        ----------
        - data_dict: dict
            Dictionary containing the inference data with keys: X, Y_cell_type, Y_niche_type, edge_index, H_latent, X_hat, H_adj, and H_quantized and Indices (if VQNiche)

        Returns
        -------
        - dict
            Dictionary containing the computed metrics for graph imputation.
        """
        graph_imputation_metrics = {}

        # build a networkx graph from the edge index
        G = nx.from_numpy_array(
                edge_index_to_adjacency_tensor(
                    data_dict['edge_index']
                ).cpu().numpy()
            )

        # build a networkx graph from the estimated adjacency matrix
        G_hat = nx.from_numpy_array(
                construct_binary_adjacency_matrix(
                    h_index_nodes=data_dict['H_adj'].detach(),
                    **self.loss_kwargs['estimate_adj_kwargs'],
                ).cpu().numpy()
            )
        
        # compute the number of edges and the maximum degree of the original and estimated graph
        graph_imputation_metrics['G_num_edges'] = G.number_of_edges()
        graph_imputation_metrics['G_hat_num_edges'] = G_hat.number_of_edges()
        graph_imputation_metrics['G_max_degree'] = max(dict(G.degree()).values())
        graph_imputation_metrics['G_hat_max_degree'] = max(dict(G_hat.degree()).values())

        # compute MMD between the degree distribution of the original and estimated graph
        mmd_degree = metrics.mmd_score(
                        [metrics.degree_histogram(G)],
                        [metrics.degree_histogram(G_hat)],
                    )
        graph_imputation_metrics['mmd_degree'] = mmd_degree

        # compute MMD between the eigenvalue distribution of the original and estimated graph
        mmd_eigenvalues = metrics.mmd_score(
                        [metrics.eigenvalues_pmf(G)],
                        [metrics.eigenvalues_pmf(G_hat)],
                    )
        graph_imputation_metrics['mmd_eigenvalues'] = mmd_eigenvalues
        
        return graph_imputation_metrics


    def collect_inference_data(self) -> dict:
        """
        Get inference data for computing training epoch stats.
        Child classes should override this method to provide the required data.

        Returns
        -------
        - dict
            Dictionary containing inference data with keys: X, edge_index, H_latent, X_hat, H_adj
        """
        raise NotImplementedError('collect_inference_data should be implemented by the subclass')


    def validation_step(
            self,
            val_batch: torch_geometric.data.Data,
            batch_idx: Optional[int] = None,
        ) -> torch.Tensor:
        """
        Validation step for the model.

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
        """
        raise NotImplementedError('Validation step not implemented')


    def test_step(
            self,
            test_batch: torch_geometric.data.Data,
            batch_idx: Optional[int] = None,
        ) -> torch.Tensor:
        """
        Test step for the model.

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
        """
        raise NotImplementedError('Test step not implemented')
    

    def on_fit_end(self) -> None:
        """
        Callback function to be executed at the end of the training loop.

        Notes
        -----
        - We use this hook to save the training and validation epoch metrics to a CSV file.
        """
        epoch_metrics_fname = Path(self.logger.experiment.dir) / 'train_val_epoch_metrics.csv'
        self.train_val_epoch_metrics.to_csv(
            path_or_buf=epoch_metrics_fname,
            sep=',',
            index=False
        )

        return super().on_fit_end()
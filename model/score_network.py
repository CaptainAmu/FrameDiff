"""Score network module."""
import torch
import math
from torch import nn
from torch.nn import functional as F
from data import utils as du
from data import all_atom
from model import ipa_pytorch
import functools as fn

Tensor = torch.Tensor


def get_index_embedding(indices, embed_size, max_len=2056):
    """Creates sine / cosine positional embeddings from a prespecified indices.

    Args:
        indices: [..., N_edge] int, offsets of positions to embed.
        embed_size: int, dimension of the embeddings to create
        max_len: int, maximum length (for scaling sine / cosine frequencies).
        
    Returns:
        pos_embedding: [..., N_edge, embed_size] float, cosine and sine embeddings.
    """
    K = torch.arange(embed_size//2, device=indices.device)
    pos_embedding_sin = torch.sin(
        indices[..., None] * math.pi / (max_len**(2*K[None]/embed_size))).to(indices.device)
    pos_embedding_cos = torch.cos(
        indices[..., None] * math.pi / (max_len**(2*K[None]/embed_size))).to(indices.device)
    pos_embedding = torch.cat([
        pos_embedding_sin, pos_embedding_cos], axis=-1)
    return pos_embedding


def get_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
    """
    Creates a sine / cosine positional embedding for the given timesteps.

    Args:
        timesteps: [T] int, timesteps to embed.
        embedding_dim: int, dimension of the embeddings to create.
        max_positions: int, maximum length (for scaling sine / cosine frequencies).

    Returns:
        emb: [T, embedding_dim] float, time embeddings
    """
    # Code from https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py
    assert len(timesteps.shape) == 1
    timesteps = timesteps * max_positions
    half_dim = embedding_dim // 2
    # Calculate the embedding frequency scaling factor
    emb = math.log(max_positions) / (half_dim - 1)
    # Create the embedding frequencies
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    # Create the embedding by multiplying the frequencies with the timesteps
    emb = timesteps.float()[:, None] * emb[None, :]
    # Create the sine and cosine embeddings by concatenating them
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    # Zero pad the embedding if the embedding dimension is odd
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1), mode='constant')
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb

class Embedder(nn.Module):

    def __init__(self, model_conf):
        """
        Initializes the Embedder module.

        Args:
            model_conf: A ModelConfig object containing the hyperparameters for the model.
        """
        super(Embedder, self).__init__()
        self._model_conf = model_conf
        self._embed_conf = model_conf.embed

        # Time step embedding
        #####
        # Note the "..._dims" are dims before the MLPs; after MLPs are "..._size".
        #####
        index_embed_size = self._embed_conf.index_embed_size  # e.g. 32
        t_embed_size = index_embed_size  # e.g. 32
        node_embed_dims = t_embed_size + 1  # Node MLP input.  e.g. 33. t_embed(32) + fixed_mask(1)
        edge_in = (t_embed_size + 1) * 2  # Edge MLP input.  e.g. 66. (t_embed(32) + fixed_mask(1)) * 2

        # Sequence index embedding
        node_embed_dims += index_embed_size  # Node MLP input.  e.g. 65. index_emb(32) + t_emb(32) + fixed_mask(1)
        edge_in += index_embed_size  # Edge MLP input.  e.g. 98. index_emb(32) + (t_emb(32) + fixed_mask(1)) * 2

        node_embed_size = self._model_conf.node_embed_size  # output dim of node embedder
        self.node_embedder = nn.Sequential(
            nn.Linear(node_embed_dims, node_embed_size),
            nn.ReLU(),
            nn.Linear(node_embed_size, node_embed_size),
            nn.ReLU(),
            nn.Linear(node_embed_size, node_embed_size),
            nn.LayerNorm(node_embed_size),  # normalize each node embedding independently 
        )  # out [N, node_embed_size]

        if self._embed_conf.embed_self_conditioning:
            edge_in += self._embed_conf.num_bins  # bins for distance between 2 residues. e.g. 98 + 8 = 106
        edge_embed_size = self._model_conf.edge_embed_size
        self.edge_embedder = nn.Sequential(
            nn.Linear(edge_in, edge_embed_size),
            nn.ReLU(),
            nn.Linear(edge_embed_size, edge_embed_size),
            nn.ReLU(),
            nn.Linear(edge_embed_size, edge_embed_size),
            nn.LayerNorm(edge_embed_size),
        )  # out [N, N, edge_embed_size]

        self.timestep_embedder = fn.partial(
            get_timestep_embedding,
            embedding_dim=self._embed_conf.index_embed_size
        )  # out [T, t_embed_size]
        self.index_embedder = fn.partial(
            get_index_embedding,
            embed_size=self._embed_conf.index_embed_size
        )  # out [N_edge, index_embed_size]

    def _cross_concat(self, feats_1d, num_batch, num_res):
        """
        Concatenates a set of 1D features by tiling them across the first and second dimensions.

        Args:
            feats_1d: [B, N, D] 1D features to be concatenated.
            num_batch: int B, number of batches.
            num_res: int N, number of residues.

        Returns:
            [B, N**2, 2*D] Concatenated features.
        """
        return torch.cat([
            torch.tile(feats_1d[:, :, None, :], (1, 1, num_res, 1)),  # Creates extra 3rd dim and tiles across it.
            torch.tile(feats_1d[:, None, :, :], (1, num_res, 1, 1)),  # Creates extra 2nd dim and tiles across it.
        ], dim=-1).float().reshape([num_batch, num_res**2, -1])

    def forward(
            self,
            *,
            seq_idx,
            t,
            fixed_mask,
            self_conditioning_ca,
        ):
        """
        Embeds a set of inputs

        Args:
            seq_idx: [B, N] Positional sequence index for each residue.
            t: [B] Sampled t in [0, 1].
            fixed_mask: [B, N] Mask of fixed (motif) residues.
            self_conditioning_ca: [B, N, 3] Ca positions of self-conditioning
                input. (Ca-sidechain frame translation previously predicted)

        Returns:
            node_embed: [B, N, node_embed_size] Node embeddings.
            edge_embed: [B, N, N, edge_embed_size] Edge embeddings.
        """
        num_batch, num_res = seq_idx.shape  # B, N
        node_feats = []

        # Set time step to epsilon=1e-5 for fixed residues.
        fixed_mask = fixed_mask[..., None]  # [B, N, 1]
        prot_t_embed = torch.tile(
            self.timestep_embedder(t)[:, None, :],  # [T, t_embed_size] -> [B, 1, t_embed_size]
            (1, num_res, 1))  # [B, N, t_embed_size]
        prot_t_embed = torch.cat([prot_t_embed, fixed_mask], dim=-1)  # [B, N, t_embed_size + 1]
        node_feats = [prot_t_embed]  # List of tensors, first is [B, N, t_embed_size + 1] 
        pair_feats = [self._cross_concat(prot_t_embed, num_batch, num_res)] # List of tensors, first is [B, N**2, 2*(t_embed_size+1)]

        # Positional index features.
        node_feats.append(self.index_embedder(seq_idx)) # [[B, N, t_embed_size + 1], [B, N, index_embed_size]]
        rel_seq_offset = seq_idx[:, :, None] - seq_idx[:, None, :]  # [B, N, N]
        rel_seq_offset = rel_seq_offset.reshape([num_batch, num_res**2])  # [B, N**2]
        pair_feats.append(self.index_embedder(rel_seq_offset))  # [[B, N**2, 2*(t_embed_size+1)], [B, N**2, index_embed_size]]

        # Self-conditioning distogram.
        if self._embed_conf.embed_self_conditioning:
            sc_dgram = du.calc_distogram(
                self_conditioning_ca,
                self._embed_conf.min_bin,
                self._embed_conf.max_bin,
                self._embed_conf.num_bins,
            )
            pair_feats.append(sc_dgram.reshape([num_batch, num_res**2, -1]))  # [[B, N**2, 2*(t_embed_size+1)], [B, N**2, index_embed_size], [B, N**2, num_bins]]

        node_embed = self.node_embedder(torch.cat(node_feats, dim=-1).float())  # [B, N, node_embed_size]
        edge_embed = self.edge_embedder(torch.cat(pair_feats, dim=-1).float())  # [B, N**2, edge_embed_size]
        edge_embed = edge_embed.reshape([num_batch, num_res, num_res, -1])  # [B, N, N, edge_embed_size]
        return node_embed, edge_embed  # [[B, N, node_embed_size], [B, N, N, edge_embed_size]]


class ScoreNetwork(nn.Module):

    def __init__(self, model_conf, diffuser):
        super(ScoreNetwork, self).__init__()
        self._model_conf = model_conf

        self.embedding_layer = Embedder(model_conf)
        self.diffuser = diffuser
        self.score_model = ipa_pytorch.IpaScore(model_conf, diffuser)  # invariant point attention score model

    def _apply_mask(self, psi_pred, psi_gt, diff_mask):
        # I updated the docstring here. I believe the original description using aatype was imprecise.
        """
        Applies a mask to the predicted psi torsion angles.

        Args:
            psi_pred: [B, N, 2] predicted psi angle (sin, cos).
            psi_gt: [B, N, 2] ground-truth psi angle (sin, cos).
            diff_mask: [B, N, 1] mask indicating where to use prediction (1=use pred, 0=use gt).

        Returns:
            masked_psi: [B, N, 2] masked psi angle (sin, cos).
        """
        return diff_mask * psi_pred + (1 - diff_mask) * psi_gt

    def forward(self, input_feats):
        """Forward computes the reverse diffusion conditionals p(X^t|X^{t+1})
        for each item in the batch

        Args:
            input_feats: dictionary of input features, containing:
                'res_mask': [B, N] mask for valid residues
                'fixed_mask': [B, N] mask for fixed (motif) residues
                'seq_idx': [B, N] positional sequence indices
                't': [B] time steps, with values in [0, 1]
                'sc_ca_t': [B, N, 3] self-conditioning Ca positions
                'torsion_angles_sin_cos': [B, N, 7, 2] ground-truth torsion angles (sin, cos)
            
        X: the noised samples from the noising process, of shape [Batch, N, D].
                Where the T time steps are t=1,...,T (i.e. not including the un-noised X^0)

        Returns:
            pred_out: dictionary of model outputs.
        """

        # Frames as [batch, res, 7] tensors.
        bb_mask = input_feats['res_mask'].type(torch.float32)  # [B, N] "BackBone" mask 
        fixed_mask = input_feats['fixed_mask'].type(torch.float32)  # [B, N]
        edge_mask = bb_mask[..., None] * bb_mask[..., None, :]  # [B, N, N]

        # Initial embeddings of positonal and relative indices.
        init_node_embed, init_edge_embed = self.embedding_layer(
            seq_idx=input_feats['seq_idx'],
            t=input_feats['t'],
            fixed_mask=fixed_mask,
            self_conditioning_ca=input_feats['sc_ca_t'],
        )  # [B, N, node_embed_size], [B, N, N, edge_embed_size]
        edge_embed = init_edge_embed * edge_mask[..., None]  # [B, N, N, edge_embed_size]
        node_embed = init_node_embed * bb_mask[..., None]  # [B, N, node_embed_size]

        # Run main network
        model_out = self.score_model(node_embed, edge_embed, input_feats)  
        # model_out: dict with keys:
        #   'psi': [B, N, 2] predicted psi angle (sin, cos)
        #   'rot_score': [B, N, 3, 3] predicted rotation score
        #   'trans_score': [B, N, 3] predicted translation score
        #   'final_rigids': [B, N] predicted Rigid objects

        # Psi angle prediction
        gt_psi = input_feats['torsion_angles_sin_cos'][..., 2, :]
        psi_pred = self._apply_mask(
            model_out['psi'], 
            gt_psi, 
            1 - fixed_mask[..., None]) # [B, N, 2];  if not fixed, use pred; if fixed, use gt.

        pred_out = {
            'psi': psi_pred,
            'rot_score': model_out['rot_score'],
            'trans_score': model_out['trans_score'],
        }
        rigids_pred = model_out['final_rigids']
        pred_out['rigids'] = rigids_pred.to_tensor_7()  # [B, N, 7]
        bb_representations = all_atom.compute_backbone(rigids_pred, psi_pred) # List of [B, N, 7] tensors
        pred_out['atom37'] = bb_representations[0].to(rigids_pred.device)
        pred_out['atom14'] = bb_representations[-1].to(rigids_pred.device)
        return pred_out

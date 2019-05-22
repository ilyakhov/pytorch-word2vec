import torch
from torch import nn


class CBOWHierSoftmax(nn.Module):
    def __init__(self, emb_count, emb_dim, cbow_op='sum', dtype=None):
        """

        :param emb_count:
        :param emb_dim:
        :param cbow_op: "sum"/"mean"
        :param dtype:
        """
        super(CBOWHierSoftmax, self).__init__()
        self.dtype = dtype
        self.cbow_op = cbow_op
        self.embeddings = nn.Embedding(
            num_embeddings=emb_count + 1,  # the last for PAD value
            embedding_dim=emb_dim,
            padding_idx=emb_count,  # with torch.no_grad(): for pad value
            max_norm=1.0,
            norm_type=2.0,
            scale_grad_by_freq=False,
            sparse=False)

    def forward(self, context, nodes, nodes_mask, turns_coeffs):
        # [b_size, window_size, emb_size]
        context_embeds = self.embeddings(
            context)

        if self.cbow_op == 'sum':
            # [b_size, 1, emb_size]
            cbow = torch.sum(context_embeds, dim=1, keepdim=True)
        elif self.cbow_op == 'mean':
            cbow = torch.mean(context_embeds, dim=1, keepdim=True)
        else:
            raise NotImplementedError(f'cbow for given: {self.cbow_op}')

        # [b_size, nodes_count, emb_size]
        nodes = self.embeddings(nodes)

        # [b_size, nodes_count]
        excitations = torch.sum(nodes * cbow, dim=2, keepdim=False)
        excitations = torch.sigmoid(turns_coeffs * excitations)
        excitations = excitations * nodes_mask
        excitations = torch.where(excitations == torch.zeros_like(excitations),
                                  torch.ones_like(excitations), excitations)
        # excitations = torch.clamp(excitations, min=1e-8, max=1-1e-8)
        excitations = torch.prod(excitations, dim=1, keepdim=False)

        # output = torch.clamp(excitations, min=1e-12, max=1-1e-12)
        loss = torch.mean(-1 * torch.log(excitations))
        return loss


class CBOWNegativeSampling(nn.Module):
    def __init__(self,
                 emb_count,
                 emb_dim,
                 neg_sampling_factor=20,
                 dtype=None,
                 device=None):
        super(CBOWNegativeSampling, self).__init__()
        self.dtype = dtype
        self.device = device
        self.emb_count = emb_count
        self.neg_sampling_factor = neg_sampling_factor
        self.i_embeddings = nn.Embedding(
            num_embeddings=emb_count,
            embedding_dim=emb_dim,
            max_norm=None,
            norm_type=2.0,
            scale_grad_by_freq=False,
            sparse=False)
        self.o_embeddings = nn.Embedding(
            num_embeddings=emb_count,
            embedding_dim=emb_dim,
            max_norm=None,
            norm_type=2.0,
            scale_grad_by_freq=False,
            sparse=False)

    def forward(self, context, target):
        b_size = context.size()[0]

        neg_samples = torch.randint(
            low=0, high=self.emb_count - 1,
            size=(b_size, 1, self.neg_sampling_factor),
            dtype=self.dtype, requires_grad=False, device=self.device
        ).squeeze()

        # [b_size, w_size, emb_size]
        context_embeds = self.o_embeddings(context)
        # [b_size, 1, emb_size]
        # cbow = torch.mean(context_embeds, dim=1, keepdim=True)
        # [b_size, 1, emb_size]
        target_embed = self.i_embeddings(target)
        t_shape = target_embed.shape
        target_embed = target_embed.view((t_shape[0], 1, t_shape[1]))
        # [b_size, emb_size, 1]
        target_embed = torch.transpose(target_embed, 1, 2)
        # [b_size, neg_count, emb_size]
        neg_context_embeds = self.o_embeddings(neg_samples).neg()

        pos = torch.bmm(context_embeds, target_embed)\
            .squeeze().sigmoid().log().mean(1)

        neg = torch.bmm(neg_context_embeds, target_embed)\
            .squeeze().sigmoid().log()\
            .view(-1, 1, self.neg_sampling_factor)\
            .sum(2).mean(1)

        loss = torch.mean(-1 * (pos + neg))

        return loss
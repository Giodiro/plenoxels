import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizerEMA(nn.Module):
    """
    https://github.com/zalandoresearch/pytorch-vq-vae/blob/master/vq-vae.ipynb
    https://github.com/deepmind/sonnet/blob/v1/sonnet/python/modules/nets/vqvae.py
    """
    def __init__(self, embedding_dim, num_embeddings, commitment_cost, decay, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon

        self.w = nn.Embedding(num_embeddings=self.num_embeddings, embedding_dim=self.embedding_dim)
        torch.nn.init.normal_(self.w.weight)
        self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('ema_w', self.w.weight.clone())

    def forward(self, inputs):
        flat_inputs = inputs.reshape(-1, self.embedding_dim)

        with torch.no_grad():
            distances = (torch.sum(flat_inputs**2, 1, keepdim=True)
                         - 2 * flat_inputs @ self.w.weight.T
                         + torch.sum(self.w.weight, 1))
            encoding_indices = torch.argmin(distances, 1)
            encodings = F.one_hot(encoding_indices, self.num_embeddings).float()  # Q, N

            quantized = self.w(encoding_indices)
            quantized = quantized.view(inputs.shape)

        if self.training:
            with torch.no_grad():
                self.ema_cluster_size -= (1 - self.decay) * (self.ema_cluster_size - torch.sum(encodings, 0))
                # Laplace smoothing of the cluster size
                n = torch.sum(self.ema_cluster_size)
                self.ema_cluster_size = (self.ema_cluster_size + self.epsilon) / (n + self.num_embeddings * self.epsilon) * n

                dw = encodings.T @ flat_inputs  # N, D
                self.ema_w -= (1 - self.decay) * (self.ema_w - dw)
                #self.ema_w = nn.Parameter(self.ema_w * self.decay + (1 - self.decay) * dw)
                normalised_ema_w = self.ema_w / self.ema_cluster_size.view(-1, 1)
                self.w.weight = nn.Parameter(normalised_ema_w)

        e_latent_loss = torch.mean(torch.square(quantized.detach() - inputs))
        loss = self.commitment_cost * e_latent_loss
        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        with torch.no_grad():
            avg_probs = torch.mean(encodings, dim=0)
            perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return loss, quantized, perplexity, encodings

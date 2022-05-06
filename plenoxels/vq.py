import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1 / self.num_embeddings, 1 / self.num_embeddings)
        self.commitment_cost = commitment_cost

    def forward(self, inputs):
        # Flatten input
        flat_input = inputs.view(-1, self.embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True)
                     + torch.sum(self.embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self.embedding.weight.T))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self.embedding.weight).view(inputs.shape)

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return loss, quantized, perplexity, encodings


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

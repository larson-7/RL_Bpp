import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical
import numpy as np
import math
from torch.nn import Parameter
# import graphviz
# import torchviz


class Attend2Pack(nn.Module):
    first_run = True
    def __init__(self,
                 input_size,
                 num_cases,
                 device='cuda',
                 embedding_dim=128,
                 num_heads=8,
                 hidden_dim=512,
                 num_attn_layers=3,
                 dropout=0.0,
                 bin_size=(20, 20, 20)):
        
        super().__init__()
        self.device = device
        self.bin_size = bin_size

        # case embedding network
        self.case_encoder = AttentionEncoder(input_size,
                                            device,
                                            embedding_dim,
                                            num_heads, 
                                            hidden_dim,
                                            num_attn_layers,
                                            dropout)
        
        # Frontier encoder
        self.frontier_encoder = FrontierEncoder(hidden_dim=embedding_dim, 
                                                width=bin_size[0],
                                                height=bin_size[1],
                                                device=device)
        self.sequence_policy = QueryBasedAttention(input_dim=embedding_dim, hidden_dim=hidden_dim, device=device)
        # # Sequence policy network
        # self.sequence_policy = Glimpse(d=embedding_dim,
        #                                M=num_heads,
        #                                N=num_cases,
        #                                C=10)  # Adjust clamping value as needed. This is a hyperparameter that we can tune.
    def forward(self, boxes, mask, observation, training=True):
        boxes_tensor = torch.tensor(boxes).float().to(self.device)
        box_embeddings = self.case_encoder(boxes_tensor)
        frontier_embedding = self.frontier_encoder(observation)
        # Sequence selection
        q_s_bar = torch.stack((box_embeddings.mean(0), frontier_embedding)).mean(0)
        # mask scores of already selected
        mask_tensor = torch.tensor(mask, device=self.device, requires_grad=False)
        logits = self.sequence_policy(inputs=box_embeddings, query=q_s_bar, mask=mask_tensor)
        # logits = self.sequence_policy.forward(b=box_embeddings, q_s_bar=q_s_bar, mask=bool_mask)
        return logits

    def get_action_and_value(self, boxes, mask, obs):
        logits = self.forward(boxes, mask, obs)
        probs = Categorical(logits=logits)
        action = probs.sample()
        return action.item(), probs.log_prob(action), probs.entropy()

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self,
                input_size=128,
                output_size=128,
                device='cuda',
                hidden_dim=512,
                num_heads=8,
                dropout=0.0):
        """
        Initializes the MultiHeadAttentionLayer.

        :param device: The device to run the computations on (e.g., 'cpu', 'cuda').
        :param input_dim: The dimensionality of the input.
        :param num_heads: The number of attention heads.
        :param hidden_dim: The dimensionality of the hidden layer in the feedforward network.
        :param dropout: Dropout rate.
        """
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden__dim must be divisible by num_heads"
        
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dim_feedforward = self.hidden_dim * 4 # following attention is all you need structure
        self.input_size = input_size
        self.output_size = output_size
        self.device = device
        self.dropout = dropout

        # first normalization layer
        self.norm1 = nn.LayerNorm(self.input_size)

        # Multi-head attention layer        
        self.multiHeadAttn = nn.MultiheadAttention(embed_dim=self.input_size,
                                                    num_heads=self.num_heads,
                                                    dropout=self.dropout,
                                                    batch_first=True,
                                                    device=self.device)
        
        # second normalization layer
        self.norm2 = nn.LayerNorm(self.input_size)

        # Feedforward network layers
        self.ffn = nn.Sequential(
            nn.Linear(self.input_size, self.dim_feedforward),
            nn.ReLU(),
            nn.Linear(self.dim_feedforward, self.input_size)
        )

    def feedforward_layer(self, inputs):
        """
        Applies the feedforward network with a residual connection and layer normalization.

        :param inputs: A float32 tensor of shape (B, N, embedded_dim).
        :returns: A float32 tensor of shape (B, N, embedded_dim).
        """
        sublayer = self.ffn(self.norm2(inputs))
        outputs = inputs + sublayer
        return outputs

    def forward(self, inputs):
        """
        Applies multi-head attention followed by the feedforward network.

        :param inputs: A PyTorch tensor of shape (B, N, case_dim).
        :returns: Embedded cases of shape (B, N, embedding_dim).
        """
        # Multi-head attention
        normed_inputs = self.norm1(inputs)
        attn_output, attn_output_weights = self.multiHeadAttn(normed_inputs,
                                                              normed_inputs,
                                                              normed_inputs)
        outputs = self.feedforward_layer(attn_output + inputs)  # Feedforward network
        return outputs

class Attention(nn.Module):
    """
    Attention model for Pointer-Net
    """

    def __init__(self, input_dim,
                 hidden_dim):
        """
        Initiate Attention

        :param int input_dim: Input's diamention
        :param int hidden_dim: Number of hidden units in the attention
        """

        super(Attention, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.input_linear = nn.Linear(input_dim, hidden_dim)
        self.context_linear = nn.Conv1d(input_dim, hidden_dim, 1, 1)
        self.V = Parameter(torch.FloatTensor(hidden_dim), requires_grad=True)
        self._inf = Parameter(torch.FloatTensor([float('-inf')]), requires_grad=False)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()

        # Initialize vector V
        nn.init.uniform(self.V, -1, 1)

    def forward(self, input,
                context,
                mask):
        """
        Attention - Forward-pass

        :param Tensor input: Hidden state h
        :param Tensor context: Attention context
        :param ByteTensor mask: Selection mask
        :return: tuple of - (Attentioned hidden state, Alphas)
        """

        # (batch, hidden_dim, seq_len)
        inp = self.input_linear(input).unsqueeze(2).expand(-1, -1, context.size(1))

        # (batch, hidden_dim, seq_len)
        context = context.permute(0, 2, 1)
        ctx = self.context_linear(context)

        # (batch, 1, hidden_dim)
        V = self.V.unsqueeze(0).expand(context.size(0), -1).unsqueeze(1)

        # (batch, seq_len)
        att = torch.bmm(V, self.tanh(inp + ctx)).squeeze(1)
        if len(att[mask]) > 0:
            att[mask] = self.inf[mask]
        alpha = self.softmax(att)

        hidden_state = torch.bmm(ctx, alpha.unsqueeze(2)).squeeze(2)

        return hidden_state, alpha

    def init_inf(self, mask_size):
        self.inf = self._inf.unsqueeze(1).expand(*mask_size)
class AttentionEncoder(nn.Module):
    """
    Encodes N cases into the embedded case dimension space using multiple attention layers.
    """

    def __init__(self, input_size, device, embedding_dim=128, num_heads=8, hidden_dim=512,
                 num_attn_layers=3, dropout=0.0):
        """
        Initializes the AttentionEncoder.

        :param input_size: The size of the input, which equals the number of dimensions that describe each case.
        :param device: The device to run the computations on (e.g., 'cpu', 'cuda').
        :param embedding_dim: The dimensionality of the embeddings.
        :param num_heads: The number of Transformer heads to use.
        :param hidden_dim: The dimensionality of the hidden layer in the feedforward network.
        :param num_attn_layers: The number of attention layers.
        :param dropout: Dropout rate.
        """
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.input_size = input_size
        self.device = device
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.num_attn_layers = num_attn_layers
        self.dropout = dropout

        # Linear layer to project input to embedding dimension
        self.embeddingL = nn.Linear(self.input_size, self.embedding_dim).to(self.device)

        # List of multi-head attention layers
        self.attn_layers = nn.ModuleList([
            MultiHeadAttentionLayer(input_size=self.embedding_dim,
                                    output_size=self.embedding_dim,
                                    device=self.device,
                                    hidden_dim=self.hidden_dim,
                                    num_heads=self.num_heads,
                                    dropout=self.dropout)
            for _ in range(self.num_attn_layers)
        ]).to(self.device)

    def forward(self, inputs):
        """
        Projects the inputs to the embedding dimension and applies multiple attention layers.

        :param inputs: A PyTorch tensor of shape (B, N, case_dim).
        :returns: Embedded cases of shape (B, N, embedding_dim).
        """
        x = self.embeddingL(inputs)  # Project input to embedding dimension
        x = x.unsqueeze(0)
        for attn_layer in self.attn_layers:
            x = attn_layer(x)  # Apply each attention layer sequentially
        x = x.squeeze(0)
        return x


class FrontierEncoder(nn.Module):
    def __init__(self, hidden_dim=128, width=10, height=10, in_channels=2, out_channels=8, kernel_size=3, stride=1, padding=1, kernel_size_2=5, stride_2=1, padding_2=1, device='cuda'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.width, self.height = width, height
        self.in_channels, self.out_channels = in_channels, out_channels
        self.kernel_size, self.stride, self.padding = kernel_size, stride, padding
        self.kernel_size_2, self.stride_2, self.padding_2 = kernel_size_2, stride_2, padding_2
        self.device = device
        self.width_1 = (self.width - self.kernel_size + 2 * self.padding) // self.stride + 1
        self.height_1 = (self.height - self.kernel_size + 2 * self.padding) // self.stride + 1
        self.width_2 = (self.width_1 - self.kernel_size_2 + 2 * self.padding_2) // self.stride_2 + 1
        self.height_2 = (self.height_1 - self.kernel_size_2 + 2 * self.padding_2) // self.stride_2 + 1
        self.previous_frontier = np.zeros((self.width_1, self.height_1), dtype=np.float32)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, device=self.device),
            nn.ReLU(),
            nn.LayerNorm([self.out_channels, self.width_1, self.height_1])
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=self.kernel_size_2, stride=self.stride_2, padding=self.padding_2, device=self.device),
            nn.ReLU(),
            nn.LayerNorm([self.out_channels, self.width_2, self.height_2])
        )

        self.fc_layer = nn.Linear(in_features=self.out_channels * self.width_2 * self.height_2, out_features=self.hidden_dim)
        self.relu = nn.ReLU()
        self.norm = nn.LayerNorm(self.hidden_dim)


    def forward(self, frontier):
        # concatenate previous frontier
        front1 = torch.tensor(self.previous_frontier).to(self.device)
        front2 = torch.tensor(frontier).float().to(self.device)
        frontiers = torch.stack((front1, front2)).to(self.device)
        x = self.conv2(self.conv1(frontiers)) # in_channels x W x H -> out_channels x W x H
        x = x.view(-1) # flatten
        x = self.norm(self.relu(self.fc_layer(x))) # in_features -> out_features
        self.previous_frontier = frontier
        return x

class QueryBasedAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, device, clamp_value = 10):
        super(QueryBasedAttention, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.device = device
        # Learnable parameters
        self.W_query = nn.Linear(input_dim, hidden_dim, bias=False).to(self.device)
        self.W_key = nn.Linear(input_dim, hidden_dim, bias=False).to(self.device)
        self.W_value = nn.Linear(input_dim, hidden_dim, bias=False).to(self.device)
        self.W_out = nn.Linear(hidden_dim, 1, bias=False).to(self.device)
        self.C = clamp_value

        # Scaling factor
        self.scale = torch.sqrt(torch.FloatTensor([hidden_dim])).to(self.device)

    def forward(self, inputs, query, mask):
        # inputs: [seq_len, input_dim]
        # query: [input_dim]
        # mask: [seq_len]

        # Compute query, key, and value
        Q = self.W_query(query).unsqueeze(0)  # [1, hidden_dim]
        K = self.W_key(inputs)  # [seq_len, hidden_dim]
        V = self.W_value(inputs)  # [seq_len, hidden_dim]

        # Compute attention scores
        attention_scores = torch.matmul(Q, K.transpose(0, 1)) / self.scale  # [1, seq_len]

        # Apply mask
        attention_scores = attention_scores.masked_fill(mask.unsqueeze(0) == 0, float('-inf'))

        # Compute attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)  # [1, seq_len]

        # Compute weighted sum of values
        attended_values = attention_weights.transpose(0, 1) * V  # [seq_len, hidden_dim]

        # Compute logits
        logits = self.W_out(attended_values).squeeze(-1)  # [seq_len]
        logits = self.C * torch.tanh(logits) # [N]

        return logits

class Glimpse(nn.Module):
    def __init__(self, d, M, N, C):
        super(Glimpse, self).__init__()
        assert d % M == 0  # d should be divisible by M
        self.d = d  # embedding dimension
        self.M = M  # number of heads
        self.N = N  # number of cases
        self.C = C  # clamping value
        self.h = d // M  # dimension of each head
        
        # Learnable projections
        self.W_k_bar = nn.Parameter(torch.Tensor(M, self.h, d)) # [M, h, d]
        self.W_v_bar = nn.Parameter(torch.Tensor(M, self.h, d)) # [M, h, d]
        self.W_k = nn.Parameter(torch.Tensor(d, d)) # [d, d]
        self.W_q = nn.Parameter(torch.Tensor(d, d)) # [d, d]
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_k_bar)
        nn.init.xavier_uniform_(self.W_v_bar)
        nn.init.xavier_uniform_(self.W_k)
        nn.init.xavier_uniform_(self.W_q)
# ?TODO: This is weird, lets start with a basic attention module from assignment 4 we can easily employ the mask.
    def forward(self, b, q_s_bar, mask):
        # b: case embeddings [N, d]
        # q_s_bar: sequence query vector [d]

        # Compute glimpse keys, values, and logit keys
        k_bar = torch.matmul(self.W_k_bar, b.t()).permute(2, 0, 1)  # [M, h, d] @ [d, N] -> [M, h, N] -> [N, M, h]
        v_bar = torch.matmul(self.W_v_bar, b.t()).permute(2, 0, 1)  # [M, h, d] @ [d, N] -> [M, h, N] -> [N, M, h]
        k = torch.matmul(self.W_k, b.t()).t()  # [d, d] @ [d, N] -> [d, N] -> [N, d]
        
        # Split query into M heads
        q_s_bar = q_s_bar.view(self.M, self.h)  # [d] -> [M, h]
        
        # Compute compatibility vector
        c_bar = torch.matmul(k_bar, q_s_bar.t()) / (self.h ** 0.5)  # [N, M, h] @ [M, h] -> [N, M, M]
        # mask scores of already selected
        inverted_mask = ~mask
        cur_mask = inverted_mask.unsqueeze(1).unsqueeze(2)
        attention_scores = c_bar.masked_fill_(cur_mask, float('-inf'))
        # Compute updated sequence query
        attn_weights = F.softmax(attention_scores, dim=0)  # [N, M, M]
        q_s = torch.matmul(attn_weights, v_bar).view(self.N, -1).sum(0)  # [N, M, M] @ [N, M, h] -> [N, M, h] -> [N, d] -> [d]
        
        # Compute final compatibility vector
        q_s_product = torch.matmul(self.W_q, q_s) # [d, d] @ [d] -> [d]
        c = torch.matmul(k, q_s_product) / (self.d ** 0.5)  # [N, d] @ [d] -> [N]
        c = self.C * torch.tanh(c) # [N]
        c = c.masked_fill_(inverted_mask, float('-inf'))
        return c


class Attention(nn.Module):
    def __init__(self, hidden_size, use_tanh=False, C=10, name='Bahdanau', device="cuda"):
        super(Attention, self).__init__()

        self.use_tanh = use_tanh
        self.C = C
        self.name = name

        if name == 'Bahdanau':
            self.W_query = nn.Linear(hidden_size, hidden_size)
            self.W_ref = nn.Conv1d(hidden_size, hidden_size, 1, 1)

            V = torch.FloatTensor(hidden_size)
            V = V.to(device)
            self.V = nn.Parameter(V)
            self.V.data.uniform_(-(1. / math.sqrt(hidden_size)), 1. / math.sqrt(hidden_size))

    def forward(self, query, ref):
        """
        Args:
            query: [batch_size x hidden_size]
            ref:   [batch_size x seq_len x hidden_size]
        """

        batch_size = ref.size(0)
        seq_len = ref.size(1)

        if self.name == 'Bahdanau':
            ref = ref.permute(0, 2, 1)
            query = self.W_query(query).unsqueeze(2)  # [batch_size x hidden_size x 1]
            ref = self.W_ref(ref)  # [batch_size x hidden_size x seq_len]
            expanded_query = query.repeat(1, 1, seq_len)  # [batch_size x hidden_size x seq_len]
            V = self.V.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1)  # [batch_size x 1 x hidden_size]
            logits = torch.bmm(V, F.tanh(expanded_query + ref)).squeeze(1)

        elif self.name == 'Dot':
            query = query.unsqueeze(2)
            logits = torch.bmm(ref, query).squeeze(2)  # [batch_size x seq_len x 1]
            ref = ref.permute(0, 2, 1)

        else:
            raise NotImplementedError

        if self.use_tanh:
            logits = self.C * F.tanh(logits)
        else:
            logits = logits
        return ref, logits

if __name__ == "__main__":
    # Example usage:
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    batch_size = 1
    n_cases = 10
    input_dim = 4  # Size of each input vector
    model = AttentionEncoder(input_dim, device=device)

    # Input tensor of shape (batch_size, seq_len, input_dim)
    x = torch.randn(batch_size, n_cases, input_dim).to(device)
    sorted_inputs = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {sorted_inputs.shape}")
import argparse
import torch
import numpy as np
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn as nn

parser = argparse.ArgumentParser(description="train.py")

parser.add_argument("-data", default="./data",
    help="Path to training and validating data")

parser.add_argument("-checkpoint", default="./data/checkpoint.pt",
    help="The saved checkpoint")

parser.add_argument("-prefix", default="exp", help="Prefix of trjfile")

parser.add_argument("-pretrained_embedding", default=None,
    help="Path to the pretrained word (cell) embedding")

parser.add_argument("-num_layers", type=int, default=3,
    help="Number of layers in the RNN cell")

parser.add_argument("-bidirectional", type=bool, default=False,
    help="True if use bidirectional rnn in encoder")

parser.add_argument("-hidden_size", type=int, default=256,
    help="The hidden state size in the RNN cell")

parser.add_argument("-embedding_size", type=int, default=256,
    help="The word (cell) embedding size")

parser.add_argument("-dropout", type=float, default=0.2,
    help="The dropout probability")

parser.add_argument("-max_grad_norm", type=float, default=5.0,
    help="The maximum gradient norm")

parser.add_argument("-learning_rate", type=float, default=0.001)

parser.add_argument("-batch", type=int, default=64,
    help="The batch size")

parser.add_argument("-generator_batch", type=int, default=32,
    help="""The maximum number of words to generate each time.
    The higher value, the more memory requires.""")

parser.add_argument("-t2vec_batch", type=int, default=256,
    help="""The maximum number of trajs we encode each time in t2vec""")

parser.add_argument("-start_iteration", type=int, default=0)

parser.add_argument("-epochs", type=int, default=30,
    help="The number of training epochs")

parser.add_argument("-print_freq", type=int, default=50,
    help="Print frequency")

parser.add_argument("-save_freq", type=int, default=1000,
    help="Save frequency")

parser.add_argument("-cuda", type=bool, default=False,
    help="True if we use GPU to train the model")

parser.add_argument("-use_discriminative", action="store_true",
    help="Use the discriminative loss if the argument is given")

parser.add_argument("-discriminative_w", type=float, default=0.1,
    help="discriminative loss weight")

parser.add_argument("-criterion_name", default="NLL",
    help="NLL (Negative Log Likelihood) or KLDIV (KL Divergence)")

parser.add_argument("-knearestvocabs", default=None,
    help="""The file of k nearest cells and distances used in KLDIVLoss,
    produced by preprocessing, necessary if KLDIVLoss is used""")

parser.add_argument("-dist_decay_speed", type=float, default=0.8,
    help="""How fast the distance decays in dist2weight, a small value will
    give high weights for cells far away""")

parser.add_argument("-max_num_line", type=int, default=20000000)

parser.add_argument("-max_length", default=200,
    help="The maximum length of the target sequence")

parser.add_argument("-mode", type=int, default=0,
    help="Running mode (0: train, 1:evaluate, 2:t2vec)")

parser.add_argument("-vocab_size", type=int, default=0,
    help="Vocabulary Size")

parser.add_argument("-bucketsize", default=[(20,30),(30,30),(30,50),(50,50),(50,70),(70,70),(70,100),(100,100)],
    help="Bucket size for training")

args = parser.parse_args()

print(args)

def submit(m0, traj, h0=None): #a quick version of t2vec
    srcdata = [int(item) for item in traj]
    srcdata = np.array(srcdata)
    srcdata = [srcdata]
    #print('srcdata', srcdata)
    embed = m0.embedding(torch.LongTensor(srcdata).reshape(-1,1))
    #embed = m0.embedding(torch.LongTensor(srcdata).reshape(-1,1))
    if (not h0 is None) and (torch.cuda.is_available()):
        h0 = h0.cuda()
        #print('h0 by cuda')
    if (not embed is None) and (torch.cuda.is_available()):
        embed = embed.cuda()
        #print('embed by cuda')
    output, hn = m0.encoder.rnn(embed, h0)
    output = output.transpose(0, 1).contiguous()
#    print('hn', hn.size())
#    print('output', output.size())
    return hn.cpu().data, output.cpu().data

def model_init(args):
    "read source sequences from trj.t and write the tensor into file trj.h5"
    m0 = EncoderDecoder(args.vocab_size, args.embedding_size,
                        args.hidden_size, args.num_layers,
                        args.dropout, args.bidirectional)
    if os.path.isfile(args.checkpoint):
        print("=> loading checkpoint '{}'".format(args.checkpoint))
        checkpoint = torch.load(args.checkpoint,map_location='cpu')
        m0.load_state_dict(checkpoint["m0"])
        if torch.cuda.is_available():
            m0.cuda()
        m0.eval()
    else:
        print("=> no checkpoint found at '{}'".format(args.checkpoint))
    return m0

import os

PAD = 0
BOS = 1
EOS = 2
UNK = 3

PAD_WORD = '<blank>'
BOS_WORD = '<s>'
EOS_WORD = '</s>'
UNK_WORD = '<unk>'

class StackingGRUCell(nn.Module):
    """
    Multi-layer CRU Cell
    """
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(StackingGRUCell, self).__init__()
        self.num_layers = num_layers
        self.grus = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)

        self.grus.append(nn.GRUCell(input_size, hidden_size))
        for i in range(1, num_layers):
            self.grus.append(nn.GRUCell(hidden_size, hidden_size))

    def forward(self, input, h0):
        """
        Input:
        input (batch, input_size): input tensor
        h0 (num_layers, batch, hidden_size): initial hidden state
        ---
        Output:
        output (batch, hidden_size): the final layer output tensor
        hn (num_layers, batch, hidden_size): the hidden state of each layer
        """
        hn = []
        output = input
        for i, gru in enumerate(self.grus):
            hn_i = gru(output, h0[i])
            hn.append(hn_i)
            if i != self.num_layers - 1:
                output = self.dropout(hn_i)
            else:
                output = hn_i
        hn = torch.stack(hn)
        return output, hn

class GlobalAttention(nn.Module):
    """
    $$a = \sigma((W_1 q)H)$$
    $$c = \tanh(W_2 [a H, q])$$
    """
    def __init__(self, hidden_size):
        super(GlobalAttention, self).__init__()
        self.L1 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.L2 = nn.Linear(2*hidden_size, hidden_size, bias=False)
        self.softmax = nn.Softmax()
        self.tanh = nn.Tanh()

    def forward(self, q, H):
        """
        Input:
        q (batch, hidden_size): query
        H (batch, seq_len, hidden_size): context
        ---
        Output:
        c (batch, hidden_size)
        """
        # (batch, hidden_size) => (batch, hidden_size, 1)
        q1 = self.L1(q).unsqueeze(2)
        # (batch, seq_len)
        a = torch.bmm(H, q1).squeeze(2)
        a = self.softmax(a)
        # (batch, seq_len) => (batch, 1, seq_len)
        a = a.unsqueeze(1)
        # (batch, hidden_size)
        c = torch.bmm(a, H).squeeze(1)
        # (batch, hidden_size * 2)
        c = torch.cat([c, q], 1)
        return self.tanh(self.L2(c))

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout,
                       bidirectional, embedding):
        """
        embedding (vocab_size, input_size): pretrained embedding
        """
        super(Encoder, self).__init__()
        self.num_directions = 2 if bidirectional else 1
        assert hidden_size % self.num_directions == 0
        self.hidden_size = hidden_size // self.num_directions
        self.num_layers = num_layers

        self.embedding = embedding
        self.rnn = nn.GRU(input_size, self.hidden_size,
                          num_layers=num_layers,
                          bidirectional=bidirectional,
                          dropout=dropout)

    def forward(self, input, lengths, h0=None):
        """
        Input:
        input (seq_len, batch): padded sequence tensor
        lengths (1, batch): sequence lengths
        h0 (num_layers*num_directions, batch, hidden_size): initial hidden state
        ---
        Output:
        hn (num_layers*num_directions, batch, hidden_size):
            the hidden state of each layer
        output (seq_len, batch, hidden_size*num_directions): output tensor
        """
        # (seq_len, batch) => (seq_len, batch, input_size)
        embed = self.embedding(input)
        lengths = lengths.data.view(-1).tolist()
        if lengths is not None:
            embed = pack_padded_sequence(embed, lengths)
        # print(embed, h0)
        output, hn = self.rnn(embed, h0)
        if lengths is not None:
            output = pad_packed_sequence(output)[0]
        return hn, output

class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, embedding):
        super(Decoder, self).__init__()
        self.embedding = embedding
        self.rnn = StackingGRUCell(input_size, hidden_size, num_layers,
                                   dropout)
        self.attention = GlobalAttention(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers

    def forward(self, input, h, H, use_attention=True):
        """
        Input:
        input (seq_len, batch): padded sequence tensor
        h (num_layers, batch, hidden_size): input hidden state
        H (seq_len, batch, hidden_size): the context used in attention mechanism
            which is the output of encoder
        use_attention: If True then we use attention
        ---
        Output:
        output (seq_len, batch, hidden_size)
        h (num_layers, batch, hidden_size): output hidden state,
            h may serve as input hidden state for the next iteration,
            especially when we feed the word one by one (i.e., seq_len=1)
            such as in translation
        """
        assert input.dim() == 2, "The input should be of (seq_len, batch)"
        # (seq_len, batch) => (seq_len, batch, input_size)
        embed = self.embedding(input)
        output = []
        # split along the sequence length dimension
        for e in embed.split(1):
            e = e.squeeze(0) # (1, batch, input_size) => (batch, input_size)
            o, h = self.rnn(e, h)
            if use_attention:
                o = self.attention(o, H.transpose(0, 1))
            o = self.dropout(o)
            output.append(o)
        output = torch.stack(output)
        return output, h

class EncoderDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_size,
                       hidden_size, num_layers, dropout, bidirectional):
        super(EncoderDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        ## the embedding shared by encoder and decoder
        self.embedding = nn.Embedding(vocab_size, embedding_size,
                                      padding_idx=PAD)
        self.encoder = Encoder(embedding_size, hidden_size, num_layers,
                               dropout, bidirectional, self.embedding)
        self.decoder = Decoder(embedding_size, hidden_size, num_layers,
                               dropout, self.embedding)
        self.num_layers = num_layers

    def load_pretrained_embedding(path):
        if os.path.isfile(path):
            w = torch.load(path)
            self.embedding.weight.data.copy_(w)

    def encoder_hn2decoder_h0(self, h):
        """
        Input:
        h (num_layers * num_directions, batch, hidden_size): encoder output hn
        ---
        Output:
        h (num_layers, batch, hidden_size * num_directions): decoder input h0
        """
        if self.encoder.num_directions == 2:
            num_layers, batch, hidden_size = h.size(0)//2, h.size(1), h.size(2)
            return h.view(num_layers, 2, batch, hidden_size)\
                    .transpose(1, 2).contiguous()\
                    .view(num_layers, batch, hidden_size * 2)
        else:
            return h

    def forward(self, src, lengths, trg):
        """
        Input:
        src (src_seq_len, batch): source tensor
        lengths (1, batch): source sequence lengths
        trg (trg_seq_len, batch): target tensor, the `seq_len` in trg is not
            necessarily the same as that in src
        ---
        Output:
        output (trg_seq_len, batch, hidden_size)
        """
        encoder_hn, H = self.encoder(src, lengths)
        decoder_h0 = self.encoder_hn2decoder_h0(encoder_hn)
        ## for target we feed the range [BOS:EOS-1] into decoder
        output, decoder_hn = self.decoder(trg[:-1], decoder_h0, H)
        return output    

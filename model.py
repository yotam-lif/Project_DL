import torch
import torch.nn as nn

AA_num = 20
AA_props = 6
Nuc_num = 4
Nuc_props = 2
DBD_nums = 26


class FCN(nn.Module):

    def __init__(self, N_input, N_hidden, N_output, dropout=False, dropout_p=0.1):
        super().__init__()

        assert len(N_hidden) > 0, "Pass list of hidden layer sizes for N_hidden"

        self.node_network = nn.Sequential(
            nn.Linear(N_input, N_hidden[0]),
            nn.ReLU()
        )

        if len(N_hidden) > 1:
            for i in range(1, len(N_hidden)):
                self.node_network.append(nn.BatchNorm1d(N_hidden[i - 1])),
                if dropout:
                    self.node_network.append(nn.Dropout(p=dropout_p))
                self.node_network.append(nn.Linear(N_hidden[i - 1], N_hidden[i])),
                self.node_network.append(nn.ReLU())
        self.node_network.append(nn.Linear(N_hidden[-1], N_output))

    def forward(self, x):
        return self.node_network(x)


class EncoderNet(nn.Module):
    def __init__(
            self,
            n_layers,
            d_model,
            n_heads,
            dropout_enc,
            window_size,
            dropout_emb
    ):
        super().__init__()

        self.AA_emb = FCN(AA_num + AA_props, [d_model * 2], d_model, dropout=dropout_emb)
        self.N_emb = FCN(Nuc_num + Nuc_props, [d_model * 2], d_model, dropout=dropout_emb)
        self.DBD_emb = FCN(DBD_nums, [d_model * 2], d_model, dropout=dropout_emb)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dropout=dropout_enc,
                                                   batch_first=True)
        self.enc = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=n_layers)

        self.red_emb = FCN(d_model, [int(d_model / 2)], 1)
        self.out = FCN(1803, [1000], window_size)

    def forward(self, inp):
        tf = inp[0]
        dna = inp[1]
        dbd = inp[2]
        embedded_inp = []

        # start by appending embeddings of AA's (# = 1503)
        for i in range(len(tf)):
            i_list = []
            tmp = self.DBD_emb(dbd[i])
            i_list.append(tmp.tolist())
            for aa in tf[i]:
                tmp = self.AA_emb(aa)
                i_list.append(tmp.tolist())
            for nuc in dna[i]:
                tmp = self.N_emb(nuc)
                i_list.append(tmp.tolist())
            embedded_inp.append(i_list)

        embedded_inp = torch.tensor(embedded_inp)
        # Now we have torch tensor of dims [batch size, length of TF + Nucleotide window size + 1 (DBD), d_model]
        # Pass through encoder and out layer
        post_enc = self.enc(embedded_inp)
        red = self.red_emb(post_enc)
        red = torch.squeeze(red)
        out = self.out(red)
        return out

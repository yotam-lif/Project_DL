import torch
import torch.nn as nn

AA_num = 20
AA_props = 6
Nuc_num = 4
Nuc_props = 2
DBD_nums = 26
seq_len = 1502


class FCN(nn.Module):

    def __init__(self, N_layers, dropout=False, dropout_p=0.1, bnorm=False):
        super().__init__()
        self.node_network = nn.Sequential()
        num = len(N_layers)
        for n in range(num-1):
            layer = nn.Linear(N_layers[n], N_layers[n+1])
            # Zero initialization
            # nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
            # nn.init.zeros_(layer.bias)
            #
            self.node_network.append(layer)
            if bnorm:
                self.node_network.append(nn.BatchNorm1d(N_layers[n+1])),
            if dropout:
                self.node_network.append(nn.Dropout(p=dropout_p))
            self.node_network.append(nn.ReLU())

    def forward(self, x):
        return self.node_network(x)


class LinNet(nn.Module):
    def __init__(
            self,
            N_emb,
            N_hidden,
            window_size
    ):
        super().__init__()

        self.window_size = window_size
        self.AA_emb = FCN(N_layers=N_emb)
        self.N_emb = FCN(N_layers=N_emb)
        self.DBD_emb = FCN(N_layers=N_emb)

        self.reduced = FCN(N_layers=N_hidden, dropout=True)

    def forward(self, inp):
        tf_ei = inp[:, :seq_len, :]
        tf_ei = self.AA_emb(tf_ei)

        dna_ei = inp[:, seq_len:seq_len + self.window_size, :]
        dna_ei = self.N_emb(dna_ei)

        dbd_ei = inp[:, -1, :]
        # add dim
        dbd_ei = dbd_ei[:, None, :]
        dbd_ei = self.DBD_emb(dbd_ei)

        embedded_inp = torch.cat((tf_ei, dna_ei, dbd_ei), dim=1)
        # Now we have torch tensor of dims [batch size, length of TF + Nucleotide window size + 1 (DBD), 1]
        embedded_inp = torch.squeeze(embedded_inp)
        # Now we have torch tensor of dims [batch size, length of TF + Nucleotide window size + 1 (DBD)]
        out = self.reduced(embedded_inp)
        return out


class EncoderNet(nn.Module):
    def __init__(
            self,
            n_layers,
            d_model,
            n_heads,
            window_size
    ):
        super().__init__()

        self.AA_emb = FCN(N_layers=[26, 128, d_model])
        self.N_emb = FCN(N_layers=[26, 128, d_model])
        self.DBD_emb = FCN(N_layers=[26, 128, d_model])

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, batch_first=True)
        self.enc = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=n_layers)

        self.red_emb = FCN(N_layers=[d_model, 64, 1], dropout=True)
        self.out = FCN(N_layers=[seq_len+1+window_size, 1200, 600, window_size], dropout=True, bnorm=True)
        self.window_size = window_size
        self.d_model = d_model

    def forward(self, inp):
        tf_ei = inp[:, :seq_len, :]
        tf_ei = self.AA_emb(tf_ei)

        dna_ei = inp[:, seq_len:seq_len + self.window_size, :]
        dna_ei = self.N_emb(dna_ei)

        dbd_ei = inp[:, -1, :]
        # add dim
        dbd_ei = dbd_ei[:, None, :]
        dbd_ei = self.DBD_emb(dbd_ei)

        embedded_inp = torch.cat((tf_ei, dna_ei, dbd_ei), dim=1)
        # Now we have torch tensor of dims [batch size, length of TF + Nucleotide window size + 1 (DBD), d_model]
        # Pass through encoder and out layer
        post_enc = self.enc(embedded_inp)
        red = self.red_emb(post_enc)
        red = torch.squeeze(red)
        out = self.out(red)
        return out

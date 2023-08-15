import torch
import numpy as np
import torch.nn as nn
import math
import dataloader

AA_num = 20
AA_props = 6
Nuc_num = 4
Nuc_props = 3
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

        self.AA_embedding = FCN(AA_num + AA_props, [d_model * 2], d_model, dropout=dropout_emb)
        self.Nuc_embedding = FCN(Nuc_num + Nuc_props, [d_model * 2], d_model, dropout=dropout_emb)
        self.DBD_embedding = FCN(DBD_nums, [d_model * 2], d_model, dropout=dropout_emb)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dropout=dropout_enc)
        self.enc = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=n_layers)

        self.out = nn.Linear(d_model, window_size)

    def forward(self, inp: (dataloader.Nucleotide, dataloader.TF)):
        # Src size must be (batch_size, src sequence length)
        # Embedding + positional encoding - Out size = (batch_size, sequence length, dim_model)
        nuc_list = inp[0]
        tf = inp[1]
        embedded_inp = []

        # start by appending embeddings of AA's (# = 1503)
        for i in range(tf.seq_len):
            tmp = tf.seq[i]

        # now append embeddings of nucleic acids (# = window size)
        for n in nuc_list:
            x = self.Nuc_embedding(n)
            embedded_inp.append(x)

        # add DBD
        x = self.DBD_embedding(tf.DBD)
        embedded_inp.append(x)
        embedded_inp = torch.FloatTensor(embedded_inp)
        # Now we have torch tensor of dims [length of TF + Nucleotide window size + 1 (DBD), d_model]
        # Pass through encoder and out layer
        intermediate = self.enc(embedded_inp)
        out = self.out(intermediate)
        # we permute to obtain size (sequence length, batch_size, dim_model),
        out = out.permute(1, 0, 2)

        return out

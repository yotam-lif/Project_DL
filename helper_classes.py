import torch
import torch.nn as nn
import numpy as np

AA_nums = 20
DBD_nums = 26
DNA_window = 300
NUC_nums = 4


class TF:
    def __init__(self, arr):

        self.seq_len = len(arr[0]) - 1
        # self.data = torch.from_numpy(arr[:][0:self.seq_len])
        self.disorder_score = torch.from_numpy(arr[1][:self.seq_len])
        self.molw = torch.from_numpy(arr[2][:self.seq_len])
        self.res_molw = torch.from_numpy(arr[3][:self.seq_len])
        self.pka = torch.from_numpy(arr[4][:self.seq_len])
        self.pkb = torch.from_numpy(arr[5][:self.seq_len])
        self.iso = torch.from_numpy(arr[6][:self.seq_len])
        self.seq = np.zeros((self.seq_len, AA_nums))
        # self.seq 1st axis is position, 2nd is one-hot for AA
        # Create one-hot representation, -1 vacancy tokens are left as zero arrays
        for i in range(self.seq_len):
            ind = int(arr[0][i])
            if ind >= 0:
                self.seq[i][ind - 1] = 1
        self.seq = torch.from_numpy(self.seq)

        self.DBD = np.zeros(DBD_nums)
        dbd_enc = int(arr[0][-1]) - 1
        self.DBD[dbd_enc] = 1

    def getData(self):
        data = torch.cat((self.seq,
                          self.disorder_score.view(-1, 1)
                          , self.molw.view(-1, 1)
                          , self.res_molw.view(-1, 1)
                          , self.pka.view(-1, 1)
                          , self.pkb.view(-1, 1)
                          , self.iso.view(-1, 1))
                         , dim=1)
        return data, self.DBD


class DNA_fragment:
    def __init__(self, arr):

        self.props = len(arr)
        self.seq_len = len(arr[0])
        self.pka = torch.from_numpy(arr[1][0:self.seq_len])
        self.molw = torch.from_numpy(arr[2][0:self.seq_len])
        self.signal = torch.from_numpy(arr[3][0:self.seq_len])

        self.seq = np.zeros((self.seq_len, NUC_nums))
        # self.seq 1st axis is position in IDR, 2nd is one-hot for AA
        # Create one-hot representation, -1 vacancy tokens are left as zero arrays
        for i in range(self.seq_len):
            ind = int(arr[0][i])
            if ind >= 0:
                self.seq[i][ind - 1] = 1
        self.seq = torch.from_numpy(self.seq)

    def getData(self):
        # without the signal
        data = torch.cat((self.seq, self.pka.view(-1, 1), self.molw.view(-1, 1)), dim=1)
        signal = self.signal
        return data, signal


class Chromosome:
    def __init__(self, sequence, chrom_num, TF_num, seq_length=300, sliding_window_step=1):
        fragment_length = 300
        self.seq_length = seq_length
        self.sliding_window_step = sliding_window_step
        self.TF_num = TF_num
        self.chrom_num = chrom_num  # the number of the chromosome [1-16] for indexing purpouses
        self.sequence = sequence  # a dictionary
        self.numfrag = np.size(list(self.sequence.keys()))

        self.chrom_len = fragment_length * (self.numfrag)

    def getSequenceData(self, loc):
        # Gets a sequence of siganl and properties for the given loc and TF
        # Note that the initialization of seq_length and sliding window step 
        # will effect the length and location of the sequence
        # returns: a sequence of length seq_length that starts at the given loc
        #          on the chromosome with signal of the given TF number

        # initializes the proper sub dictionary relevant for the TF
        # TF_inst = self.sequence(self.getFragKey(TF))

        # calculating idices
        start = loc * self.sliding_window_step
        end = start + self.seq_length - 1
        ind_frag_start = start // 300 + 1  # plus 1 because nameing convention starts at 1
        start_loc = start % 300
        ind_frag_end = end // 300 + 1  # plus 1 because nameing convention starts at 1s
        end_loc = end % 300

        # initializing the DNA_Fragments
        frag_start = DNA_fragment(self.sequence[getKey(ind_frag_start)])
        data, signal = frag_start.getData()
        if ind_frag_start != ind_frag_end:
            data = data[start_loc:, :]
            signal = signal[start_loc:]
            for i in range(ind_frag_end - ind_frag_start - 1):
                frag_i = DNA_fragment(self.sequence[getKey(ind_frag_start + i + 1)])
                data_i, signal_i = frag_i.getData()
                data = torch.cat((data, data_i), dim=0)
                signal = torch.cat((signal, signal_i), dim=0)
            frag_end = DNA_fragment(self.sequence[getKey(ind_frag_end)])
            data_end, signal_end = frag_end.getData()
            data_end = data_end[:end_loc + 1, :]
            signal_end = signal_end[:end_loc + 1]
            data = torch.cat((data, data_end), dim=0)
            signal = torch.cat((signal, signal_end), dim=0)
        else:  # the sequence is on the same fragment
            data = data[start_loc:end_loc + 1, :]
            signal = signal[start_loc:end_loc + 1]
        return data, signal

    def getFragKey(self, TF):
        if TF < 10:
            return 'DNA_data_frags_00' + str(TF) + '_chrom' + str(self.chrom_num)
        elif TF < 100:
            return 'DNA_data_frags_0' + str(TF) + '_chrom' + str(self.chrom_num)
        else:
            return 'DNA_data_frags_' + str(TF) + '_chrom' + str(self.chrom_num)


def getKey(i):
    if i < 10:
        return 'fragment_00' + str(i)
    elif i < 100:
        return 'fragment_0' + str(i)
    else:
        return 'fragment_' + str(i)


def forward(inputs, targets):
    loss = -1 * (targets * torch.log(inputs) + (1 - targets) * torch.log(1 - inputs))
    return loss.mean()


class CustomLoss(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.func = f

    def forward(self, inputs, targets):
        return self.f(inputs, targets)

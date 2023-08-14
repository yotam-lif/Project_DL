from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import h5py

AA_nums = 20
DBD_nums = 26
DNA_window = 300


class TF:
    def __init__(self, arr):

        self.props = len(arr)
        self.seq_len = len(arr[0]) - 1

        self.asarray = arr
        self.disorder_score = arr[1]
        self.molw = arr[2]
        self.res_molw = arr[3]
        self.pka = arr[4]
        self.pkb = arr[5]
        self.iso = arr[6]
        self.seq = np.zeros((self.seq_len, AA_nums))
        # self.seq 1st axis is position in IDR, 2nd is one-hot for AA
        # Create one-hot representation, -1 vacancy tokens are left as zero arrays
        for i in range(self.seq_len):
            ind = arr[0][i]
            if ind >= 0:
                self.seq[i][ind] = 1

        self.DBD = np.zeros(DBD_nums)
        dbd_enc = arr[0][-1]
        self.DBD[dbd_enc] = 1


class TFDNA_ds(Dataset):
    def __init__(self, DNA_path, TF_path):
        # initialize TF array
        f_TF = h5py.File(TF_path, 'r')
        dset = f_TF['final_array']
        self.num_TFs = len(dset)
        self.TF_arr = np.array(self.num_TFs)
        for i in range(self.num_TFs):
            self.TF_arr[i] = TF(dset[i])
        # TF array initialized

        # initialize DNA signal per TF
        f_DNA = h5py.File(DNA_path, 'r')
        # ???
        # signal data initialized

    # def __len__(self):
    #     return len(self.label)

    # def __getitem__(self, idx):
    #     g = dgl.graph(([], []), num_nodes=self.n_points[idx])
    #     x_tensor = torch.FloatTensor(self.x[idx])
    #     y_tensor = torch.FloatTensor(self.y[idx])
    #     g.ndata['xy'] = torch.stack((x_tensor, y_tensor), dim=1)
    #     ### Normalize to the range [-1,1] ###
    #     g.ndata['xy'] = (g.ndata['xy'] - 28 / 2) / (28 / 2)
    #
    #     y = self.label[idx]
    #
    #     return g, y

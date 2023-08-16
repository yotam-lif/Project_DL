from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import h5py
from helper_classes import TF as TF
from helper_classes import Chromosome as Chromosome 

AA_nums = 20
DBD_nums = 26
DNA_window = 300
NUC_nums = 4


class TFDNA_ds(Dataset):
    def __init__(self, DNA_path, TF_path,seq_length=300,sliding_window_step=1):
        """
        Initializes the data set.
        #seq_length - the length of DNA sequence we are using as sample for the model
        #sliding_window_step - the step size for the sliding window, if 0 there is no sliding window
        #                       in which case there is no overlap between samples
        """
        
        num_chroms =16
        # initialize TF array
        f_TF = h5py.File(TF_path, 'r')
        dset = f_TF['final_array']
        self.num_TFs = np.shape(dset)[0]
        self.TF_arr = []
        for i in range(self.num_TFs):
            self.TF_arr.append(TF(dset[i]))
        # TF array initialized

        # initialize DNA signal
        self.seq_length = seq_length
        self.sliding_window_step=sliding_window_step
        self.chrom_lengths  =self.get_chrom_lengths(
            num_chroms=num_chroms, file_name = (DNA_path+'/DNA_data_fragments_'))
        self.num_ind = self.num_ind_per_chrom(self.chrom_lengths,self.seq_length,self.sliding_window_step)
        
        self.chromosome_files=[]
        for i in range(num_chroms):
            path = DNA_path+'/DNA_data_fragments_'+str(i+1)+'.h5'
            file = h5py.File(path,'r')
            self.chromosome_files.append(file)
        # signal data initialized

    def __len__(self):
        return int(np.sum(self.num_ind)*self.num_TFs)

    def __getitem__(self, idx):
        TF_num, chrom_num, loc_on_chrom = self.get_TF_chrom_and_loc_from_ind(
            idx,chrom_lengths=self.chrom_lengths,seq_length=self.seq_length,sliding_window_step=self.sliding_window_step)
        chrom_key  = self.getChromKey(TF_num, chrom_num)
        chrom_file = self.chromosome_files[chrom_num-1]
        chrom = Chromosome(
            chrom_file[chrom_key],chrom_num,TF_num,seq_length=self.seq_length,sliding_window_step=self.sliding_window_step)
        data_DNA, signal = chrom.getSequenceData(loc_on_chrom)
        data_TF,DBD = self.TF_arr[TF_num-1].getData()
        data =(data_TF,data_DNA ,DBD)
    
        return data, signal

    def get_chrom_lengths(self,num_chroms =16, file_name = 'signal_159_TFs/DNA_data_fragments_',TF=1):
        #returns an array witht the chromosomes lengths for indexing purposes
        chrom_lenghts  = np.zeros(num_chroms)
        for i in range(num_chroms):
            path = file_name+str(i+1)+'.h5'
            file = h5py.File(path,'r')
            chrom_key  = 'DNA_data_frags_00'+str(TF)+'_chrom'+str(i+1)
            chrom  = Chromosome(file[chrom_key],i+1,1)
            chrom_lenghts[i]=chrom.chrom_len
        return chrom_lenghts

    def get_index_from_TF_and_chromosome(self,ind_on_chrom,TF_num,chrom_num,chrom_lengths,seq_length=300,sliding_window_step=1):
        #ind_on_chrom -the index on the chromosome. Should be in range 1-num_ind_per_chrom
        #TF_num - integer describing the number of transcription factor (should get values in range 1-154)
        #chrom_num - the index of chromosome (should get values in range 1-16)
        #chrom_lengths = an array of the chrom lengths
        #seq_length - the length of DNA sequence we are using as sample for the model
        #sliding_window_step - the step size for the sliding window, if 0 there is no sliding window
        #                       in which case there is no overlap between samples
        #returns - an integer index which is a 1-1 map for this parameters
        if sliding_window_step ==0:
            sliding_window_step=seq_length
        ind_per_chrom =self.num_ind_per_chrom(chrom_lengths,seq_length,sliding_window_step)
        cum_chrom_ind = np.zeros(np.size(chrom_lengths)+1)
        cum_chrom_ind[1:] = np.cumsum(ind_per_chrom)
        DNA_tot_ind_per_TF = np.sum(ind_per_chrom)
        ind = (TF_num-1)*DNA_tot_ind_per_TF+cum_chrom_ind[chrom_num-1]+ind_on_chrom
        return ind


    def num_ind_per_chrom(self,chrom_lengths,seq_length=300,sliding_window_step=1):
        if sliding_window_step==0:
            sliding_window_step = seq_length
        return ((chrom_lengths-seq_length)//sliding_window_step+1)

    def get_TF_chrom_and_loc_from_ind(self,ind,chrom_lengths,seq_length=300,sliding_window_step=1):
        #gets and index and outputs the TF num, chromosome and index on chromosome
        #chrom_lengths = an array of the chrom lengths
        #seq_length - the length of DNA sequence we are using as sample for the model
        #sliding_window_step - the step size for the sliding window, if 0 there is no sliding window
        #                       in which case there is no overlap between samples
        #returns:
        #TF_num - integer describing the number of transcription factor (should get values in range 1-154)
        #chrom_num - the index of chromosome (should get values in range 1-16)
        #loc - the index on the chromosome (has different meaning for differen seq_length 
        #       and sliding window)

        if sliding_window_step ==0:
            sliding_window_step=seq_length
        ind_per_chrom =self.num_ind_per_chrom(chrom_lengths,seq_length,sliding_window_step)
        cum_chrom_ind = np.zeros(np.size(chrom_lengths)+1)
        cum_chrom_ind[1:] = np.cumsum(ind_per_chrom)
        DNA_tot_ind_per_TF = np.sum(ind_per_chrom)
        TF_num = ind//DNA_tot_ind_per_TF +1
        loc_on_DNA = ind%DNA_tot_ind_per_TF
        # print(loc_on_DNA)
        # print(cum_chrom_ind>loc_on_DNA)
        chrom_num = np.argmax((cum_chrom_ind>loc_on_DNA))
        loc_on_chrom = loc_on_DNA-cum_chrom_ind[chrom_num-1]

        return int(TF_num), int(chrom_num), int(loc_on_chrom)
    
    def getChromKey(self,TF,chrom_num):
        if TF<10:
            return 'DNA_data_frags_00'+str(TF)+'_chrom'+str(chrom_num)
        elif TF<100:
            return 'DNA_data_frags_0'+str(TF)+'_chrom'+str(chrom_num)
        else:
            return 'DNA_data_frags_'+str(TF)+'_chrom'+str(chrom_num)

def getKey(i):
    if i<10:
        return 'fragment_00'+str(i)
    elif i<100:
        return 'fragment_0'+str(i)
    else:
        return 'fragment_'+str(i)






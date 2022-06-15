################################################################################################
# Author:        Vincent Lauinger, 
# Affiliation:   Communications Engineering Lab (CEL), Karlsruhe Institute of Technology (KIT)
# Contact:       vincent.lauinger@kit.edu
# Last revision: 15th of June 2022
################################################################################################

import scipy.io as io
from datetime import datetime
import torch
import func_VAENN_MQAM as process

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device == "cuda": torch.cuda.init()
print('Run code on: ', device)             

########################################################################################################
# Define System Variables
mod = '64-QAM' # Modulation Format: {4,16,64}-QAM
sps = 2        #  samples per symbol

net_type_vec = ['Net']  # ['Net_BN'] # topology
channel = 'h1' # 'h2'   # channel model (following Caciularu et al.)

M_vec = [25]            # taps of the estimated channel impulse response    
k1_vec, k2_vec= [25],[3]  # kernel size (of layer1, layer2)

batch_len_vec= [300]      # length of training/updating batch in symbols

lr_optim_vec = [4e-3] #[1e-3, 3e-3, 7e-3] # learning rate
SNR_vec = [24] #np.arange(12,22,2)   # [15,20] # singal to noise ratio

iter = 3                # number of independet simulation runs per parameter setting
N_valid = 15000 #50000  # number of symbols per evaluation step (SER estimation)
train_len = 4000 #1200  # number of total training symbols per epoch
num_epochs = 500 #200   # number of epochs 
epe = 2                 # number of epochs per evaluation

savePATH = ""           # "YOUR/DEFAULT/SAVE/PATH/"


SER = torch.empty(len(SNR_vec), len(k2_vec), len(k1_vec), len(M_vec), len(lr_optim_vec), len(batch_len_vec), iter, num_epochs//epe, dtype=torch.float32, device=device)

for net_type in net_type_vec:       
    n=0
    for batch_len in batch_len_vec:
            l=0
            for lr_optim in lr_optim_vec:
                m=0
                for M in M_vec:
                    k1=0
                    for kernel_1 in k1_vec:
                        k2=0
                        for kernel_2 in k2_vec:
                            s=0
                            for SNR in SNR_vec:
                                for i in range(iter):
                                    SER[s,k2,k1,m,l,n,i,:] =  process.processing(mod, sps, SNR, M, kernel_1, kernel_2, lr_optim, batch_len, N_valid, train_len, num_epochs, epe, channel, net_type)
                                s += 1
                            k2 += 1
                        k1 += 1
                    m += 1
                l += 1
            n += 1


name = f"{savePATH}SERvsSNR_{net_type}_{channel}_{mod}_{sps}_{N_valid}_{epe}_{train_len}_{datetime.today().strftime('%y%m%d%H%M%S')}.mat"
save_dict = {
            'SER': SER.cpu().detach().numpy(),
            'SNR': SNR_vec,
            'k2': k2_vec,
            'k1': k1_vec,
            'M': M_vec,
            'lr': lr_optim_vec,
            'N_train': batch_len_vec
            }
io.savemat(name,{'dict': save_dict})


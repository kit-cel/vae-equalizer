################################################################################################
# Author:        Vincent Lauinger, 
# Affiliation:   Communications Engineering Lab (CEL), Karlsruhe Institute of Technology (KIT)
# Contact:       vincent.lauinger@kit.edu
# Last revision: 15th of June 2022
################################################################################################

import scipy.io as io
from datetime import datetime
import torch
import func_VAELE_MQAM_shaping as process

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device == "cuda": torch.cuda.init()
print('Run code on: ', device)

################################################################################################

mod = '64-QAM'          # Modulation Format: {4,16,64}-QAM
sps = 2                 # samples per symbol

channel = 'h1'  # 'h2'  # channel model (following Caciularu et al.)

M_vec = [25]            # taps of the estimated channel impulse response

N_train_vec = [350]     # length of training/updating batch in symbols

lr_optim_vec = [5e-3]   #[1e-3, 3e-3, 7e-3] # learning rate
SNR_vec = [24]  # np.arange(14,24)  #[15,20] # singal to noise ratio
nu_vec = [0]    # [0] [0.0270955] [0.0872449] [0.1222578] # coefficient determining the shaping and corresponding to an entropy of 6, 5.72, 4.6, 4.125 (for PCS-64-QAM)

iter = 20               # number of independet simulation runs per parameter setting
N_valid = 15000 # 50000 # number of symbols per evaluation step (SER estimation)
train_len = 1200 # 1200 # number of total training symbols per epoch
num_epochs = 500 # 200  # number of epochs
epe = 2                 # number of epochs per evaluation

savePATH = ""  # "YOUR/DEFAULT/SAVE/PATH/"

SER = torch.empty(len(SNR_vec), 1, 1, len(M_vec), len(lr_optim_vec), len(N_train_vec), iter, num_epochs//epe, dtype=torch.float32, device=device)

n = 0
for N_train in N_train_vec:
        l = 0
        for lr_optim in lr_optim_vec:
            m = 0
            for M in M_vec:
                s = 0
                for SNR in SNR_vec:
                    nn = 0
                    for nu in nu_vec:
                        for i in range(iter):
                            SER[s, 0, 0, m, l, n, i, :] = process.processing(mod, sps, SNR, nu, M, lr_optim, N_train, N_valid, train_len, num_epochs, epe, channel)
                            nn += 1
                    s += 1
                m += 1
            l += 1
        n += 1

name = f"{savePATH}SERvsSNR_VAELE_shaping_{nu}_{channel}_{mod}_{sps}_{N_valid}_{epe}_{train_len}_{datetime.today().strftime('%y%m%d%H%M%S')}.mat"
save_dict = {
            'SER': SER.cpu().detach().numpy(),
            'SNR': SNR_vec,
            'M': M_vec,
            'lr': lr_optim_vec,
            'N_train': N_train_vec,
            'nu': nu_vec
            }
io.savemat(name,{'dict': save_dict})


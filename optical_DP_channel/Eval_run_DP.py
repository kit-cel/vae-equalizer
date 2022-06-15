################################################################################################
# Author:        Vincent Lauinger, 
# Affiliation:   Communications Engineering Lab (CEL), Karlsruhe Institute of Technology (KIT)
# Contact:       vincent.lauinger@kit.edu
# Last revision: 15th of June 2022
################################################################################################

import numpy as np
import scipy.io as io
from datetime import datetime
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device == "cuda": torch.cuda.init()
print('Run code on: ', device)             


mod = '64-QAM' # modulation format:  {4,16,64}-QAM
sps = 2        # oversampling factor in samples per symbol

loss_type = 'VAE' # 'VAE' 'VAEflex' 'CMA' 'CMAbatch' 'CMAflex'  
channel = 'h0' # corresponds to optical channel with PMD and ISI caused by CD 

nu_vec = [0] # [0] [0.0270955] [0.0872449] [0.1222578] # coefficient determining the shaping and corresponding to an entropy of 6, 5.72, 4.6, 4.125 (for PCS-64-QAM)

symb_rate_vec = [90e9] #[40e9,64e9,80e9,90e9,100e9] # Symbol rate in Baud

tau_pmd =  0.1e-12 * np.sqrt(1000)   # pol. mode dispersion coeff. --- tau_pmd = D_pmd*sqrt(L) in ps/sqrt(km) * sqrt(km)   
tau_cd = -26e-24 * 1  # chromatic dispersion coeff. --- tau_cd = beta_2 * L_residual in ps**2/km * km - -26 ps**2/km <=> 20ps/nm/km @ 1550nm 
phiIQ = np.array([0.0314, 0.0314], dtype=np.complex64) # static IQ-shift in rad
theta_vec = [np.pi/10]  # HV shift in rad
theta_diff_vec = [0.06*np.pi] #[0, 0.03*np.pi, 0.06*np.pi, 0.09*np.pi, 0.12*np.pi] # diff. HV shift (per frame)

SNR_vec = [23] #np.arange(20,30,2)   # [15,20] Signal to noise ratio in dB

M_vec = [25] # filter length (number of taps)

batch_len_vec= [100] #[10, 20]      # length of training/updating batch in symbols
flex_step_vec = [10] # step length of the flex-schemes in symbols (only relevant for flex-schemes)

lr_optim_vec = [2.5e-3, 2e-3, 3e-3] # learning rate


iter = 5    # number of independet simulation runs per parameter setting
N_lrhalf = 170 #20   # number of frames until learning rate is halved # to disable lr scheduler, set N_lrhalf >= num_epochs
num_frames = 170    # maximum number of frames 
N_frame_max = 10000 # maximum number of symbols per frame

savePATH = "" # "YOUR/DEFAULT/SAVE/PATH/"


SER = torch.empty(4,len(SNR_vec), len(symb_rate_vec), len(nu_vec), len(theta_diff_vec), len(M_vec), len(lr_optim_vec), len(batch_len_vec), len(flex_step_vec), len(theta_vec), iter, num_frames, dtype=torch.float32, device=device)
Var_est = torch.empty(2,len(SNR_vec), len(symb_rate_vec), len(nu_vec), len(theta_diff_vec), len(M_vec), len(lr_optim_vec), len(batch_len_vec), len(flex_step_vec), len(theta_vec), iter, num_frames, dtype=torch.float32, device=device)
var_real = torch.empty(2,len(SNR_vec), len(symb_rate_vec), len(nu_vec), len(theta_diff_vec), len(M_vec), len(lr_optim_vec), len(batch_len_vec), len(flex_step_vec), len(theta_vec), iter,1, dtype=torch.float32, device=device)

if loss_type == 'VAE':
    import func_VAELE_DP_MQAM_shaping as process
elif loss_type == 'CMA':
    import func_CMA_DP_MQAM_shaping as process
elif loss_type == 'VAEflex':
    import func_VAEflex_DP_MQAM_shaping as process
elif loss_type == 'CMAflex':
    import func_CMAflex_DP_MQAM_shaping as process
elif loss_type == 'CMAbatch':
    import func_CMAbatch_DP_MQAM_shaping as process

n = 0
for nu in nu_vec:       
    nt=0
    for batch_len in batch_len_vec:
            l=0
            for lr_optim in lr_optim_vec:
                m=0
                for M in M_vec:
                    t1=0
                    for theta_diff in theta_diff_vec:
                        sr=0
                        for symb_rate in symb_rate_vec:
                            ss=0
                            for flex_step in flex_step_vec:
                                v = 0
                                for theta in theta_vec:
                                    s = 0
                                    for SNR in SNR_vec:
                                        for i in range(iter):
                                            SER[:,s,sr,n,t1,m,l,nt,ss,v,i,:], Var_est[:,s,sr,n,t1,m,l,nt,ss,v,i,:], var_real[:,s,sr,n,t1,m,l,nt,ss,v,i,0] =  process.processing(mod, sps, SNR, nu, M, theta_diff,theta,lr_optim, batch_len, N_frame_max, num_frames, flex_step, channel, symb_rate,tau_cd,tau_pmd,phiIQ, N_lrhalf)
                                        s += 1
                                    v += 1
                                ss += 1
                            sr += 1
                        t1 += 1
                    m += 1
                l += 1
            nt += 1
    n += 1



name = f"{savePATH}SERvsSNR_{loss_type}_DP_{mod}_N_lrhalf_{N_lrhalf}_N_train_{N_frame_max}_{datetime.today().strftime('%y%m%d%H%M%S')}.mat"
save_dict = {
            'SER': SER.cpu().detach().numpy(),
            'Var_est': Var_est.cpu().detach().numpy(),
            'var_real': var_real.cpu().detach().numpy(),
            'SNR': SNR_vec,
            'nu': nu_vec,
            'theta_diff': theta_diff_vec,
            'theta': theta_vec,
            'M': M_vec,
            'lr': lr_optim_vec,
            'batch_len': batch_len_vec,
            'symb_rate': symb_rate_vec,
            'symb_step': flex_step_vec
            }
io.savemat(name,{'dict': save_dict})


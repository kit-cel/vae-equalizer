################################################################################################
# Author:        Vincent Lauinger, (based on code of Moritz Luca Schmid, CEL, KIT)
# Affiliation:   Communications Engineering Lab (CEL), Karlsruhe Institute of Technology (KIT)
# Contact:       vincent.lauinger@kit.edu
# Last revision: 15th of June 2022
################################################################################################

import numpy as np
import torch
import matplotlib.pyplot as plt

torch.set_grad_enabled(False)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("We are using the following device for learning:",device)

################################## CHOOSE CHANNEL MODEL AND MOD. FORMAT ##################################

#h_channel_orig= np.array([0.04, -0.05, 0.07, -0.21, -0.5, 0.72, 0.36, 0, 0.21, 0.03, 0.07])  # Proakis Channel A, Fig. 9.4-5a -- good-quality telephone channel
#h_channel_orig= np.array([0.407, 0.815, 0.407])  # channel impulse response (Proakis Channel B) Fig. 9.4-5b
#h_channel_orig= np.array([0.227, 0.460, 0.688, 0.460, 0.227])  # channel impulse response (Proakis Channel C), Fig. 9.4-5c

h_channel_orig= np.array([0.0545+1j*0.05, 0.2823-1j*0.11971, -0.7676+1j*0.2788, -0.0641-1j*0.0576, 0.0466-1j*0.02275]).astype(np.complex64) # Caciularu: channel h_1
#h_channel_orig= np.array([0.0545+1j*0.0165, -1.3449-1j*0.4523, 1.0067+1j*1.1524, 0.3476+1j*0.3153]).astype(np.complex64) # Caciularu: channel h_2
#h_channel_orig= np.array([0,0+1j*0, 0+1j*0, 1+1j*0, 0+1j*0, 0+1j*0,0]).astype(np.complex64) # test channel

mod = '64-QAM'

####################################################################

M = len(h_channel_orig)
sps = 1     # only implemented for 1 sps


h_channel = np.zeros((sps*(h_channel_orig.shape[-1]-1)+1), dtype=np.complex64)
h_channel[0::sps] = h_channel_orig
h_channel /= np.linalg.norm(h_channel)    # Normalisation of the channel
h_tensor = torch.tensor(h_channel, dtype=torch.cfloat, device=device, requires_grad=False)

constellations = {'4-QAM': np.array([-1,-1,1,1]) + 1j*np.array([-1,1,-1,1]), 
                '16-QAM': np.array([-3,-3,-3,-3,-1,-1,-1,-1,1,1,1,1,3,3,3,3]) + 1j*np.array([-3,-1,1,3,-3,-1,1,3,-3,-1,1,3,-3,-1,1,3]),
                '64-QAM': np.array([-7,-7,-7,-7,-7,-7,-7,-7,-5,-5,-5,-5,-5,-5,-5,-5,-3,-3,-3,-3,-3,-3,-3,-3,-1,-1,-1,-1,-1,-1,-1,-1,1,1,1,1,1,1,1,1,3,3,3,3,3,3,3,3,5,5,5,5,5,5,5,5,7,7,7,7,7,7,7,7]) \
                + 1j*np.array([-7,-5,-3,-1,1,3,5,7,-7,-5,-3,-1,1,3,5,7,-7,-5,-3,-1,1,3,5,7,-7,-5,-3,-1,1,3,5,7,-7,-5,-3,-1,1,3,5,7,-7,-5,-3,-1,1,3,5,7,-7,-5,-3,-1,1,3,5,7,-7,-5,-3,-1,1,3,5,7])}

constellation = constellations[mod] / np.sqrt(np.mean(np.abs(constellations[mod])**2))
const_torch = torch.tensor(constellation, dtype=torch.cfloat, device=device, requires_grad=False)

amp_levels = constellation.real
num_lev = int(np.sqrt(len(amp_levels)))
amp_levels = torch.tensor(amp_levels[::num_lev], device=device, dtype=torch.float32)
const_mean = torch.tensor(np.mean(np.abs(constellation)), device=device, dtype=torch.float32)

################################## HELPER FUNCTIONS ##################################

def rcfir(T,sps, beta):
    #T = 6 # pulse duration in symbols
    #sps = 2 # oversampling factor in samples per symbol
    #beta = 0.1 # roll-off factor
    t = np.arange(-T*sps/2,T*sps/2, 1/sps, dtype=np.float32)
    h = np.sinc(t)*np.cos(np.pi*beta*t)/(1-(2*beta*t)**2)
    h[np.abs(t)==1/2/beta] = np.pi/4*np.sinc(1/(2*beta))
    h = h/np.linalg.norm(h)    # Normalisation of the pulseforming filter
    return h

def rrcfir(T,sps, beta):
    #T = 6 # pulse duration in symbols
    #sps = 2 # oversampling factor in samples per symbol
    #beta = 0.1 # roll-off factor
    t = np.arange(-T*sps/2,T*sps/2, 1/sps, dtype=np.float32)
    h = (np.sin(np.pi*t*(1-beta)) + 4*beta*t*np.cos(np.pi*t*(1+beta)) ) / (np.pi*t*(1-(4*beta*t)**2))
    h[np.abs(t)==1/4/beta] = beta/np.sqrt(2) * ( (1+2/np.pi) * np.sin(np.pi/4/beta) + (1-2/np.pi) * np.cos(np.pi/4/beta) )
    h[t==0] = (1 + beta*(4/np.pi - 1))
    h = h/np.linalg.norm(h)    # Normalisation of the pulseforming filter
    return h


def generate_data_shaping(N, amp_levels, SNR, h_channel, nu):
    T = 8 # length of pulse-shaping filter in symbols
    beta = 0.1 # foll-off factor

    amps = amp_levels.detach().cpu().numpy()
    sc = np.min(np.abs(amps))
    amps_scale = amps/sc

    P = np.exp(-nu*np.abs(amps_scale)**2); P = P / np.sum(P)

    N_conv = N+len(h_channel)+4*T
    tx_up = np.zeros(sps*(N_conv-1)+1, dtype=np.complex64)

    rng = np.random.default_rng()
    data = rng.choice(amps, (2,N_conv), p=P)   
    tx_sig = data[0,:] + 1j*data[1,:] 
    tx_up[::sps] = tx_sig       # sps-upsampled signal by zero-insertion

    h_pulse = rcfir(T,sps,beta)     # rcfir @ 1sps --> Dirac
    rx_sig = np.convolve(tx_up, h_pulse, mode='valid') 
    rx_sig = np.convolve(rx_sig, h_channel, mode='valid') 

    sigma_n = np.sqrt( sps*np.mean(np.abs(rx_sig)**2)/2  /10**(SNR/10) )
    rx_sig += sigma_n* (np.random.randn(*rx_sig.shape) +1j*np.random.randn(*rx_sig.shape))
    
    rx_tensor = torch.from_numpy(np.asarray([np.real(rx_sig[:sps*N]),np.imag(rx_sig[:sps*N])])).to(device,torch.float32)     
    data_tensor = torch.from_numpy(np.asarray([data[0,(T+M-1):(N+T+M-1)],data[1,(T+M-1):(N+T+M-1)]])).to(device,torch.float16)

    return rx_tensor, data_tensor, torch.tensor(P,device=device, dtype=torch.float32)

def SER_func(rx, tx): 
    N = tx.shape[1]
    amp_lev_mat = amp_levels.repeat(N,1).transpose(0,1)

    dec = torch.empty_like(tx, device=device, dtype=torch.int16)
    data = torch.empty_like(tx, device=device, dtype=torch.int16)
    SER = torch.ones(4, device=device, dtype=torch.float32)
    
    scale = (num_lev-1)/2
    data[:,:] = torch.round(scale*tx.float()+scale)
    rx *= torch.mean(torch.sqrt(tx[0,:].float()**2 + tx[1,:].float()**2)) /torch.mean(torch.sqrt(rx[0,:]**2 + rx[1,:]**2))
    sig_I, sig_Q = rx[0,:N], rx[1,:N] # / torch.sqrt(2*torch.mean(rx[0,:N]**2)), rx[1,:N] / torch.sqrt(2*torch.mean(rx[1,:N]**2))
    ### zero phase-shift
    dec[0,:], dec[1,:] = torch.argmin(torch.abs(sig_I-amp_lev_mat), dim=0), torch.argmin(torch.abs(sig_Q-amp_lev_mat), dim=0)
    SER[0] = torch.mean( (data - dec).bool().any(dim=0) , dtype=torch.float32)
    
    ### pi phase-shift
    dec_pi = -(dec-scale*2)
    SER[1] = torch.mean( (data - dec_pi).bool().any(dim=0) , dtype=torch.float32)

    ### pi/4 phase-shift
    dec_pi4 = torch.empty_like(dec)
    dec_pi4[0,:], dec_pi4[1,:] = -(dec[1,:]-scale*2), dec[0,:]
    SER[2] = torch.mean( (data - dec_pi4).bool().any(dim=0) , dtype=torch.float32)

    ### 3pi/4 phase-shift
    dec_3pi4 = torch.empty_like(dec)
    dec_3pi4[0,:], dec_3pi4[1,:] = -(dec_pi4-scale*2)
    SER[3] = torch.mean( (data - dec_3pi4).bool().any(dim=0) , dtype=torch.float32)

    return torch.min(SER)   

def find_shift_symb(rx, tx, N_shift): 
    mat = torch.empty(1000-N_shift//2,N_shift, device=rx.device, dtype=torch.float32)

    for i in range(N_shift):
        mat[:,i] = rx[0,i:1000-N_shift//2+i:]
    corr = torch.matmul(tx[0,N_shift//2:1000].to(torch.float32),mat)
    if torch.max(torch.abs(corr)) >= 0.02*rx.shape[-1]:
        return torch.argmax(torch.abs(corr))-N_shift//2
    else:
        corr_IQ = torch.matmul(tx[1,N_shift//2:1000].to(torch.float32),mat)
        if torch.max(torch.abs(corr_IQ)) >= torch.max(torch.abs(corr)):
            return torch.argmax(torch.abs(corr_IQ))-N_shift//2
        else:
            return torch.argmax(torch.abs(corr))-N_shift//2
            
def compute_lmmse(channel, SNR, order, n1):
    """
    Compute MMSE filter, given the channel taps, the SNR and the filter order.
    :param channel: 1D tensor with channel taps
    :param SNR: SNR in dB
    :param order: Number of taps, the MMSE filter is supposed to have
    :param n1: Number of acausal channel taps.
    :returns: 1D tensor with channel taps of computed MMSE filter of length 'order'.
    """
    sigma_w = 1/2  /10**(SNR/10) #1/EsN0_lin
    L = len(channel)-1
    H = torch.zeros((order, order+L),dtype=torch.cfloat, device=device)
    for i, h in enumerate(H):
        h[i:i+L+1] = torch.flip(channel, dims=[0])
    return torch.flip( torch.matmul( torch.linalg.inv(sigma_w * torch.eye(order) + torch.matmul(H, H.T)), H[:,-(n1+1)]) ,dims=[0]).to(device)

def compute_feedforward(channel, SNR, order):
    """
    Compute feedforward filter of DFE (essentially a causal MMSE filter)
    :param channel: 1D tensor with channel taps
    :param  SNR: SNR in dB
    :param order: Number of taps, the MMSE filter is supposed to have
    """
    sigma_w = 1/2  /10**(SNR/10) #1/EsN0_lin
    L = len(channel)-1
    H = torch.zeros([order, order],dtype=torch.cfloat, device=device)
    for i in range(order-L):
        H[i, i:i+L+1] = channel
    for i in range(L):
        H[order-L+i, order-L+i:] = channel[:L-i]
    return torch.matmul( torch.linalg.inv(sigma_w * torch.eye(order,dtype=torch.cfloat, device=device) + torch.matmul(H, H.T)), torch.cat((torch.zeros(order-L-1,dtype=torch.cfloat, device=device), torch.flip(channel, dims=[0])))) 

def compute_feedback_filter(channel, feedforward):
    """
    Compute filter taps of feedback filter, based on feedforward filter taps.
    We set the filter order of the feedback section equal to the feedforward section.
    :param channel: 1D tensor with channel taps
    :param feedforward: Filter taps of feedforward section.
    """
    K1 = len(feedforward)-1 
    L = len(channel)-1
    feedback_taps = torch.zeros(L,dtype=torch.cfloat, device=device)
    for k in range(L):
        feedback_taps[k] = - torch.dot(feedforward[-(L-k):], torch.flip(channel[k+1:], dims=[0]))
    return feedback_taps

def dfe(feedforward_output, feedforward_filter, feedback_filter, init_decisions_idxs):
    """
    Apply DFE. For the first symbols, where no feedback exists, we apply an acausal MMSE filter for better performance.
    """
    K1 = len(feedforward_filter) -1 # feedforward
    K2 = len(feedback_filter) # feedback

    #assert len(init_decisions_idxs.shape) == 2
    batch_size = init_decisions_idxs.shape[0]

    state = torch.zeros(feedforward_output.shape, dtype=torch.cfloat, device=device)
    state_idxs = torch.zeros(feedforward_output.shape)
    # decide for the first K2 symbols, to init the feedback filter
    state_idxs[:K2] = init_decisions_idxs[:K2]
    state[:K2] = const_torch[(init_decisions_idxs[:K2]).long()]
    for k in range(feedforward_output.shape[0]-K2):
        vk = feedforward_output[K2+k]
        correction = torch.dot(feedback_filter, torch.flip(state[k:k+K2],dims=[-1]))
        Ik = vk + correction.flatten()
        Ik_hat_idx = nearest_neighbor(Ik)# hard decision on idx
        state_idxs[k+K2] = Ik_hat_idx
        state[k+K2] = const_torch[Ik_hat_idx.long()]
    return state_idxs

def nearest_neighbor(rx_syms):
    """
        Accepts a sequence of (possibly equalized) complex symbols.
        Each sample is hard decided to the constellation symbol, which is nearest (Euclidean distance).
        The output are the idxs of the constellation symbols.
    """
    const_mat = const_torch.repeat(rx_syms.shape[-1],1).transpose(0,1)
    # Compute distances to all possible symbols.
    distance = torch.abs(const_mat - rx_syms[..., None].squeeze())
    hard_dec_idx = torch.argmin(distance, dim=0)
    return hard_dec_idx

def compl_conv(rx,h):
    K = h.shape[-1]
    rx_conv = rx.view(1,1,rx.shape[-1])
    h_conv = torch.flip(h, dims=[-1]).view(1,1,K)
    out = torch.nn.functional.conv1d(torch.real(rx_conv), torch.real(h_conv), padding=K//2) - torch.nn.functional.conv1d(torch.imag(rx_conv), torch.imag(h_conv), padding=K//2) + 1j * (torch.nn.functional.conv1d(torch.imag(rx_conv), torch.real(h_conv), padding=K//2) + torch.nn.functional.conv1d(torch.real(rx_conv), torch.imag(h_conv), padding=K//2) )
    return out.squeeze()

################################## MAIN PART ##################################

# channel parameter
SNR_vec = np.arange(15,23,1) # #np.array([9,10,18])

nu = 0.0270955 #distr-parameter for R_c=0.8, taget_SE=2.8 ##0 # [0.1222578] #[0.0872449] # [0.0270955]

N_valid = 128000
N_cut = 20

lmmse_filter_order = 20
M_dfe = 11

num_epochs = 5

n1 = (lmmse_filter_order-1) // 2 + 1  # number of acausal taps
num_snr = SNR_vec.shape[-1]
SER_mmse = torch.zeros(num_snr,num_epochs)
SER_dfe = torch.zeros(num_snr,num_epochs)

for snr_ind in range(num_snr):
    SNR = SNR_vec[snr_ind]
    lmmse_taps = compute_lmmse(h_tensor, SNR, lmmse_filter_order, lmmse_filter_order//2+1)

    # DFE (feedforward and feedback filter)
    ff = compute_feedforward(h_tensor, SNR, M_dfe)
    feedback_filter = compute_feedback_filter(h_tensor, ff)
    
    for epoch in range(num_epochs):
        rx_tensor_valid, data_tensor_valid, P = generate_data_shaping(
            N_valid, amp_levels, SNR, h_channel, nu)
        rx = torch.complex(rx_tensor_valid[0,:], rx_tensor_valid[1,:])

        mmse_soft_decision = compl_conv(rx,lmmse_taps)
        # Quantize to hard decisions.
        symbol_ests_idxs = nearest_neighbor(mmse_soft_decision[1::sps])
        mmse_hard_decision = torch.view_as_real(const_torch[symbol_ests_idxs.long()]).T

        shift = find_shift_symb( torch.view_as_real(mmse_soft_decision).T,data_tensor_valid, 21)
        SER_mmse[snr_ind,epoch] = SER_func(torch.view_as_real(mmse_soft_decision).T[:,N_cut+11+shift:-11-N_cut], data_tensor_valid[:,N_cut+11:-11-shift-N_cut]) 

        # Apply DFE.
        # Apply feedforward filter.
        ff_result = compl_conv(rx,ff)
        # Quantize to hard decisions.
        # DFE
        dfe_idxs = dfe(ff_result, ff, feedback_filter, symbol_ests_idxs)
        dfe_hard_decisions = torch.view_as_real(const_torch[dfe_idxs.long()]).T

        shift = find_shift_symb(dfe_hard_decisions,data_tensor_valid, 24)
        SER_dfe[snr_ind,epoch] = SER_func(dfe_hard_decisions[:,N_cut+11+shift:-11-N_cut], data_tensor_valid[:,N_cut+11:-11-shift-N_cut]) 

        print(epoch, SNR, '\t\t\t\t\t\t\tSER_mmse = ',  SER_mmse[snr_ind,epoch].item(), '\t\t\t SER_dfe = ',  SER_dfe[snr_ind,epoch].item()   )   #, '\n h = ', h_est.tolist())



### plotting constellation for debugging ###

def plot_constellation(E):
    fontSize = 35
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(1, 1, 1)
    if np.iscomplexobj(E) == True:
        if E.shape[0] == 2:
            ax.scatter(E[0].real, E[0].imag, s = 1, c='red', alpha=0.5, label="X")
            ax.scatter(E[1].real, E[1].imag, s = 1, c='blue', alpha=0.5, label="Y")
        else:
            ax.scatter(E.real, E.imag, s = 1, c='red', alpha=0.5, label="X")
    else:
        if E.shape[1] == 2:
            if E.shape[0] == 2:
                ax.scatter(E[0,0,:], E[0,1,:], s = 3, c='red', alpha=0.8, label="X")
                ax.scatter(E[1,0,:], E[1,1,:], s = 3, c='blue', alpha=0.8, label="Y")
            else:
                ax.scatter(E[0,0,:], E[0,1,:], s = 3, c='red', alpha=0.8, label="X")
        else:
            ax.scatter(E[0,:], E[1,:], s = 3, c='red', alpha=0.8, label="X")

    plt.legend(loc='best', fontsize=fontSize)
    plt.xlabel('In-Phase', fontsize=fontSize)
    plt.ylabel('Quadrature', fontsize=fontSize)
    plt.xticks(fontsize=fontSize)
    plt.yticks(fontsize=fontSize)
    plt.grid(True)
    plt.show()

plot_constellation(mmse_soft_decision)

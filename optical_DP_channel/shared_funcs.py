################################################################################################
# Author:        Vincent Lauinger, 
# Affiliation:   Communications Engineering Lab (CEL), Karlsruhe Institute of Technology (KIT)
# Contact:       vincent.lauinger@kit.edu
# Last revision: 15th of June 2022
################################################################################################

import numpy as np
from numpy.core.numeric import Inf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#import helper_funcs_luca as hefu

def rcfir(T,sps, beta):     # raised-cosine filter
    #T = 6 # pulse duration in symbols
    #sps = 2 # oversampling factor in samples per symbol
    #beta = 0.1 # roll-off factor
    t = np.arange(-T*sps/2,T*sps/2, 1/sps, dtype=np.float32)
    h = np.sinc(t)*np.cos(np.pi*beta*t)/(1-(2*beta*t)**2)
    h[np.abs(t)==1/2/beta] = np.pi/4*np.sinc(1/(2*beta))
    h = h/np.linalg.norm(h)    # Normalisation of the pulseforming filter
    return h

def rrcfir(T,sps, beta):    # root-raised-cosine filter
    #T = 6 # pulse duration in symbols
    #sps = 2 # oversampling factor in samples per symbol
    #beta = 0.1 # roll-off factor
    t = np.arange(-T*sps/2,T*sps/2, 1/sps, dtype=np.float32)
    h = (np.sin(np.pi*t*(1-beta)) + 4*beta*t*np.cos(np.pi*t*(1+beta)) ) / (np.pi*t*(1-(4*beta*t)**2))
    h[np.abs(t)==1/4/beta] = beta/np.sqrt(2) * ( (1+2/np.pi) * np.sin(np.pi/4/beta) + (1-2/np.pi) * np.cos(np.pi/4/beta) )
    h[t==0] = (1 + beta*(4/np.pi - 1))
    h = h/np.linalg.norm(h)    # Normalisation of the pulseforming filter
    return h

def simulate_dispersion(rx,symb_rate,sps,tau_cd,tau_pmd,phiIQ,theta):
    # simulate residual CD, PMD, pol. rot and IQ-shift in f-domain
    rx_fft = np.fft.fft(rx, axis=1)
    freq = np.fft.fftfreq(rx.shape[1], 1/symb_rate/sps)
    exp_cd, exp_pmd = np.exp(1j*2*(np.pi*freq)**2*tau_cd), np.exp(1j*np.pi*tau_pmd*freq)
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)     # 
    exp_phiIQ = np.exp(-1j*phiIQ)
    
    # simulate pol. ritation and PMD with rotationary matrix  
    R = np.asarray([[cos_theta*exp_phiIQ[0], sin_theta*exp_phiIQ[0]], [-sin_theta*exp_phiIQ[1], cos_theta*exp_phiIQ[1]]])
    R_T = np.asarray([[cos_theta*exp_phiIQ[0], -sin_theta*exp_phiIQ[0]], [sin_theta*exp_phiIQ[1], cos_theta*exp_phiIQ[1]]])
    Diag_pmd = np.asarray([[exp_pmd, 0], [0, 1/exp_pmd]])
    H = R_T @ Diag_pmd @ R
    
    RX_fft = np.zeros((2,rx.shape[1]), dtype = np.complex128)
    RX_fft[0,:], RX_fft[1,:] = (H[0,0]*rx_fft[0,:] + H[0,1]*rx_fft[1,:])*exp_cd, (H[1,0]*rx_fft[0,:] + H[1,1]*rx_fft[1,:])*exp_cd 
    return np.complex64(np.fft.ifft(RX_fft, axis=1))    # return signal in t-domain

def simulate_channel(tx_up, h_pulse, h_channel):
    pol = tx_up.shape[0]
    rx_sig = np.zeros((pol,tx_up.shape[1]-h_pulse.shape[0]-h_channel.shape[0]+2), dtype=np.complex64)

    for i in range(tx_up.shape[0]): # num. of pol.
        temp = np.convolve(tx_up[i,:], h_pulse, mode='valid')   # convolve with pulse shaping
        rx_sig[i,:] = np.convolve(temp, h_channel, mode='valid')    # convolve with (additional) channel IR
    return rx_sig 

def generate_data_shaping(N, amps, SNR, h_channel, P, pol, symb_rate, sps, tau_cd, tau_pmd, phiIQ, theta, device):
    T = 8 # length of pulse-shaping filter in symbols
    beta = 0.1 # roll-off factor

    M = len(h_channel)  # number of channel taps

    N_conv = N+len(h_channel)+4*T
    tx_up = np.zeros((pol,sps*(N_conv-1)+1), dtype=np.complex64)
    rx_sig = np.zeros((pol,sps*N), dtype=np.complex64)

    rng = np.random.default_rng()
    data = rng.choice(amps, (pol*2,N_conv), p=P)    # draw random amplitude level from corresponding pmf P
    tx_up[:,::sps] = data[0::pol,:] + 1j*data[1::pol,:]       # sps-upsampled signal by zero-insertion

    h_pulse = rrcfir(T,sps,beta)
    temp = simulate_channel(tx_up, h_pulse, h_channel)
    temp = simulate_dispersion(temp,symb_rate,sps,tau_cd,tau_pmd,phiIQ,theta)

    sigma_n = np.sqrt( np.mean(np.abs(temp)**2)*sps/2  /10**(SNR/10) ) #var/2 due to I/Q, *sps due to oversampling with zeros
    temp += sigma_n* (np.random.randn(*temp.shape) +1j*np.random.randn(*temp.shape)) # Standard-normal distribution with exp(1/2*x**2)

    rx_sig = temp 
       
    rx_tensor = torch.from_numpy(np.asarray([np.real(rx_sig[:,:sps*N]),np.imag(rx_sig[:,:sps*N])])).permute(1,0,2).to(device,torch.float32)      
    data_tensor = torch.from_numpy(np.asarray([data[0::pol,(T+M-1):(N+T+M-1)],data[1::pol,(T+M-1):(N+T+M-1)]])).permute(1,0,2).to(device,torch.float16)
    return rx_tensor, data_tensor, sigma_n

def loss_function_shaping(q, rx, h_est, amp_levels, P):
    device = q.device
    pol = q.shape[0]
    N = rx.shape[-1]
    sps = N//q.shape[-1]
    mh = h_est.shape[3]//2
    Mh = 2*mh
    num_lev = amp_levels.shape[0]

    Eq = torch.zeros(pol,2,N, device=device, dtype=torch.float32)
    Var = torch.zeros(pol,2,N, device=device, dtype=torch.float32)

    h = h_est

    # compute expecation (with respect to q) of x and x**2
    amp_lev_mat = amp_levels.repeat(pol,q.shape[2],2).transpose(1,2)
    temp_lev = amp_lev_mat * q
    temp_pow = (amp_lev_mat**2) * q
 
    Eq[:,0,::sps], Eq[:,1,::sps] = torch.sum(temp_lev[:,:num_lev,:], dim=1), torch.sum(temp_lev[:,num_lev:,:], dim=1)
    Var[:,0,::sps], Var[:,1,::sps] = torch.sum(temp_pow[:,:num_lev,:], dim=1), torch.sum(temp_pow[:,num_lev:,:], dim=1)
    Var -= Eq**2

    h_absq = torch.sum(h**2, dim=2)

    D_real = torch.zeros(2,N-Mh, device=device, dtype=torch.float32)  
    D_imag = torch.zeros(2,N-Mh, device=device, dtype=torch.float32)
    E = torch.zeros(2, device=device, dtype=torch.float32)    
    idx = np.arange(Mh,N)
    nm = idx.shape[0]
    
    for j in range(Mh+1): # h[chi,nu,c,k]
        D_real += h[:,0,0:1,j].expand(-1,nm) * Eq[0,0:1,idx-j].expand(pol,-1) - h[:,0,1:2,j].expand(-1,nm) * Eq[0,1:2,idx-j].expand(pol,-1) \
            + h[:,1,0:1,j].expand(-1,nm) * Eq[1,0:1,idx-j].expand(pol,-1) - h[:,1,1:2,j].expand(-1,nm) * Eq[1,1:2,idx-j].expand(pol,-1)
        D_imag += h[:,0,1:2,j].expand(-1,nm) * Eq[0,0:1,idx-j].expand(pol,-1) + h[:,0,0:1,j].expand(-1,nm) * Eq[0,1:2,idx-j].expand(pol,-1) \
            + h[:,1,1:2,j].expand(-1,nm) * Eq[1,0:1,idx-j].expand(pol,-1) + h[:,1,0:1,j].expand(-1,nm) * Eq[1,1:2,idx-j].expand(pol,-1)
        Var_sum = torch.sum(Var[:,:,idx-j], dim=(1,2))
        E += h_absq[:,0,j] * Var_sum[0] + h_absq[:,1,j] * Var_sum[1]
    
    P_mat = P.repeat(q.shape[-1]-Mh,2).transpose(0,1)   # P(x)
    data_entropy = torch.sum( -q[0,:,mh:-mh]*torch.log(q[0,:,mh:-mh]/P_mat+ 1e-12) - q[1,:,mh:-mh]*torch.log(q[1,:,mh:-mh]/P_mat+ 1e-12) )
    C = torch.sum(rx[:,:,mh:-mh]**2, dim=(1,2))
    C += -2*torch.sum( rx[:,0,mh:-mh]*D_real + rx[:,1,mh:-mh]*D_imag, dim=1) + torch.sum( D_real**2 + D_imag**2, dim=1) + E
    #print((N-M)*torch.log(C),data_entropy)
    retval = torch.sum((N-Mh)*torch.log(C)) - data_entropy
    return retval, (C/(N-Mh)).detach()  # second value is the estimated variance


def CPE(y):
    # carrier phase estimation based on Viterbi-Viterbi algorithm
    pi = torch.full([1],3.141592653589793, device=y.device, dtype=torch.float32)
    pi2, pi4 = pi/2, pi/4
    M_ma = 501     # length of moving average filter 
    y_corr = torch.zeros_like(y)
    y_pow4 = torch.zeros_like(y)
    ax, bx, ay, by = y[0,0,:], y[0,1,:], y[1,0,:], y[1,1,:]
    
    # taking the signal to the 4th power to cancel out modulation
    # # (a+jb)^4 = a^4 - 6a^2b^2 + b^4 +j(4a^3b - 4ab^3)
    ax2, bx2, ay2, by2 = ax**2, bx**2, ay**2, by**2
    y_pow4[0,0,:] = ax2*ax2 - torch.full_like(ax,6)*ax2*bx2 + bx2*bx2 
    y_pow4[0,1,:] = torch.full_like(ax,4)*(ax2*ax*bx - ax*bx2*bx)
    y_pow4[1,0,:] = ay2*ay2 - torch.full_like(ay,6)*ay2*by2 + by2*by2 
    y_pow4[1,1,:] = torch.full_like(ay,4)*(ay2*ay*by - ay*by2*by)
    
    # moving average filtering
    kernel_ma = torch.full((1,1,M_ma), 1/M_ma, device=y.device, dtype=torch.float32)
    y_conv = torch.empty(4,1,y_pow4.shape[2], device=y.device, dtype=torch.float32)
    y_conv[0,0,:], y_conv[1,0,:], y_conv[2,0,:], y_conv[3,0,:] = y_pow4[0,0,:], y_pow4[0,1,:], y_pow4[1,0,:], y_pow4[1,1,:]
    y_ma = F.conv1d(y_conv,kernel_ma,bias=None,padding=M_ma//2)

    phiX_corr = torch.atan2(y_ma[1,0,:],-y_ma[0,0,:])/4
    diff_phiX = phiX_corr[1:] - phiX_corr[:-1]
    ind_X_pos, ind_X_neg = torch.nonzero(diff_phiX>pi4), torch.nonzero(diff_phiX<-pi4)
    for i in ind_X_pos:     # unwrapping
        phiX_corr[i+1:] -=  pi2
    for j in ind_X_neg:
        phiX_corr[j+1:] +=  pi2
    cos_phiX, sin_phiX = torch.cos(phiX_corr), torch.sin(phiX_corr)

    phiY_corr = torch.atan2(y_ma[3,0,:],-y_ma[2,0,:])/4
    diff_phiY = phiY_corr[1:] - phiY_corr[:-1]
    ind_Y_pos, ind_Y_neg = torch.nonzero(diff_phiY>pi4), torch.nonzero(diff_phiY<-pi4)
    for ii in ind_Y_pos:    # unwrapping 
        phiY_corr[ii+1:] -=  pi2
    for jj in ind_Y_neg:
        phiY_corr[jj+1:] +=  pi2
    cos_phiY, sin_phiY = torch.cos(phiY_corr), torch.sin(phiY_corr)

    # compensating phase offset
    y_corr[0,0,:] = ax*cos_phiX - bx*sin_phiX
    y_corr[0,1,:] = bx*cos_phiX + ax*sin_phiX
    y_corr[1,0,:] = ay*cos_phiY - by*sin_phiY
    y_corr[1,1,:] = by*cos_phiY + ay*sin_phiY
    return y_corr #, phiX_corr, phiY_corr

def SER_IQflip(q, tx): 
    # estimate symbol error rate from estimated a posterioris q
    device = q.device
    num_lev = q.shape[1]//2
    dec = torch.empty_like(tx, device=device, dtype=torch.int16)
    data = torch.empty_like(tx, device=device, dtype=torch.int16)
    data_IQinv = torch.empty_like(data)
    SER = torch.ones(2,2,4, device=device, dtype=torch.float32)
    
    scale = (num_lev-1)/2
    data = torch.round(scale*tx.float()+scale) # decode TX
    data_IQinv[:,0,:], data_IQinv[:,1,:] = data[:,0,:], -(data[:,1,:]-scale*2)  # compensate potential IQ flip
    ### zero phase-shift
    dec[:,0,:], dec[:,1,:] = torch.argmax(q[:,:num_lev,:], dim=1), torch.argmax(q[:,num_lev:,:], dim=1) # hard decision on max(q)
    SER[0,:,0] = torch.mean( ((data - dec).bool().any(dim=1)).to(torch.float), dim=-1, dtype=torch.float32)
    SER[1,:,0] = torch.mean( ((data_IQinv - dec).bool().any(dim=1)).to(torch.float), dim=-1, dtype=torch.float32)
    
    ### pi phase-shift
    dec_pi = -(dec-scale*2)
    SER[0,:,1] = torch.mean( ((data - dec_pi).bool().any(dim=1)).to(torch.float), dim=-1, dtype=torch.float32)
    SER[1,:,1] = torch.mean( ((data_IQinv - dec_pi).bool().any(dim=1)).to(torch.float), dim=-1, dtype=torch.float32)

    ### pi/4 phase-shift
    dec_pi4 = torch.empty_like(dec)
    dec_pi4[:,0,:], dec_pi4[:,1,:] = -(dec[:,1,:]-scale*2), dec[:,0,:]
    SER[0,:,2] = torch.mean( ((data - dec_pi4).bool().any(dim=1)).to(torch.float), dim=-1, dtype=torch.float32)
    SER[1,:,2] = torch.mean( ((data_IQinv - dec_pi4).bool().any(dim=1)).to(torch.float), dim=-1, dtype=torch.float32)

    ### 3pi/4 phase-shift
    dec_3pi4 = -(dec_pi4-scale*2)
    SER[0,:,3] = torch.mean( ((data - dec_3pi4).bool().any(dim=1)).to(torch.float), dim=-1, dtype=torch.float32)
    SER[1,:,3] = torch.mean( ((data_IQinv - dec_3pi4).bool().any(dim=1)).to(torch.float), dim=-1, dtype=torch.float32)

    SER_out = torch.amin(SER, dim=(0,-1))   # choose minimum estimation per polarization
    return SER_out 


def SER_constell_shaping(rx, tx, amp_levels, nu_sc, var): 
    # estimate symbol error rate from output constellation by considering PCS
    device = rx.device
    num_lev = amp_levels.shape[0]
    data = torch.empty_like(tx, device=device, dtype=torch.int32)
    data_IQinv = torch.empty_like(data)
    SER = torch.ones(2,2,4, device=device, dtype=torch.float32)

    # calculate decision boundaries based on PCS
    d_vec = (1+2*nu_sc*var[0])*(amp_levels[:-1] + amp_levels[1:])/2
    d_vec0 = torch.cat(((-Inf*torch.ones(1)),d_vec),dim=0)
    d_vec1 = torch.cat((d_vec,Inf*torch.ones(1)))
    
    scale = (num_lev-1)/2
    data = torch.round(scale*tx.float()+scale).to(torch.int32) # decode TX
    data_IQinv[:,0,:], data_IQinv[:,1,:] = data[:,0,:], -(data[:,1,:]-scale*2)  # compensate potential IQ flip

    rx *= torch.mean(torch.sqrt(tx[:,0,:].float()**2 + tx[:,1,:].float()**2)) /torch.mean(torch.sqrt(rx[:,0,:]**2 + rx[:,1,:]**2)) # normalize constellation output

    ### zero phase-shift  torch.sqrt(2*torch.mean(rx[0,:N*sps:sps]**2))
    SER[0,:,0] = dec_on_bound(rx,data,d_vec0, d_vec1)
    SER[1,:,0] = dec_on_bound(rx,data_IQinv,d_vec0, d_vec1)
    
    ### pi phase-shift
    rx_pi = -(rx).detach().clone()
    SER[0,:,1] = dec_on_bound(rx_pi,data,d_vec0, d_vec1)
    SER[1,:,1] = dec_on_bound(rx_pi,data_IQinv,d_vec0, d_vec1)

    ### pi/4 phase-shift
    rx_pi4 = torch.empty_like(rx)
    rx_pi4[:,0,:], rx_pi4[:,1,:] = -(rx[:,1,:]).detach().clone(), rx[:,0,:]
    SER[0,:,2] = dec_on_bound(rx_pi4,data,d_vec0, d_vec1)
    SER[1,:,2] = dec_on_bound(rx_pi4,data_IQinv,d_vec0, d_vec1)

    ### 3pi/4 phase-shift
    rx_3pi4 = -(rx_pi4).detach().clone()
    SER[0,:,3] = dec_on_bound(rx_3pi4,data,d_vec0, d_vec1)
    SER[1,:,3] = dec_on_bound(rx_3pi4,data_IQinv,d_vec0, d_vec1)

    SER_out = torch.amin(SER, dim=(0,-1))       # choose minimum estimation per polarization
    return SER_out 

def dec_on_bound(rx,tx_int,d_vec0, d_vec1):
    # hard decision based on the decision boundaries d_vec0 (lower) and d_vec1 (upper)
    SER = torch.zeros(rx.shape[0], dtype = torch.float32, device = rx.device)
    
    xI0 = d_vec0.index_select(dim=0,index=tx_int[0,0,:])
    xI1 = d_vec1.index_select(dim=0,index=tx_int[0,0,:])
    corr_xI = torch.bitwise_and((xI0 <= rx[0, 0, :]), (rx[0, 0, :] < xI1))
    xQ0 = d_vec0.index_select(dim=0,index=tx_int[0,1,:])
    xQ1 = d_vec1.index_select(dim=0,index=tx_int[0,1,:])
    corr_xQ = torch.bitwise_and((xQ0 <= rx[0, 1, :]), (rx[0, 1, :] < xQ1))

    yI0 = d_vec0.index_select(dim=0,index=tx_int[1,0,:])
    yI1 = d_vec1.index_select(dim=0,index=tx_int[1,0,:])
    corr_yI = torch.bitwise_and((yI0 <= rx[1, 0, :]), (rx[1, 0, :] < yI1))
    yQ0 = d_vec0.index_select(dim=0,index=tx_int[1,1,:])
    yQ1 = d_vec1.index_select(dim=0,index=tx_int[1,1,:])
    corr_yQ = torch.bitwise_and((yQ0 <= rx[1, 1, :]), (rx[1, 1, :] < yQ1))

    ex, ey = ~(torch.bitwise_and(corr_xI,corr_xQ)), ~(torch.bitwise_and(corr_yI,corr_yQ))   # no error only if both I or Q are correct
    SER[0], SER[1] = torch.sum(ex)/ex.nelement(), torch.sum(ey)/ey.nelement()   # SER = numb. of errors/ num of symbols
    return SER


def find_shift(q, tx, N_shift, amp_levels, pol): 
    # find shiftings in both polarization and time by correlation with expectation of x^I with respect to q
    corr_max = torch.empty(2,2,2, device = q.device, dtype=torch.float32)
    num_lev = q.shape[1]//2
    corr_ind = torch.empty_like(corr_max)
    len_corr = q.shape[-1] 
    amp_mat = amp_levels.repeat(pol,len_corr,1).transpose(1,2)
    E = torch.sum(amp_mat * q[:,:num_lev,:len_corr], dim=1) # calculate expectation E_q(x^I) of in-phase component

    # correlate with (both polarizations) and shifted versions in time --> find max. correlation
    E_mat = torch.empty(2,len_corr,N_shift, device=q.device, dtype=torch.float32)
    for i in range(N_shift):
        E_mat[:,:,i] = torch.roll(E,i-N_shift//2,-1)
    corr_max[0,:,:], corr_ind[0,:,:] = torch.max( torch.abs(tx[:,0,:len_corr].float() @ E_mat), dim=-1)
    corr_max[1,:,:], corr_ind[1,:,:] = torch.max( torch.abs(tx[:,1,:len_corr].float() @ E_mat), dim=-1) 
    corr_max, ind_max = torch.max(corr_max,dim=0); #corr_ind = corr_ind[ind_max]

    ind_XY = torch.zeros(2,device=q.device, dtype=torch.int16); ind_YX = torch.zeros_like(ind_XY)
    ind_XY[0] = corr_ind[ind_max[0,0],0,0]; ind_XY[1] = corr_ind[ind_max[1,1],1,1]
    ind_YX[0] = corr_ind[ind_max[0,1],0,1]; ind_YX[1] = corr_ind[ind_max[1,0],1,0] 

    if (corr_max[0,0]+corr_max[1,1]) >= (corr_max[0,1]+corr_max[1,0]):
        return N_shift//2-ind_XY, 0
    else:
        return N_shift//2-ind_YX, 1

def find_shift_symb_full(rx, tx, N_shift): 
    # find shiftings in both polarization and time by correlation with the constellation output's in-phase component x^I 
    corr_max = torch.empty(2,2,2, device = rx.device, dtype=torch.float32)
    corr_ind = torch.empty_like(corr_max)
    len_corr = rx.shape[-1] #torch.max(q.shape[-1],1000)
    E = rx[:,0,:len_corr] 

    # correlate with (both polarizations) and shifted versions in time --> find max. correlation
    E_mat = torch.empty(2,len_corr,N_shift, device=rx.device, dtype=torch.float32)
    for i in range(N_shift):
        E_mat[:,:,i] = torch.roll(E,i-N_shift//2,-1)
    corr_max[0,:,:], corr_ind[0,:,:] = torch.max( torch.abs(tx[:,0,:len_corr].float() @ E_mat), dim=-1)
    corr_max[1,:,:], corr_ind[1,:,:] = torch.max( torch.abs(tx[:,1,:len_corr].float() @ E_mat), dim=-1)
    corr_max, ind_max = torch.max(corr_max,dim=0); 

    ind_XY = torch.zeros(2,device=rx.device, dtype=torch.int16); ind_YX = torch.zeros_like(ind_XY)
    ind_XY[0] = corr_ind[ind_max[0,0],0,0]; ind_XY[1] = corr_ind[ind_max[1,1],1,1]
    ind_YX[0] = corr_ind[ind_max[0,1],0,1]; ind_YX[1] = corr_ind[ind_max[1,0],1,0] 
    
    if (corr_max[0,0]+corr_max[1,1]) >= (corr_max[0,1]+corr_max[1,0]):
        return N_shift//2-ind_XY, 0
    else:
        return N_shift//2-ind_YX, 1


def CMA(Rx, R, h, lr, sps, eval):
    device = Rx.device
    M = h.shape[-1]
    N = Rx.shape[-1]
    mh = M//2       

    # zero-padding
    pad = torch.zeros(2,2,mh, device = device, dtype=torch.float32)
    y = torch.cat((pad, Rx, pad), -1)
    y /= torch.mean(y[:,0,:]**2 + y[:,1,:]**2 )     # scaling for mapping to R=1

    out = torch.zeros(2,2,N//sps, device = device, dtype=torch.float32)
    e = torch.empty(N//sps, 2, device = device, dtype=torch.float32)

    for i in torch.arange(mh,N+mh,sps):     # downsampling included
        ind = torch.arange(-mh+i,i+mh+1)
        k = i//sps - mh

        # 2x2 butterfly FIR
        out[0,0,k] = torch.matmul(y[0,0,ind],h[0,0,0,:]) - torch.matmul(y[0,1,ind],h[0,0,1,:]) + torch.matmul(y[1,0,ind],h[0,1,0,:]) - torch.matmul(y[1,1,ind],h[0,1,1,:])  # Estimate Symbol 
        out[1,0,k] = torch.matmul(y[0,0,ind],h[1,0,0,:]) - torch.matmul(y[0,1,ind],h[1,0,1,:]) + torch.matmul(y[1,0,ind],h[1,1,0,:]) - torch.matmul(y[1,1,ind],h[1,1,1,:]) 
        
        out[0,1,k] = torch.matmul(y[0,0,ind],h[0,0,1,:]) + torch.matmul(y[0,1,ind],h[0,0,0,:]) + torch.matmul(y[1,0,ind],h[0,1,1,:]) + torch.matmul(y[1,1,ind],h[0,1,0,:])  # Estimate Symbol 
        out[1,1,k] = torch.matmul(y[0,0,ind],h[1,0,1,:]) + torch.matmul(y[0,1,ind],h[1,0,0,:]) + torch.matmul(y[1,0,ind],h[1,1,1,:]) + torch.matmul(y[1,1,ind],h[1,1,0,:]) 

        e[k,0] = R - out[0,0,k]**2 - out[0,1,k]**2     # Calculate error
        e[k,1] = R - out[1,0,k]**2 - out[1,1,k]**2
        
        if eval == True:
            h[0,0,0,:] += 2*lr*e[k,0]* ( out[0,0,k]*y[0,0,ind] + out[0,1,k]*y[0,1,ind])       # Update filters
            h[0,0,1,:] += 2*lr*e[k,0]* ( out[0,1,k]*y[0,0,ind] - out[0,0,k]*y[0,1,ind])
            h[0,1,0,:] += 2*lr*e[k,0]* ( out[0,0,k]*y[1,0,ind] + out[0,1,k]*y[1,1,ind])
            h[0,1,1,:] += 2*lr*e[k,0]* ( out[0,1,k]*y[1,0,ind] - out[0,0,k]*y[1,1,ind])

            h[1,0,0,:] += 2*lr*e[k,1]* ( out[1,0,k]*y[0,0,ind] + out[1,1,k]*y[0,1,ind]) 
            h[1,0,1,:] += 2*lr*e[k,1]* ( out[1,1,k]*y[0,0,ind] - out[1,0,k]*y[0,1,ind])
            h[1,1,0,:] += 2*lr*e[k,1]* ( out[1,0,k]*y[1,0,ind] + out[1,1,k]*y[1,1,ind])
            h[1,1,1,:] += 2*lr*e[k,1]* ( out[1,1,k]*y[1,0,ind] - out[1,0,k]*y[1,1,ind])
    return out, h, e

def CMAbatch(Rx, R, h, lr, batchlen, sps, eval):
    device = Rx.device
    M = h.shape[-1]
    N = Rx.shape[-1]
    mh = M//2       

    # zero-padding
    pad = torch.zeros(2,2,mh, device = device, dtype=torch.float32)
    y = torch.cat((pad, Rx, pad), -1)
    y /= torch.mean(y[:,0,:]**2 + y[:,1,:]**2 )

    out = torch.zeros(2,2,N//sps, device = device, dtype=torch.float32)
    e = torch.empty(N//sps, 2, device = device, dtype=torch.float32)

    buf = torch.empty(2,2,2,N//sps, M, device = device, dtype=torch.float32)


    for i in torch.arange(mh,N+mh,sps):     # downsampling included
        ind = torch.arange(-mh+i,i+mh+1)
        k = i//sps - mh

        # 2x2 butterfly FIR
        out[0,0,k] = torch.matmul(y[0,0,ind],h[0,0,0,:]) - torch.matmul(y[0,1,ind],h[0,0,1,:]) + torch.matmul(y[1,0,ind],h[0,1,0,:]) - torch.matmul(y[1,1,ind],h[0,1,1,:])  # Estimate Symbol 
        out[1,0,k] = torch.matmul(y[0,0,ind],h[1,0,0,:]) - torch.matmul(y[0,1,ind],h[1,0,1,:]) + torch.matmul(y[1,0,ind],h[1,1,0,:]) - torch.matmul(y[1,1,ind],h[1,1,1,:]) 
        
        out[0,1,k] = torch.matmul(y[0,0,ind],h[0,0,1,:]) + torch.matmul(y[0,1,ind],h[0,0,0,:]) + torch.matmul(y[1,0,ind],h[0,1,1,:]) + torch.matmul(y[1,1,ind],h[0,1,0,:])  # Estimate Symbol 
        out[1,1,k] = torch.matmul(y[0,0,ind],h[1,0,1,:]) + torch.matmul(y[0,1,ind],h[1,0,0,:]) + torch.matmul(y[1,0,ind],h[1,1,1,:]) + torch.matmul(y[1,1,ind],h[1,1,0,:]) 

        e[k,0] = R - out[0,0,k]**2 - out[0,1,k]**2     # Calculate error
        e[k,1] = R - out[1,0,k]**2 - out[1,1,k]**2

        if eval == True:
            
            buf[0,0,0,k,:] = out[0,0,k]*y[0,0,ind] + out[0,1,k]*y[0,1,ind]          # buffering the filter update increments 
            buf[0,0,1,k,:] = out[0,1,k]*y[0,0,ind] - out[0,0,k]*y[0,1,ind]
            buf[0,1,0,k,:] = out[0,0,k]*y[1,0,ind] + out[0,1,k]*y[1,1,ind]
            buf[0,1,1,k,:] = out[0,1,k]*y[1,0,ind] - out[0,0,k]*y[1,1,ind]

            buf[1,0,0,k,:] = out[1,0,k]*y[0,0,ind] + out[1,1,k]*y[0,1,ind] 
            buf[1,0,1,k,:] = out[1,1,k]*y[0,0,ind] - out[1,0,k]*y[0,1,ind]
            buf[1,1,0,k,:] = out[1,0,k]*y[1,0,ind] + out[1,1,k]*y[1,1,ind]
            buf[1,1,1,k,:] = out[1,1,k]*y[1,0,ind] - out[1,0,k]*y[1,1,ind]      

            if (k%batchlen == 0 and k!=0):  # batch-wise updating
                h[0,0,0,:] += 2*lr*torch.sum(torch.mul(buf[0,0,0,k-batchlen:k,:].T,e[k-batchlen:k,0]),dim=1)       # Update filters
                h[0,0,1,:] += 2*lr*torch.sum(torch.mul(buf[0,0,1,k-batchlen:k,:].T,e[k-batchlen:k,0]),dim=1) 
                h[0,1,0,:] += 2*lr*torch.sum(torch.mul(buf[0,1,0,k-batchlen:k,:].T,e[k-batchlen:k,0]),dim=1) 
                h[0,1,1,:] += 2*lr*torch.sum(torch.mul(buf[0,1,1,k-batchlen:k,:].T,e[k-batchlen:k,0]),dim=1) 

                h[1,0,0,:] += 2*lr*torch.sum(torch.mul(buf[1,0,0,k-batchlen:k,:].T,e[k-batchlen:k,1]),dim=1) 
                h[1,0,1,:] += 2*lr*torch.sum(torch.mul(buf[1,0,1,k-batchlen:k,:].T,e[k-batchlen:k,1]),dim=1) 
                h[1,1,0,:] += 2*lr*torch.sum(torch.mul(buf[1,1,0,k-batchlen:k,:].T,e[k-batchlen:k,1]),dim=1) 
                h[1,1,1,:] += 2*lr*torch.sum(torch.mul(buf[1,1,1,k-batchlen:k,:].T,e[k-batchlen:k,1]),dim=1) 
    return out, h, e

def CMAflex(Rx, R, h, lr, batchlen, symb_step, sps, eval):
    device = Rx.device
    M = h.shape[-1]
    N = Rx.shape[-1]
    mh = M//2       

    # zero-padding
    pad = torch.zeros(2,2,mh, device = device, dtype=torch.float32)
    y = torch.cat((pad, Rx, pad), -1)
    y /= torch.mean(y[:,0,:]**2 + y[:,1,:]**2 )

    out = torch.zeros(2,2,N//sps, device = device, dtype=torch.float32)
    e = torch.empty(N//sps, 2, device = device, dtype=torch.float32)

    buf = torch.empty(2,2,2,N//sps, M, device = device, dtype=torch.float32)


    for i in torch.arange(mh,N+mh,sps):     # downsampling included
        ind = torch.arange(-mh+i,i+mh+1)
        k = i//sps - mh

        out[0,0,k] = torch.matmul(y[0,0,ind],h[0,0,0,:]) - torch.matmul(y[0,1,ind],h[0,0,1,:]) + torch.matmul(y[1,0,ind],h[0,1,0,:]) - torch.matmul(y[1,1,ind],h[0,1,1,:])  # Estimate Symbol 
        out[1,0,k] = torch.matmul(y[0,0,ind],h[1,0,0,:]) - torch.matmul(y[0,1,ind],h[1,0,1,:]) + torch.matmul(y[1,0,ind],h[1,1,0,:]) - torch.matmul(y[1,1,ind],h[1,1,1,:]) 
        
        out[0,1,k] = torch.matmul(y[0,0,ind],h[0,0,1,:]) + torch.matmul(y[0,1,ind],h[0,0,0,:]) + torch.matmul(y[1,0,ind],h[0,1,1,:]) + torch.matmul(y[1,1,ind],h[0,1,0,:])  # Estimate Symbol 
        out[1,1,k] = torch.matmul(y[0,0,ind],h[1,0,1,:]) + torch.matmul(y[0,1,ind],h[1,0,0,:]) + torch.matmul(y[1,0,ind],h[1,1,1,:]) + torch.matmul(y[1,1,ind],h[1,1,0,:]) 

        e[k,0] = R - out[0,0,k]**2 - out[0,1,k]**2     # Calculate error
        e[k,1] = R - out[1,0,k]**2 - out[1,1,k]**2

        if eval == True:

            buf[0,0,0,k,:] = out[0,0,k]*y[0,0,ind] + out[0,1,k]*y[0,1,ind]          # buffering the filter update increments 
            buf[0,0,1,k,:] = out[0,1,k]*y[0,0,ind] - out[0,0,k]*y[0,1,ind]
            buf[0,1,0,k,:] = out[0,0,k]*y[1,0,ind] + out[0,1,k]*y[1,1,ind]
            buf[0,1,1,k,:] = out[0,1,k]*y[1,0,ind] - out[0,0,k]*y[1,1,ind]

            buf[1,0,0,k,:] = out[1,0,k]*y[0,0,ind] + out[1,1,k]*y[0,1,ind] 
            buf[1,0,1,k,:] = out[1,1,k]*y[0,0,ind] - out[1,0,k]*y[0,1,ind]
            buf[1,1,0,k,:] = out[1,0,k]*y[1,0,ind] + out[1,1,k]*y[1,1,ind]
            buf[1,1,1,k,:] = out[1,1,k]*y[1,0,ind] - out[1,0,k]*y[1,1,ind]      

            if (k%symb_step == 0 and k>=batchlen): # batch-wise updating with flexible step length
                h[0,0,0,:] += 2*lr*torch.sum(torch.mul(buf[0,0,0,k-batchlen:k,:].T,e[k-batchlen:k,0]),dim=1)       # Update filters
                h[0,0,1,:] += 2*lr*torch.sum(torch.mul(buf[0,0,1,k-batchlen:k,:].T,e[k-batchlen:k,0]),dim=1) 
                h[0,1,0,:] += 2*lr*torch.sum(torch.mul(buf[0,1,0,k-batchlen:k,:].T,e[k-batchlen:k,0]),dim=1) 
                h[0,1,1,:] += 2*lr*torch.sum(torch.mul(buf[0,1,1,k-batchlen:k,:].T,e[k-batchlen:k,0]),dim=1) 

                h[1,0,0,:] += 2*lr*torch.sum(torch.mul(buf[1,0,0,k-batchlen:k,:].T,e[k-batchlen:k,1]),dim=1) 
                h[1,0,1,:] += 2*lr*torch.sum(torch.mul(buf[1,0,1,k-batchlen:k,:].T,e[k-batchlen:k,1]),dim=1) 
                h[1,1,0,:] += 2*lr*torch.sum(torch.mul(buf[1,1,0,k-batchlen:k,:].T,e[k-batchlen:k,1]),dim=1) 
                h[1,1,1,:] += 2*lr*torch.sum(torch.mul(buf[1,1,1,k-batchlen:k,:].T,e[k-batchlen:k,1]),dim=1) 
    return out, h, e

class twoXtwoFIR(nn.Module):        # complex-valued 2x2 butterfly FIR filter
    def __init__(self, M_est, sps):
        super(twoXtwoFIR, self).__init__()
        # implemented with real-valued input and output ---> 2*2=4 input, 1*2 output per conv-layer (seperately for I and Q output)
        self.conv_w = nn.Conv1d(4, 2, M_est, bias=False, padding=M_est//2, stride=sps).to(dtype=torch.float32) #actually, not a conv but xcorr
        nn.init.dirac_(self.conv_w.weight)    # Dirac-initilization

        self.sm = nn.Softmin(dim=0)     # soft demapper -- softmin includes minus in exponent


    def forward(self, x, amp_levels, var, nu_sc):
        dev, dty = x.device, x.dtype

        # reshape input, negative for correct complex convolution: y_I = h_I*x_I - h_Q*x_Q -> -x_Q
        x_in_I = torch.empty(1,4,x.shape[-1], device=dev, dtype=dty)
        x_in_I[0,:2,:], x_in_I[0,2:,:] = x[:,0,:], -x[:,1,:]
        x_in_Q = torch.empty_like(x_in_I)
        x_in_Q[0,:2,:], x_in_Q[0,2:,:] = -x_in_I[0,2:,:], x_in_I[0,:2,:]

        out_I, out_Q = self.conv_w(x_in_I).squeeze(), self.conv_w(x_in_Q).squeeze()

        N = out_I.shape[-1]
        n = amp_levels.shape[0]
        q_est = torch.empty(2,2*n,N, device=dev, dtype=dty)
        out = torch.empty(2,2,N, device=dev, dtype=dty)
        amp_lev_mat = amp_levels.repeat(N,1).transpose(0,1)
        amp_lev_mat_sq = amp_lev_mat**2

        out[:,0,:], out[:,1,:] = out_I, out_Q
        # Soft demapping
        # correction term for PCS:  + nu_sc * amp_levels**2 -- see SD-FEC in Junho Cho, "Probabilistic Constellation Shaping for OpticalFiber Communications"
        q_est[0, :n, :], q_est[0, n:, :] = self.sm((out_I[0, :]-amp_lev_mat)**2/2/var[0] + nu_sc * amp_lev_mat_sq), self.sm(
            (out_Q[0, :]-amp_lev_mat)**2/2/var[0] + nu_sc * amp_lev_mat_sq)
        q_est[1,:n,:], q_est[1,n:,:]= self.sm( (out_I[1,:]-amp_lev_mat)**2/2/var[1]  + nu_sc * amp_lev_mat_sq), self.sm( (out_Q[1,:]-amp_lev_mat)**2/2/var[1]  + nu_sc * amp_lev_mat_sq)

        #q_est[0,:n,:], q_est[0,n:,:]= self.sm( (out_I[0,:]-amp_lev_mat)**2/2/var[0] ), self.sm( (out_Q[0,:]-amp_lev_mat)**2/2/var[0]  )
        #q_est[1,:n,:], q_est[1,n:,:]= self.sm( (out_I[1,:]-amp_lev_mat)**2/2/var[1]  ), self.sm( (out_Q[1,:]-amp_lev_mat)**2/2/var[1] )
        return q_est, out 

def soft_dec(out, var, amp_levels, nu_sc): 
    # Soft demapping with correction term for PCS:  + nu_sc * amp_levels**2 -- see SD-FEC in Junho Cho, "Probabilistic Constellation Shaping for OpticalFiber Communications"  
    dev, dty = out.device, out.dtype
    N = out.shape[-1]
    n = amp_levels.shape[0]
    q_est = torch.empty(2,2*n,N, device=dev, dtype=dty)
    amp_lev_mat = amp_levels.repeat(N,1).transpose(0,1)
    amp_lev_mat_sq = amp_lev_mat**2
    out_I, out_Q = out[:,0,:], out[:,1,:]

    sm = nn.Softmin(dim=0)
    q_est[0,:n,:], q_est[0,n:,:]= sm( (out_I[0,:]-amp_lev_mat)**2/2/var[0] + nu_sc * amp_lev_mat_sq), sm( (out_Q[0,:]-amp_lev_mat)**2/2/var[0] + nu_sc * amp_lev_mat_sq)
    q_est[1,:n,:], q_est[1,n:,:]= sm( (out_I[1,:]-amp_lev_mat)**2/2/var[1] + nu_sc * amp_lev_mat_sq), sm( (out_Q[1,:]-amp_lev_mat)**2/2/var[1] + nu_sc * amp_lev_mat_sq)
    return q_est

def init(channel, mod, device, nu, sps, M_est, SNR):
    if channel == 'h1':     # h_1 in Caciularu et al.
        h_channel_orig= np.array([0.0545+1j*0.05, 0.2823-1j*0.11971, -0.7676+1j*0.2788, -0.0641-1j*0.0576, 0.0466-1j*0.02275]).astype(np.complex64)
    elif channel == 'h2':   # h_1 in Caciularu et al.
        h_channel_orig= np.array([0.0545+1j*0.0165, -1.3449-1j*0.4523, 1.0067+1j*1.1524, 0.3476+1j*0.3153]).astype(np.complex64)
    elif channel == 'h0':   # only optical channel model, no further IR
        h_channel_orig= np.array([1]).astype(np.complex64)      

    h_channel = np.zeros((sps*(h_channel_orig.shape[-1]-1)+1), dtype=np.complex64)
    h_channel[0::sps] = h_channel_orig        # upsampling channel IR by inserting zeros
    h_channel /= np.linalg.norm(h_channel)    # Normalization of the channel

    constellations = {'4-QAM': np.array([-1,-1,1,1]) + 1j*np.array([-1,1,-1,1]), 
                '16-QAM': np.array([-3,-3,-3,-3,-1,-1,-1,-1,1,1,1,1,3,3,3,3]) + 1j*np.array([-3,-1,1,3,-3,-1,1,3,-3,-1,1,3,-3,-1,1,3]),
                '64-QAM': np.array([-7,-7,-7,-7,-7,-7,-7,-7,-5,-5,-5,-5,-5,-5,-5,-5,-3,-3,-3,-3,-3,-3,-3,-3,-1,-1,-1,-1,-1,-1,-1,-1,1,1,1,1,1,1,1,1,3,3,3,3,3,3,3,3,5,5,5,5,5,5,5,5,7,7,7,7,7,7,7,7]) \
                + 1j*np.array([-7,-5,-3,-1,1,3,5,7,-7,-5,-3,-1,1,3,5,7,-7,-5,-3,-1,1,3,5,7,-7,-5,-3,-1,1,3,5,7,-7,-5,-3,-1,1,3,5,7,-7,-5,-3,-1,1,3,5,7,-7,-5,-3,-1,1,3,5,7,-7,-5,-3,-1,1,3,5,7])}

    pol = 2     # number of channels (polarizations)

    constellation = constellations[mod] / np.sqrt(np.mean(np.abs(constellations[mod])**2))  # normalize modulation format

    amp_levels = constellation.real         # ASK levels (poitive and negative amplitude levels)
    num_lev = int(np.sqrt(len(amp_levels))) # number of ASK levels
    amps = amp_levels[::num_lev]            # amplitude levels
    amp_levels = torch.tensor(amps, device=device, dtype=torch.float32)
    sc = np.min(np.abs(amps))               # scaling factor for having lowest level equal 1
    nu_sc = nu/sc**2                        # re-scaled shaping factor

    P = np.exp(-nu*np.abs(amps/sc)**2); P = P / np.sum(P)   # pmf of the amlitude levels

    shape_mat = np.zeros((num_lev,num_lev))
    for i in range(num_lev):
        shape_mat[i,:] = P
    P_mat = (shape_mat*shape_mat.T)/np.sum(shape_mat*shape_mat.T)   # matrix with the corresponding probabilities for each constellation point
    # H_P = -np.sum(np.log2(P_mat)*P_mat)       # entrpy of the modulation format
    pow_mean = np.sum(P_mat.reshape(-1)* np.abs(constellation)**2)  # mean power of the constellation

    var = torch.full((2,),pow_mean/10**(SNR/10)/2 , device=device, dtype=torch.float32)     # noise variance for the soft demapper

    h_est = np.zeros([pol,pol,2,M_est])     # initialize estimated impulse response
    #h_est[:,:,:,M_est//2] = 0.1
    h_est[0,0,0,M_est//2], h_est[1,1,0,M_est//2] = 1, 1     # Dirac initialization
    h_est = torch.tensor(h_est,requires_grad=True, dtype=torch.float32, device=device)

    return h_est, h_channel, P, amp_levels, amps, pol, nu_sc, var, pow_mean

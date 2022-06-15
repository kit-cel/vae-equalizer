################################################################################################
# Author:        Vincent Lauinger, 
# Affiliation:   Communications Engineering Lab (CEL), Karlsruhe Institute of Technology (KIT)
# Contact:       vincent.lauinger@kit.edu
# Last revision: 15th of June 2022
################################################################################################

import numpy as np
import torch
import torch.nn.functional as F
   
import matplotlib.pyplot as plt

torch.set_grad_enabled(False)

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

def generate_data(N, M, amps, SNR, h_channel, sps, device, P):
    T = 8 # length of pulse-shaping filter in symbols
    beta = 0.1 # foll-off factor

    N_conv = N+len(h_channel)+4*T
    tx_up = np.zeros(sps*(N_conv-1)+1, dtype=np.complex64)

    rng = np.random.default_rng()
    data = rng.choice(amps, (2,N_conv), p=P)   
    tx_sig = data[0,:] + 1j*data[1,:]   # applying moduation
    tx_up[::sps] = tx_sig       # sps-upsampled signal by zero-insertion

    h_pulse = rrcfir(T,sps,beta)
    rx_sig = np.convolve(tx_up, h_pulse, mode='valid') 
    rx_sig = np.convolve(rx_sig, h_channel, mode='valid') 

    sigma_n = np.sqrt( sps*np.mean(np.abs(rx_sig)**2)/2  /10**(SNR/10) )
    rx_sig += sigma_n* (np.random.randn(*rx_sig.shape) +1j*np.random.randn(*rx_sig.shape))

    rx_tensor = torch.from_numpy(np.asarray([np.real(rx_sig[:sps*N]),np.imag(rx_sig[:sps*N])])).to(device,torch.float32)     
    data_tensor = torch.from_numpy(np.asarray([data[0,(T+M-1):(N+T+M-1)],data[1,(T+M-1):(N+T+M-1)]])).to(device,torch.float16)
    
    return rx_tensor, data_tensor

def SER_CMA(rx, tx, sps, amp_levels, num_lev, device): 
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

    return torch.min(SER)   # SER = BER for BPSK


def SER_symb(rx, tx, sps, amp_levels, num_lev, device): 
    N = tx.shape[1]
    amp_lev_mat = amp_levels.repeat(N,1).transpose(0,1)

    dec = torch.empty_like(tx, device=device, dtype=torch.int16)
    data = torch.empty_like(tx, device=device, dtype=torch.int16)
    SER = torch.ones(4, device=device, dtype=torch.float32)
    
    scale = (num_lev-1)/2
    data[:,:] = torch.round(scale*tx.float()+scale)
    sig_I, sig_Q = rx[0,:N*sps:sps] / torch.sqrt(2*torch.mean(rx[0,:N*sps:sps]**2)), rx[1,:N*sps:sps] / torch.sqrt(2*torch.mean(rx[1,:N*sps:sps]**2))
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

    return torch.min(SER)   # SER = BER for BPSK

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

def CMA(Rx, R, h, lr, sps, eval):
    device = Rx.device
    M = h.shape[1]
    N = Rx.shape[1]
    mh = M//2       

    # zero-padding
    pad = torch.zeros(2,mh, device = device, dtype=torch.float32)
    y = torch.cat((pad, Rx, pad), 1)

    out = torch.zeros(2,N//sps, device = device, dtype=torch.float32)
    e = torch.empty(N//sps, device = device, dtype=torch.float32)

    for i in torch.arange(mh,N+mh,sps):     # downsampling included
        ind = torch.arange(-mh+i,i+mh+1)
        k = i//sps - mh

        out[0,k] = torch.matmul(y[0,ind],h[0,:]) - torch.matmul(y[1,ind],h[1,:])     # Estimate Symbol 
        out[1,k] = torch.matmul(y[0,ind],h[1,:]) + torch.matmul(y[1,ind],h[0,:]) 

        e[k] = R - out[0,k]**2 - out[1,k]**2     # Calculate error
        
        if eval == True:
            h[0,:] += 2*lr*e[k]* ( out[0,k]*y[0,ind] + out[1,k]*y[1,ind] )       # Update filters
            h[1,:] += 2*lr*e[k]* ( out[1,k]*y[0,ind] - out[0,k]*y[1,ind] )
        
    return out, h, e

def CPE(y):
    #pi = torch.full([1],3.141592653589793, device=y.device, dtype=torch.float32)
    M_ma = 501
    y_corr = torch.zeros_like(y)
    y_pow4 = torch.zeros_like(y)
    
    # # (a+jb)^4 = a^4 - 6a^2b^2 + b^4 +j(4a^3b - 4ab^3)
    a, b = y[0,:], y[1,:]
    a2, b2 = a**2, b**2
    
    y_pow4[0,:], y_pow4[1,:] = a2**2 - 6*a2*b2 + b2**2,     4*(a2*a*b - a*b2*b)
    
        # implement Moving Average-filter

    kernel_ma = torch.full((1,1,M_ma), 1/M_ma, device=y.device, dtype=torch.float32)
    y_conv = torch.empty(2,1,y_pow4.shape[1], device=y.device, dtype=torch.float32)
    y_conv[0,0,:], y_conv[1,0,:] = y_pow4[0,:], y_pow4[1,:]
    y_ma = F.conv1d(y_conv,kernel_ma,bias=None,padding=M_ma//2)
    
    phi = torch.atan2(y_ma[1,0,:],-y_ma[0,0,:])/4
    
    cos_phi, sin_phi = torch.cos(phi), torch.sin(phi)

    y_corr[0,:] = a*cos_phi - b*sin_phi
    y_corr[1,:] = b*cos_phi + a*sin_phi

    return y_corr       #, phi


################################## MAIN FUNCTION ##################################

def processing(mod, sps, SNR, nu, M_est, lr_optim, N_valid, N_train, num_epochs, epe, channel):
    device = 'cpu' if torch.cuda.is_available() else 'cpu'
    print("We are using the following device for learning:",device)

    if channel == 'h1':
        h_channel_orig= np.array([0.0545+1j*0.05, 0.2823-1j*0.11971, -0.7676+1j*0.2788, -0.0641-1j*0.0576, 0.0466-1j*0.02275]).astype(np.complex64)
    elif channel == 'h2':
        h_channel_orig= np.array([0.0545+1j*0.0165, -1.3449-1j*0.4523, 1.0067+1j*1.1524, 0.3476+1j*0.3153]).astype(np.complex64)
    #h_channel= np.array([0,0+1j*0, 0+1j*0, 1+1j*0, 0+1j*0, 0+1j*0,0]).astype(np.complex64)
    M = len(h_channel_orig)

    constellations = {'4-QAM': np.array([-1,-1,1,1]) + 1j*np.array([-1,1,-1,1]), 
                '16-QAM': np.array([-3,-3,-3,-3,-1,-1,-1,-1,1,1,1,1,3,3,3,3]) + 1j*np.array([-3,-1,1,3,-3,-1,1,3,-3,-1,1,3,-3,-1,1,3]),
                '64-QAM': np.array([-7,-7,-7,-7,-7,-7,-7,-7,-5,-5,-5,-5,-5,-5,-5,-5,-3,-3,-3,-3,-3,-3,-3,-3,-1,-1,-1,-1,-1,-1,-1,-1,1,1,1,1,1,1,1,1,3,3,3,3,3,3,3,3,5,5,5,5,5,5,5,5,7,7,7,7,7,7,7,7]) \
                + 1j*np.array([-7,-5,-3,-1,1,3,5,7,-7,-5,-3,-1,1,3,5,7,-7,-5,-3,-1,1,3,5,7,-7,-5,-3,-1,1,3,5,7,-7,-5,-3,-1,1,3,5,7,-7,-5,-3,-1,1,3,5,7,-7,-5,-3,-1,1,3,5,7,-7,-5,-3,-1,1,3,5,7])}


    h_channel = np.zeros((sps*(h_channel_orig.shape[-1]-1)+1), dtype=np.complex64)
    h_channel[0::sps] = h_channel_orig
    h_channel /= np.linalg.norm(h_channel)    # Normalisation of the channel

    constellation = constellations[mod] / np.sqrt(np.mean(np.abs(constellations[mod])**2))
    
    amp_levels = constellation.real
    num_lev = int(np.sqrt(len(amp_levels)))
    amps = amp_levels[::num_lev]
    amp_levels = torch.tensor(amps, device=device, dtype=torch.float32)
    sc_amp = np.min(np.abs(amps))
    amps_scale = amps/sc_amp

    P = np.exp(-nu*np.abs(amps_scale)**2); P = P / np.sum(P)

    h_est = np.zeros([2,M_est])  #h_est = np.zeros([2,len(h_channel)]) 
    h_est[0,M_est//2] = 1 #h_est[0,len(h_channel)//2] = 1
    h_est = torch.tensor(h_est,requires_grad=True, dtype=torch.float32, device=device)

    SER_valid = torch.empty(num_epochs//epe, device=device, dtype=torch.float32)

    R = 1
    with torch.set_grad_enabled(False):  
        for epoch in range(num_epochs):
            rx_tensor, data_tensor = generate_data(N_train, M, amps, SNR, h_channel, sps, device, P)
            output, h_est, e = CMA(rx_tensor, R, h_est, lr_optim, sps, True) 
            loss = torch.mean(torch.abs(e)) 
                
            if epoch % epe == 0:
                rx_tensor_valid, data_tensor_valid = generate_data(N_valid, M, amps, SNR, h_channel, sps, device, P) 
                output_valid, h_est, e_valid = CMA(rx_tensor_valid, R, h_est, lr_optim, sps, False) 
                output_CPE = CPE(output_valid)  
                
                # SER_orig = SER_symb(rx_tensor_valid[:,M:],data_tensor_valid[:,:-M], sps, amp_levels, num_lev, device)
                shift = find_shift_symb(output_CPE,data_tensor_valid, 21)
                SER_valid[epoch//epe] = SER_CMA(output_CPE[:,11+shift:-11], data_tensor_valid[:,11:-11-shift], sps, amp_levels, num_lev, device) 

                print(epoch, loss.item(), shift.item(), '\t\t\t\t\t\tSER = ',  SER_valid[epoch//epe].item()  ) #, '\t\t\t\t\t\t(unproc. SER = ',  SER_orig.item(),')'   )   #, '\n h = ', h_est.tolist())
    return SER_valid


################################## PLOTTING FUNCTIONS ##################################

def create_constellation_plot(E):
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

def plot_constellation():   # scatter plot of the expectation
    create_constellation_plot(output_CPE)


def plot_correlation():
    data = data_tensor_valid[1,:1000-1].detach().cpu().numpy()
    corr = np.correlate(output_CPE[1,1:1000].detach().cpu().numpy(),data,"same")
    print(np.argmax(np.abs(corr)))
    plt.plot(corr)
    plt.show()
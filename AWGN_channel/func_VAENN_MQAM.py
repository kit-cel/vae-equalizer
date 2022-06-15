################################################################################################
# Author:        Vincent Lauinger, 
# Affiliation:   Communications Engineering Lab (CEL), Karlsruhe Institute of Technology (KIT)
# Contact:       vincent.lauinger@kit.edu
# Last revision: 15th of June 2022
################################################################################################

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
  
import matplotlib.pyplot as plt

################################## HELPER FUNCTIONS ##################################
def rcfir(T,sps, beta): # raised cosine filter
    #T = 6 # pulse duration in symbols
    #sps = 2 # oversampling factor in samples per symbol
    #beta = 0.1 # roll-off factor
    t = np.arange(-T*sps/2,T*sps/2, 1/sps, dtype=np.float32)
    h = np.sinc(t)*np.cos(np.pi*beta*t)/(1-(2*beta*t)**2)
    h[np.abs(t)==1/2/beta] = np.pi/4*np.sinc(1/(2*beta))
    h = h/np.linalg.norm(h)    # Normalisation of the pulseforming filter
    return h

def rrcfir(T,sps, beta): # root-raised cosine filter
    #T = 6 # pulse duration in symbols
    #sps = 2 # oversampling factor in samples per symbol
    #beta = 0.1 # roll-off factor
    t = np.arange(-T*sps/2,T*sps/2, 1/sps, dtype=np.float32)
    h = (np.sin(np.pi*t*(1-beta)) + 4*beta*t*np.cos(np.pi*t*(1+beta)) ) / (np.pi*t*(1-(4*beta*t)**2))
    h[np.abs(t)==1/4/beta] = beta/np.sqrt(2) * ( (1+2/np.pi) * np.sin(np.pi/4/beta) + (1-2/np.pi) * np.cos(np.pi/4/beta) )
    h[t==0] = (1 + beta*(4/np.pi - 1))
    h = h/np.linalg.norm(h)    # Normalisation of the pulseforming filter
    return h

def generate_data(N, M, constellation, SNR, h_channel, sps, device):
    T = 8 # length of pulse-shaping filter in symbols
    beta = 0.1 # foll-off factor

    N_conv = N+len(h_channel)+4*T
    tx_up = np.zeros(sps*(N_conv-1)+1, dtype=np.complex64)

    data = np.random.randint(len(constellation),size=N_conv)   
    tx_sig = constellation[data]    # applying modulation
    tx_up[::sps] = tx_sig       # sps-upsampled signal by zero-insertion

    h_pulse = rrcfir(T,sps,beta)
    rx_sig = np.convolve(tx_up, h_pulse, mode='valid') # applying pulse shaping
    rx_sig = np.convolve(rx_sig, h_channel, mode='valid') # applying channel h

    sigma_n = np.sqrt(1/2)  /10**(SNR/20)   # noise varaince
    rx_sig += sigma_n* (np.random.randn(*rx_sig.shape) +1j*np.random.randn(*rx_sig.shape)) # applying AWGN
    
    rx_tensor = torch.from_numpy(np.asarray([np.real(rx_sig[:sps*N]),np.imag(rx_sig[:sps*N])])).to(device,torch.float32)      
    data_tensor = torch.from_numpy(np.asarray([np.real(tx_sig[(T+M-1):(N+T+M-1)]),np.imag(tx_sig[(T+M-1):(N+T+M-1)])])).to(device,torch.float16)
    return rx_tensor, data_tensor

def loss_function(q, rx, h, device, amp_levels):
    sps = rx.shape[-1]//q.shape[1]
    N = q.shape[1]*sps
    mh = h.shape[1]//2
    Mh = 2*mh
    num_lev = amp_levels.shape[0]

    Eq = torch.zeros(2,N, device=device, dtype=torch.float32)
    Eq2 = torch.zeros(2,N, device=device, dtype=torch.float32)

    amp_lev_mat = amp_levels.repeat(q.shape[1],2).transpose(0,1)
    temp_lev = amp_lev_mat * q
    temp_pow = (amp_lev_mat**2) * q

    Eq[0,::sps], Eq[1,::sps] = torch.sum(temp_lev[:num_lev,:], dim=0), torch.sum(temp_lev[num_lev:,:], dim=0)
    Eq2[0,::sps], Eq2[1,::sps] = torch.sum(temp_pow[:num_lev,:], dim=0), torch.sum(temp_pow[num_lev:,:], dim=0)
    
    D_real = torch.zeros(N-Mh, device=device, dtype=torch.float32)  
    D_imag = torch.zeros(N-Mh, device=device, dtype=torch.float32)
    E = torch.zeros(N-Mh, device=device, dtype=torch.float32)    
    idx = np.arange(Mh,N)
    
    for j in range(Mh+1):
        D_real += h[0,j] * Eq[0,idx-j] - h[1,j] * Eq[1,idx-j]
        D_imag += h[0,j] * Eq[1,idx-j] + h[1,j] * Eq[0,idx-j]
        E += torch.sum( (h[0,j]**2 + h[1,j]**2) * (Eq2[:,idx-j] - Eq[:,idx-j]**2) , dim=0)
    
    data_entropy = torch.sum( -q[:,mh:-mh]*torch.log(q[:,mh:-mh] + 1e-12) )
    C = torch.sum(rx[:,mh:-mh]**2)
    C += - 2*torch.sum( rx[0,mh:N-mh]*D_real + rx[1,mh:N-mh]*D_imag ) + torch.sum( D_real**2 + D_imag**2 + E )
    retval = (N-Mh)*torch.log(C) - data_entropy
    return retval

def SER_q(q, tx, sps, num_lev, device): 
    N = tx.shape[-1]
    dec = torch.empty_like(tx, device=device, dtype=torch.int16)
    data = torch.empty_like(tx, device=device, dtype=torch.int16)
    SER = torch.ones(4, device=device, dtype=torch.float32)
    
    scale = (num_lev-1)/2
    data[:,:] = torch.round(scale*tx.float()+scale)
    ### zero phase-shift
    dec[0,:], dec[1,:] = torch.argmax(q[:num_lev,:N], dim=0), torch.argmax(q[num_lev:,:N], dim=0)
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

def find_shift(q, tx, N_shift, amp_levels, num_lev, device): 
    amp_mat = amp_levels.repeat(1000,1).transpose(0,1)
    E = torch.sum(amp_mat * q[0,:num_lev,:1000], dim=0)

    E_mat = torch.empty(1000,N_shift, device=device, dtype=torch.float32)

    for i in range(N_shift):
        E_mat[:,i] = torch.roll(E,i-N_shift//2,0)
    corr = torch.matmul(tx[0,:1000].to(torch.float32),E_mat)
    if torch.max(torch.abs(corr)) >= 0.02*q.shape[-1]:
        return N_shift//2-torch.argmax(torch.abs(corr))
    else:
        corr_IQ = torch.matmul(tx[1,:1000].to(torch.float32),E_mat)
        if torch.max(torch.abs(corr_IQ)) >= torch.max(torch.abs(corr)):
            return N_shift//2-torch.argmax(torch.abs(corr_IQ))
        else:
            return N_shift//2-torch.argmax(torch.abs(corr))

class Net(nn.Module):
    def __init__(self, kernel_1, kernel_2, num_lev, sps):
        super(Net, self).__init__()
        self.fc1 = nn.Conv1d(2, 2*num_lev, kernel_1, bias=True, padding=kernel_1 // 2)
        self.fc2 = nn.Conv1d(2*num_lev, 2*num_lev, kernel_2, bias=True, padding=kernel_2 // 2,stride=sps)
        nn.init.xavier_uniform_(self.fc1.weight, gain=1)    # nn.init.kaiming_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight, gain=1)

    def forward(self, x):
        out = self.fc2( F.elu(self.fc1(x)) )
        num_lev = out.shape[1]//2
        N_out = out.shape[2]
        sps = x.shape[-1]//N_out
        x_res = torch.zeros(1,2,N_out, device=x.device, dtype=torch.float32)
        for i in range(sps):
            x_res += x[:,:,i:sps*N_out:sps] / sps
        sm = nn.Softmax(dim=1)
        netout = torch.empty(1,2*num_lev,N_out, device=x.device, dtype=torch.float32)
        netout[0,:num_lev,:], netout[0,num_lev:,:] = sm(out[:,:num_lev,:] + x_res[:,0,:]), sm(out[:,num_lev:,:] + x_res[:,1,:]) 
        return netout

class Net_BN(nn.Module):
    def __init__(self, kernel_1, kernel_2, num_lev, sps):
        super(Net_BN, self).__init__()
        self.fc1 = nn.Conv1d(2, 2*num_lev, kernel_1, bias=True, padding=kernel_1 // 2)
        self.fc2 = nn.Conv1d(2*num_lev, 2*num_lev, kernel_2, bias=True, padding=kernel_2 // 2,stride=sps)
        self.batch1 = nn.BatchNorm1d(2*num_lev)
        nn.init.kaiming_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight, gain=1)

    def forward(self, x):
        out = self.fc2( self.batch1(F.elu(self.fc1(x)) ))
        num_lev = out.shape[1]//2
        N_out = out.shape[2]
        sps = x.shape[-1]//N_out
        x_res = torch.zeros(1,2,N_out, device=x.device, dtype=torch.float32)
        for i in range(sps):
            x_res += x[:,:,i:sps*N_out:sps] / sps
        sm = nn.Softmax(dim=1)
        netout = torch.empty(1,2*num_lev,N_out, device=x.device, dtype=torch.float32)
        netout[0,:num_lev,:], netout[0,num_lev:,:] = sm(out[:,:num_lev,:] + x_res[:,0,:]), sm(out[:,num_lev:,:] + x_res[:,1,:]) 
        return netout

################################## MAIN FUNCTION ##################################

def processing(mod, sps, SNR, M_est, kernel_1,kernel_2,lr_optim, batch_len, N_valid, N_train, num_epochs, epe, channel, net_type):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
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
    amp_levels = torch.tensor(amp_levels[::num_lev], device=device, dtype=torch.float32)

    # CNN
    if net_type == 'Net':
        net = Net(kernel_1, kernel_2, num_lev, sps)
    elif net_type == 'Net_BN':
        net = Net_BN(kernel_1, kernel_2, num_lev, sps)

    # copy network to GPU    
    net.to(device)

    # initialise IR for channel estimation
    h_est = np.zeros([2,M_est])  #h_est = np.zeros([2,len(h_channel)]) 
    h_est[0,M_est//2] = 1 #h_est[0,len(h_channel)//2] = 1
    h_est = torch.tensor(h_est,requires_grad=True, dtype=torch.float32, device=device)

    # initialise optimizer
    optimizer = optim.Adam(net.parameters(), lr=lr_optim, amsgrad=True)

    # add h_est as parameter
    optimizer.add_param_group({"params": h_est})

    minibatch = torch.empty(1,2,batch_len*sps, device=device, dtype=torch.float32)
    batch_valid = torch.empty(1,2,N_valid*sps, device=device, dtype=torch.float32)
    SER_valid = torch.empty(num_epochs//epe, device=device, dtype=torch.float32)

    for epoch in range(num_epochs):
            
        net.train()
        with torch.set_grad_enabled(True):
            rx_tensor, data_tensor = generate_data(N_train, M, constellation, SNR, h_channel, sps, device)

            for m in range(N_train//batch_len):
                optimizer.zero_grad()
                minibatch[0,:,:] = rx_tensor[:,m*batch_len*sps:(m+1)*batch_len*sps]
            
                # process batch
                minibatch_output = net(minibatch)

                loss = loss_function(minibatch_output.squeeze(), minibatch.squeeze(), h_est, device, amp_levels)
                loss.backward()
                optimizer.step()    
            
        if epoch % epe == 0:
            net.eval()
            with torch.set_grad_enabled(False):                            
                rx_tensor_valid, data_tensor_valid = generate_data(N_valid, M, constellation, SNR, h_channel, sps, device) 
                
                batch_valid[0,:,:] = rx_tensor_valid
                batch_output_valid = net(batch_valid)

                #SER_orig = SER_symb(rx_tensor_valid[:,M:],data_tensor_valid[:,:-M], sps, amp_levels, num_lev, device)
                
                shift = find_shift(batch_output_valid,data_tensor_valid, 21, amp_levels, num_lev, device)
                SER_valid[epoch//epe] = SER_q(batch_output_valid[0,:,11+shift:-11], data_tensor_valid[:,11:-11-shift], sps, num_lev, device) 

                print(epoch, loss.item(), shift.item(), '\t\t\t\t\t\tSER = ',  SER_valid[epoch//epe].item())       #, '\t\t\t\t\t\t(unproc. SER = ',  SER_orig.item(),')'   )   #, '\n h = ', h_est.tolist())

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
    qq = batch_output_valid[0,:,:].cpu()
    E = torch.zeros(2,qq.shape[1], dtype=torch.float32) 
    amp_mat = amp_levels.cpu().repeat(qq.shape[1],2).transpose(0,1)
    temp = amp_mat * qq
    E[0,:], E[1,:] = torch.sum(temp[:num_lev,:], dim=0), torch.sum(temp[num_lev:,:], dim=0)
    create_constellation_plot(E)


def plot_correlation():
    qq = batch_output_valid[0,:,:].cpu()
    E = torch.zeros(2,qq.shape[1], dtype=torch.float32) 
    amp_mat = amp_levels.cpu().repeat(qq.shape[1],2).transpose(0,1)
    temp = amp_mat * qq
    E[0,:], E[1,:] = torch.sum(temp[:num_lev,:], dim=0), torch.sum(temp[num_lev:,:], dim=0)
    data = data_tensor_valid[1,:1000-1].detach().cpu().numpy()
    corr = np.correlate(E[1,1:1000].detach().cpu().numpy(),data,"same")
    print(np.argmax(np.abs(corr)))
    plt.plot(corr)
    plt.show()



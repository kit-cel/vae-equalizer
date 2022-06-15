################################################################################################
# Author:        Vincent Lauinger, 
# Affiliation:   Communications Engineering Lab (CEL), Karlsruhe Institute of Technology (KIT)
# Contact:       vincent.lauinger@kit.edu
# Last revision: 15th of June 2022
################################################################################################

import numpy as np
import torch
    
import matplotlib.pyplot as plt

import shared_funcs as sfun


def processing(mod, sps, SNR, nu, M_est, theta_diff,theta,lr_optim, batch_len, N_train_max, num_frames, flex_step, channel, symb_rate,tau_cd,tau_pmd,phiIQ, N_lrhalf):
    device = 'cpu' if torch.cuda.is_available() else 'cpu'
    print("We are using the following device for learning:",device)

    h_est, h_channel, P, amp_levels, amps, pol, nu_sc, var, pow_mean = sfun.init(channel, mod, device, nu, sps, M_est, SNR)

    SER_valid = torch.empty(4,num_frames, device=device, dtype=torch.float32)
    Var_est = torch.zeros(pol,num_frames, device=device, dtype=torch.float32)

    R = 1 #pow_mean #1
    N_cut = 10  # number of symbols cut off to prevent edge effects of convolution

    with torch.set_grad_enabled(False):  
        for frame in range(num_frames):

            if (frame%N_lrhalf == 0 and frame!=0):
                lr_optim *= 0.5     # learning rate scheduler

            rx_tensor, data_tensor, sigma_n = sfun.generate_data_shaping(N_train_max, amps, SNR, h_channel, P, pol,symb_rate,sps,tau_cd,tau_pmd,phiIQ,theta, device)
            out_const, h_est, e = sfun.CMA(rx_tensor, R, h_est, lr_optim, sps, True) 
            #loss = torch.mean(torch.abs(e)) 
            theta += theta_diff     # update theta per frame

            out_const = sfun.CPE(out_const[:,:,N_cut:-N_cut])        # carrier phase estimation # cut off edge symbols to avoid edge effects
            data_tensor = data_tensor[:,:,N_cut:-N_cut]
            shift,r = sfun.find_shift_symb_full(out_const,data_tensor, 21)      # find correlation within 21 symbols
            out_const = out_const.roll(r,0)      # compensate pol. shift
            out_const[0,:,:], out_const[1,:,:] = out_const[0,:,:].roll(int(-shift[0]),-1), out_const[1,:,:].roll(int(-shift[1]),-1)
            SER_valid[:2,frame] = sfun.SER_constell_shaping(out_const[:,:,11:-11-torch.max(torch.abs(shift))], data_tensor[:,:,11:-11-torch.max(torch.abs(shift))], amp_levels, nu_sc, var) 
            print(frame,'\t\ttraining: loss = ', torch.sum(e).item(), '\tshift_x = ',  shift[0].item(), '\tshift_y = ',  shift[1].item(), '\tr = ',  r  )    # compensate time shift (in multiple symb.)
            print('\t\t\t\t\t\t\tSER_x = ',  SER_valid[0,frame].item(), '\tSER_y = ',  SER_valid[1,frame].item(), '\t(constell. with shaping)' ) 
      
            out_train = sfun.soft_dec(out_const,var, amp_levels, nu_sc)     # soft demapper
            shift,r = sfun.find_shift(out_train,data_tensor, 21, amp_levels, pol)       # find correlation within 21 symbols
            out_train = out_train.roll(r,0)      # compensate pol. shift
            out_train[0,:,:], out_train[1,:,:] = out_train[0,:,:].roll(int(-shift[0]),-1), out_train[1,:,:].roll(int(-shift[1]),-1)     # compensate time shift (in multiple symb.)
            SER_valid[2:,frame] = sfun.SER_IQflip(out_train[:,:,11:-11-torch.max(torch.abs(shift))], data_tensor[:,:,11:-11-torch.max(torch.abs(shift))]) 
                
            print('\t\t\t\t\t\t\tSER_x = ',  SER_valid[2,frame].item(), '\tSER_y = ',  SER_valid[3,frame].item(), '\t(soft demapper)' )    

    return SER_valid, Var_est, var


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





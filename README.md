# Variational Autoencoder (VAE) based equalizer

This repo contains implementations of VAE based equlizers for an AWGN channel and an linear optical dual-pol. (DP) channel, including probabilitic constellation shaping (PCS).
The main scripts are the "Eval_run_xxx.py", which also allow sweeps of selected parameters. For the AWGN channel, there are eval scripts for each of the implemented equalizers. For the optical DP channel, there is one eval script with an additional parameter, which selects the equalizer; hence, some parameters are not relevant for certain equalizers.

Feel free to run, share, play with and modify this code! However, it would be nice if you always include a reference to this work---especially, if you use parts of this code for work resulting in a publication. 

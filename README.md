# Variational Autoencoder (VAE) based Equalizer

This repository contains implementations of VAE-based equlizers for an AWGN channel and an linear optical dual-polarization (DP) channel, including probabilitic constellation shaping (PCS).

The code was used to generate the results in the paper

V. Lauinger, F. Buchali and L. Schmalen, "Blind equalization and channel estimation in coherent optical communications using variational autoencoders," _IEEE J. Sel. Areas Commun._, Jul. 2022, accepted for publication, preprint available at https://arxiv.org/abs/2204.11776

The main scripts are the "Eval_run_xxx.py", which also allow sweeps of selected parameters. For the AWGN channel, there are eval scripts for each of the implemented equalizers. For the optical DP channel, there is one eval script with an additional parameter, which selects the equalizer; hence, some parameters are not relevant for certain equalizers.

Feel free to run, share, play with and modify this code! If you use parts of this code for work resulting in a publication, please always include a reference to this work and the accompanying paper.

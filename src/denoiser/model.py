import julius
import torch
from copy import copy
import torch.nn as nn
from typing import Tuple
import speechbrain as sb
from speechbrain.nnet.complex_networks.c_CNN import CConv2d
from speechbrain.nnet.complex_networks.c_normalization import CBatchNorm


class FC2N2D(nn.Module):
    """This function implements a Fully Complex Convolutional Network on STFT.
    Based on the "Deep Complex Networks", Trabelsi C. et al.
    Arguments
    ---------
    num_channels : int
        Number of input channels.
    rep_channels : int
        Number of channels in the intermediate channels.
    kernel_size: tuple
        Kernel size of the convolutional filters.
    compute_stft : object
        Function to compute the STFT representations.
    resynth : object
        Method to compute the inverse STFT and reconstruct the audio.
    resampling : bool
        Use or not a resampling method to double the sampling rate.
    """
    def __init__(
        self,
        rep_channels:int=64,
        kernel_size:Tuple[int, int]=(5, 5),
        compute_stft:object=None,
        resynth:object=None,
        resampling:bool=True
    ):
        super().__init__()
        self.rep_channels=rep_channels
        self.kernel_size=kernel_size
        self.compute_stft=compute_stft
        self.resynth=resynth
        self.resampling=resampling

        self.nnet_layers = sb.nnet.containers.Sequential(
            self.conv_block(1, self.rep_channels),
            self.conv_block(self.rep_channels, self.rep_channels),
            self.conv_block(self.rep_channels, self.rep_channels),
            self.conv_block(self.rep_channels, self.rep_channels),
            CConv2d(
                input_shape=(None, None, None, 2*self.rep_channels),
                out_channels=1,
                kernel_size=self.kernel_size
            ),
            nn.Tanh()
        )

    def forward(self, noisy_wavs):
        length = noisy_wavs.shape[1]
        x = copy(noisy_wavs)
        
        # Resample to avoid aliasing artifacts
        if self.resampling:
            x = self.resample(x, 1, 2)

        # Extract features
        noisy_spec = self.compute_features(x)

        # Mask prediction and mono-channel
        mask = self.nnet_layers(noisy_spec)
        predict_spec = torch.mul(mask, noisy_spec)
        predict_spec = predict_spec.unsqueeze(-1)

        # resynth the time-frequency representation
        x_hat = self.resynth(predict_spec, sig_length=length)

        # Return the sampled back audio
        if self.resampling:
            x = self.resample(x, 2, 1)

        return x_hat

    """
    Extract spectrogram and manipulate the waveform
    """
    def compute_features(self, x):
        feats = self.compute_stft(x)
        feats = feats.transpose(3, 4)

        # Separate real and imaginary parts from the STFT
        real, img = feats[..., 0], feats[..., 1]

        return torch.cat([real, img], dim=-1)
    
    def resample(self, x, from_sample, to_sample):
        x = x.transpose(1, 2) # B, L, C => B, C, L
        x = julius.resample_frac(x, from_sample, to_sample)

        return x.transpose(1, 2) # B, C, L => B, L, C
    
    def conv_block(self, in_channels, base_channels):
        return sb.nnet.containers.Sequential(
            CConv2d(
                input_shape=(None, None, None, 2*in_channels),
                out_channels=base_channels,
                kernel_size=self.kernel_size,
                padding="same"
            ),
            CBatchNorm(
                input_shape=(None, None, None, 2*base_channels)
            ),
            nn.ReLU()
        )
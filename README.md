# Training My Own HiFi-GAN

[Paper link](https://arxiv.org/pdf/2010.05646.pdf)

In this repo, I trained my own HiFi-GAN. It takes mel spectrograms of audio and generates audio from it. It works really well and the input and generated output are very similar. It might fool a person who does not know that person speaking is not real. 

This is my first foray deep into audio, so these are my notes as I learn them. Most of which might be from the paper. 

### Neural-Speech Synthesis Models
Most neural speech synthesis models use a two-stage pipeline: 
1.  Predicting a low resolution intermediate representation such as mel-spectrograms or linguistic features. Basically, this step encodes text into low-level representation of speech. If you don't know what mel-spectrograms are, check out this [link](https://medium.com/analytics-vidhya/understanding-the-mel-spectrogram-fca2afa2ce53). 
2.  Synthesizing raw waveform audio from the intermediate representation. This generates audio from the representation. 

Speech audio consists of sinusoidal signals with various periods. This periodic pattern is very important to generate realistic speech audio. HiFi GAN has a discriminator with small sub-discriminators that process only a specific part of a raw waveform. 

## HiFI-GAN 
It has 1 generator, and 2 discriminators. The discriminators are multi-scale and multi-period.

### Generator
The generator is a CNN. It's input is a mel-spectrogram and it upsamples it through transposed convolutions until the length of the outpu tsequence matches the temporal resolution of raw waveforms. Each TranposedConv is followed by a multi-receptive field fusion (MRF) module. 

**Multi-Receptive Field Fusion:** The MRF observes patterns of various lengths in parallel. It contains multiple residual blocks, and it outputs the sum of the outputs from each of the blocks. The kernel sizes and dilation rates are all selected to be as diverse as possible to recognize as many patterns as possible. The MRF also has a lot of hyperparameters like the hidden dimension $h_u$, kernel size $k_u$ of the TransposedConv, kernel sizes $k_r$ and dilation rates $D_r$ of MRF modules. 

### Discriminator 
Realistic speech audio depends on identifying long-term dependencies. A very short phoneme that is spoken for about 100 ms has about 1600 samples (with sample rate of 16000) or 2200 samples (sample rate of 22050). This means that there is a strong correlation between ~2200 samples in the audio data. This can be solved by increasing the receptive fields of the generator and the discriminator. 

The problem that we do not have a solution for (before this paper was published) was that all the various priodic signals of an input audio could not be utilized to generate good quality audio. The paper suggests the use of **multi-period discriminator** (**MPD**). An MPD has multiple sub-discriminators each handling a portion of periodic signals of input audio. 

**Multi-Period Discriminator:** Each of the sub-discriminators accept equally spaced samples of an input audio. The space is given by period $p$. The sub-discriminators are designed in such a way to allow them to capture different implicit structures from each other by looking at different parts of an input audio. The periods in the paper were set to $[2, 3, 5, 7, 11]$ to avoid overlaps as much as possible. 

A 1D raw audio of length $T$ is reshaped into a 2D data of height $T/p$ and width $p$. Then a 2D convolution is applied to the reshaped data. The kernel size of every CNN in MPD is $(k \times 1)$ to make it access each periodic sample independently. Each sub-discriminator is a stack of strided ConvNet with Leaky-ReLU. Weight normalization is also applied to MPD. 

**Multi-Scale Discriminator:** Each sub-discriminator in MPD only accepts disjoint samples, so the MSD consecutively evaluates the audio sequence. MSD has 3 sub-discriminators that each operate on different input scales: raw audio, $\times 2$ average-pooled audio, and $\times 4$ average-pooled audio. Similar to the MPD, each of the sub-discriminators in the MSD is a stack of strided and grouped ConvNet with leaky ReLU. The size of the overall network is increased by reducing stride and adding more layers. Weight normalization is applied to each sub-discriminator except for the first one (because it operates on raw audio). Thus, spectral normalization is applied to stablize training. 

* MPD operates on disjoint samples of raw waveforms.
* MSD operates on smoothed waveforms. 

### Training Losses 
**GAN Loss** For the generator and discriminator, the training objectives follow the LS-GAN, which replaced the binary cross-entropy loss of the origin GAN with *least squares loss* functions for non-vanishing gradient flows. The discriminator is trained to classify ground truth samples to 1, and synthesized samples to 0. The generator is trained to trick the discriminator by updating the sample quality to be classified to a value amost equal to 1. The losses for the generator $G$ and discriminator $D$ are defined as: 
$$L_{Adb}(D;G) = \mathbb{E}_{(x,s)}[(D(x) - 1)^2 + (D(G(s)))^2]$$ 
$$L_{Adv}(G;D) = \mathbb{E}_s[(D(G(s)) - 1)^2]$$
where $x$ is the ground truth audio and $s$ is the input condition (mel-spectrogram of the ground truth audio). 

**Mel-Spectrogram Loss** This loss serves to improve the training efficiency of the generator and the fidelity of the generated audio. Previous work as indicated that applying a reconstruction loss to the GAN model helps to generate realistic results. The mel-spectrogram loss can be expected to have the effect of focusing more on improving the perceptual quality due to the characteristics of the human auditory system. The mel-spectrogram loss is the L1 distance between the mel-spectrogram of a waveform synthesized by the generator and that of a ground truth waveform. It is given by: 
$$L_{Mel}(G) = \mathbb{E}_{(x,s)}[||\phi(x) - \phi(G(s))||_1]$$
where $\phi$ is the function that transforms the waveform to the corresponding mel-spectrogram. 
Note: L1 Loss is basically the absolute distance between each and every element on the output and target output. 

**Feature Matching Loss** It is a learned similarity metric measured by the difference between the features of the discriminator between a ground truth sample and a generated sample. Every intermediate feature of the discriminator is extracted, and the L1 distance between the ground truth sample and a conditionally generated sample in each feature space is calculated. It is defined as: 
$$L_{FM}(G;D) = \mathbb{E}_{(x,s)}[\sum_{i=1}^T {1\over{N_i}}||D^i(x) - D^i(G(s))||_1]$$
where $T$ denotes the number of layers in the discriminator; $D^i$ and $N_i$ denotes the features and the number of features in the $i$-th layer of the discriminator, respectively. 

**Final Loss** The final objectives for the generator and the discriminator are as follows: 
$$L_G = L_{Adv}(G;D) + \lambda_{fm}L_{FM}(G;D) + \lambda_{mel}L_{Mel}(G)$$ 
$$L_D = L_{Adv}(D;G)$$ 
where the paper sets $\lambda_{fm}=2$ and $\lambda_{mel} = 45$.

This concludes the architecture of the network. 
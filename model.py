import torch 
import torch.nn.functional as F 
import torch.nn as nn 
import torch.nn.utils as utils 

LRELU_SLOPE = 0.1 

def get_padding(kernel_size, dilation=1): 
  return int((kernel_size*dilation - dilation)/2) 

def init_weights(m, mean=0.0, std=0.01): 
  if isinstance(m, nn.Conv1d): 
    m.weight.data.normal_(mean, std)

class res_block1(nn.Module): 
  def __init__(self, h, channels, kernel_size=3, dilation=(1, 3, 5)): 
    super().__init__() 
    self.h = h
    self.convs1 = nn.ModuleList([
      utils.weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                                  padding=get_padding(kernel_size, dilation[0]))), 
      utils.weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                                  padding=get_padding(kernel_size, dilation[1]))), 
      utils.weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2], 
                                  padding=get_padding(kernel_size, dilation[2])))
    ])
    self.convs1.apply(init_weights)

    self.convs2 = nn.ModuleList([
      utils.weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=1, 
                                  padding=get_padding(kernel_size, 1))), 
      utils.weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=1, 
                                  padding=get_padding(kernel_size, 1))), 
      utils.weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=1, 
                                  padding=get_padding(kernel_size, 1)))
    ])
    self.convs2.apply(init_weights) 

  def forward(self, x): # lrelu -> cnn1 -> lrelu -> cnn2 -> residual x
    for c1, c2 in zip(self.convs1, self.convs2): 
      xt = F.leaky_relu(x, LRELU_SLOPE) 
      xt = c1(xt) 
      xt = F.leaky_relu(xt, LRELU_SLOPE) 
      xt = c2(xt) 
      x = xt + x 
    return x 

  def remove_weight_norm(self): 
    for l in self.convs1: 
      utils.remove_weight_norm(l) 
    for l in self.convs2: 
      utils.remove_weight_norm(l) 

class res_block2(nn.Module): 
  def __init__(self, h, channels, kernel_size=3, dilation=(1, 3)): 
    super().__init__() 
    self.h = h 
    self.convs = nn.ModuleList([ 
      utils.weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0], 
                                  padding=get_padding(kernel_size, dilation[0]))), 
      utils.weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1], 
                                  padding=get_padding(kernel_size, dilation[1])))
    ])
    self.convs.apply(init_weights) 

  def forward(self, x): 
    for c in self.convs: 
      xt = F.leaky_relu(x, LRELU_SLOPE) 
      xt = c(xt) 
      x = xt + x 
    return x 
  
  def remove_weight_norm(self): 
    for l in self.convs: 
      utils.remove_weight_norm(l) 

class generator(nn.Module): 
  def __init__(self, h): 
    super().__init__()  
    self.h = h 
    self.num_kernels = len(h.resblock_kernel_sizes) 
    self.num_upsamples = len(h.upsample_rates) 
    self.conv_pre = utils.weight_norm(nn.Conv1d(80, h.upsample_initial_channel, 7, 1, padding=3))
    resblock = res_block1 if h.resblock == '1' else res_block2 

    self.ups = nn.ModuleList() 
    for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)): 
      self.ups.append(utils.weight_norm(
        nn.ConvTranspose1d(h.upsample_initial_channel//(2**i), h.upsample_initial_channel//(2**(i+1)), 
                           k, u, padding=(k-u)//2)))
    self.resblocks = nn.ModuleList() 
    for i in range(len(self.ups)): 
      ch = h.upsample_initial_channel//(2**(i+1))
      for j, (k, d) in enumerate(zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)): 
        self.resblocks.append(resblock(h, ch, k, d))

    self.conv_post = utils.weight_norm(nn.Conv1d(ch, 1, 7, 1, padding=3))
    self.ups.apply(init_weights) 
    self.conv_post.apply(init_weights) 

  def forward(self, x): 
    x = self.conv_pre(x) # This is the first layer that upsamples the number of channels from 80 to 8192
    for i in range(self.num_upsamples): # Stacks the transpose-conv + resblocks 'num_upsamples' times.
      x = F.leaky_relu(x, LRELU_SLOPE) 
      x = self.ups[i](x) # Decreases the num of channels
      xs = None 
      for j in range(self.num_kernels): # Each iteration inputs into the resblocks
        if xs is None: 
          xs = self.resblocks[i*self.num_kernels+j](x) 
        else: 
          xs += self.resblocks[i*self.num_kernels+j](x) 
      x = xs / self.num_kernels # In the end, all the individual outputs from the resblocks is meaned. 
      # After all the resblocks, the final output is the dim of 32 in the current configuration.
    x = F.leaky_relu(x) 
    x = self.conv_post(x) # Takes the 32 input channels and gives 1 channel of output
    x = torch.tanh(x) 
    return x # Final output is (bs, 1, 2097152) for default config. 
  
  def remove_weight_norm(self): 
    print('Removing weight norm...')
    for l in self.ups: 
      utils.remove_weight_norm(l) 
    for l in self.resblocks: 
      l.remove_weight_norm() 
    utils.remove_weight_norm(self.conv_pre) 
    utils.remove_weight_norm(self.conv_post) 

class discriminator_p(nn.Module): 
  def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False): 
    super().__init__() 
    self.period = period 
    norm_f = utils.weight_norm if use_spectral_norm == False else utils.spectral_norm
    self.convs = nn.ModuleList([ 
      norm_f(nn.Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))), 
      norm_f(nn.Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))), 
      norm_f(nn.Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))), 
      norm_f(nn.Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))), 
      norm_f(nn.Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))), 
    ])
    self.conv_post = norm_f(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

  def forward(self, x): 
    fmap = list() 

    b, c, t = x.shape 
    if t % self.period != 0: 
      n_pad = self.period - (t % self.period) 
      x = F.pad(x, (0, n_pad), 'reflect')
      t = t + n_pad 
    x = x.view(b, c, t // self.period, self.period) 

    for l in self.convs: 
      x = l(x) 
      x = F.leaky_relu(x, LRELU_SLOPE) 
      fmap.append(x) 
    x = self.conv_post(x) 
    fmap.append(x) 
    x = torch.flatten(x, 1, -1)

    return x, fmap 

class multi_period_discriminator(nn.Module): 
  def __init__(self): 
    super().__init__() 
    self.discriminators = nn.ModuleList([
      discriminator_p(i) for i in [2, 3, 5, 7, 11]
    ])
  
  def forward(self, y, y_hat): # Takes actual out (y) and fake out (y_hat)
    y_d_rs, y_d_gs, fmap_rs, fmap_gs = list(), list(), list(), list()
    for i, d in enumerate(self.discriminators): # each discriminator has a different kernel size (but 1 depth) to compute only 1 period of audio. 
      y_d_r, fmap_r = d(y) # calculates discrimination score for real (hence, 'r'). Look, I didn't pick the variables names okay. 
      y_d_g, fmap_g = d(y_hat) # 'g' stands for generated
      y_d_rs.append(y_d_r) 
      fmap_rs.append(fmap_r) 
      y_d_gs.append(y_d_g) 
      fmap_gs.append(fmap_g) 

    return y_d_rs, y_d_gs, fmap_rs, fmap_gs 

class discriminator_s(nn.Module): 
  def __init__(self, use_spectral_norm=False): 
    super().__init__() 
    norm_f = utils.weight_norm if use_spectral_norm == False else utils.spectral_norm 
    self.convs = nn.ModuleList([
      norm_f(nn.Conv1d(1, 128, 15, 1, padding=7)), 
      norm_f(nn.Conv1d(128, 128, 41, 2, groups=4, padding=20)), 
      norm_f(nn.Conv1d(128, 256, 41, 2, groups=16, padding=20)), 
      norm_f(nn.Conv1d(256, 512, 41, 4, groups=16, padding=20)), 
      norm_f(nn.Conv1d(512, 1024, 41, 4, groups=16, padding=20)), 
      norm_f(nn.Conv1d(1024, 1024, 41, 1, groups=16, padding=20)), 
      norm_f(nn.Conv1d(1024, 1024, 5, 1, padding=2)), 
    ])
    self.conv_post = norm_f(nn.Conv1d(1024, 1, 3, 1, padding=1)) 

  def forward(self, x): 
    fmap = list() 
    for l in self.convs: 
      x = l(x) 
      x = F.leaky_relu(x, LRELU_SLOPE) 
      fmap.append(x) 
    x = self.conv_post(x) 
    fmap.append(x) 
    x = torch.flatten(x, 1, -1)
    return x, fmap 
  
class multi_scale_discriminator(nn.Module): 
  def __init__(self): 
    super().__init__() 
    self.discriminators = nn.ModuleList([
      discriminator_s(use_spectral_norm=True), 
      discriminator_s(), 
      discriminator_s(),
    ])
    self.meanpools = nn.ModuleList([
      nn.AvgPool1d(4, 2, padding=2), 
      nn.AvgPool1d(4, 2, padding=2)
    ])

  def forward(self, y, y_hat): # in MSD, you do not reshape the input data to differentiate between different period of the input audio. 
    y_d_rs, y_d_gs, fmap_rs, fmap_gs = list(), list(), list(), list()
    for i, d in enumerate(self.discriminators): 
      if i != 0: # you do not average-pool the raw audio. Also, you use spectral_norm on the raw audio. 
        y = self.meanpools[i-1](y) # average-pooling the inputs
        y_hat = self.meanpools[i-1](y_hat)
      y_d_r, fmap_r = d(y) # discrimination scores for the inputs
      y_d_g, fmap_g = d(y_hat)
      y_d_rs.append(y_d_r) 
      fmap_rs.append(fmap_r) # fmap are the feature maps. It's audio
      y_d_gs.append(y_d_g) 
      fmap_gs.append(fmap_g) 
    return y_d_rs, y_d_gs, fmap_rs, fmap_gs 
  
def feature_loss(fmap_r, fmap_g): # it is the mean absolute error a.k.a L1 Loss
  loss = 0 # all the losses calculated is added to the total loss. 
  for dr, dg in zip(fmap_r, fmap_g): 
    for rl, gl in zip(dr, dg): 
      loss += torch.mean(torch.abs(rl - gl))
  return loss*2 # 2 is just a factor added to increase the influence of this loss to the overall loss

def discriminator_loss(disc_real_outputs, disc_generated_outputs): 
  loss, r_losses, g_losses = 0, list(), list() 
  for dr, dg in zip(disc_real_outputs, disc_generated_outputs): 
    r_loss = torch.mean((1-dr)**2) # real loss 
    g_loss = torch.mean(dg**2) # gen loss 
    loss += (r_loss + g_loss) # GAN Loss 
    r_losses.append(r_loss.item()) 
    g_losses.append(g_loss.item()) 
  return loss, r_losses, g_losses 

def generator_loss(disc_outputs): 
  loss, gen_losses = 0, list() 
  for dg in disc_outputs: 
    l = torch.mean((1-dg)**2) # GAN Loss for generators
    gen_losses.append(l) 
    loss += l 
  return loss, gen_losses 

if __name__ == '__main__': 
  model = multi_period_discriminator()
  [print(model.discriminators[i].period) for i in range(5)] 
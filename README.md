# pyperlin
GPU accelerated Perlin Noise in python

Limitation: resolutions needs to divide shape

Using pytoch as the array backend makes the GPU acceleration trivial. Also it uses batch sampling to better use parallelization.

By playing with the parameters of Perlin noise, it is possible to create different textures. (code at the bottom)
![alt text](pyperlin/examples/clouds_and_fire.png?raw=true)

## Usage
```python
import torch
from pyperlin import FractalPerlin2D

shape = (32,256,256) #for batch size = 32 and noises' shape = (256,256)
resolutions = [(2**i,2**i) for i in range(1,7)] #for lacunarity = 2.0
factors = [.5**i for i in range(8)] #for persistence = 0.5
g_cuda = torch.Generator(device='cuda') #for GPU acceleration
fp = FractalPerlin2D(shape, resolutions, factors, generator=g_cuda)
noise = fp() #sampling
```

## Benchmarks
CPU: Intel(R) Core(TM) i7-6850K CPU @ 3.60GHz

GPU: Nvidia Titan XP
| Algorithms | (1,256,256);1octaves | (1,256,256);6octaves | (32,256,256);6octaves | (32,1024,1024);8octaves |
| --- | --- | --- | --- | --- |
| noise.pnoise2  | 73.8 ms | 86.3 ms | 2.85 s | 48 s |
| pyperlin.FractalPerlin2D (cpu)  | 1.15 ms | 11.5 ms | 377 ms | 8.49 s |
| pyperlin.FractalPerlin2D (gpu)  | 481 µs | 2.84 ms | 16.8 ms | 121 ms |


## More Examples
```python
import torch
from pyperlin import FractalPerlin2D
import matplotlib.pyplot as plt

shape = (1,1024,1024) #for batch size = 1 and noises' shape = (1024,1024)
factors = [.5**i for i in range(8)] #for persistence = 0.5
g_cuda = torch.Generator(device='cuda') #for GPU acceleration

clouds_resolutions = [(2**i,2**i) for i in range(1,7)] #for lacunarity = 2.0
clouds = FractalPerlin2D(shape, clouds_resolutions, factors, generator=g_cuda)().cpu().numpy()[0]

fire_resolutions = [(2**i,4**i) for i in range(1,4)] #for lacunarity = 2.0 and 4.0
fire = FractalPerlin2D(shape, fire_resolutions, factors, generator=g_cuda)().cpu().numpy()[0]

fig = plt.figure(figsize=(10,5))

ax1 = fig.add_subplot(121)
ax1.set_axis_off()
ax1.set_title('Clouds')
ax1.imshow(clouds, vmax=1.2, cmap=plt.get_cmap('Blues'))

ax2 = fig.add_subplot(122)
ax2.set_axis_off()
ax2.set_title('Fire')
ax2.imshow(fire, vmax=.3, cmap=plt.get_cmap('YlOrBr'))

fig.show()
```
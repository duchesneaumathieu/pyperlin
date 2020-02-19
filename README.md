# pyperlin
GPU accelerated Perlin Noise in python

Using pytoch as the array backend makes the GPU acceleration trivial. Also it uses batch sampling to better use parallelization.

E.g.
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

Limitation: resolutions needs to divide shape

## Benchmarks
CPU: Intel(R) Core(TM) i7-6850K CPU @ 3.60GHz

GPU: Nvidia Titan XP
| Algorithms | (1,256,256);1octaves | (1,256,256);6octaves | (32,256,256);6octaves | (32,1024,1024);8octaves |
| --- | --- | --- | --- | --- |
| noise.pnoise2  | 73.8 ms | 86.3 ms | 2.85 s | 48 s |
| pyperlin.FractalPerlin2D (cpu)  | 1.15 ms | 11.5 ms | 377 ms | 8.49 s |
| pyperlin.FractalPerlin2D (gpu)  | 481 µs | 2.84 ms | 16.8 ms | 121 ms |

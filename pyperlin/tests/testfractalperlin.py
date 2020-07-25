#unittest
import unittest, torch
from pyperlin import FractalPerlin2D

class TestFractalPerlin2D(unittest.TestCase):
    def test_fractal_perlin_2d(self):
        shape = (32,256,256)
        resolutions = [(2**i,2**i) for i in range(1,7)] #for lacunarity = 2.0
        factors = [.5**i for i in range(8)]
        g_cuda = torch.Generator(device='cpu')
        fp = FractalPerlin2D(shape, resolutions, factors, generator=g_cuda)
        noise = fp().numpy()
        self.assertEqual(noise.shape, shape)

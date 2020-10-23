#unittest
import unittest, torch
from pyperlin import FractalPerlin2D

class TestFractalPerlin2D(unittest.TestCase):
    def torch_generator(self, seed):
        return torch.Generator().manual_seed(seed)
    
    def test_shape(self):
        shape = (32,256,256)
        resolutions = [(2**i,2**i) for i in range(1,7)] #for lacunarity = 2.0
        factors = [.5**i for i in range(6)]
        noise = FractalPerlin2D(shape, resolutions, factors)().numpy()
        self.assertEqual(noise.shape, shape)
        
    def test_different_batch_elements(self):
        shape = (32,256,256)
        resolutions = [(2**i,2**i) for i in range(1,7)] #for lacunarity = 2.0
        factors = [.5**i for i in range(6)]
        noise = FractalPerlin2D(shape, resolutions, factors)()
        self.assertFalse(torch.allclose(noise[0], noise[1]))
    
    def test_replicability(self):
        shape = (32,256,256)
        resolutions = [(2**i,2**i) for i in range(1,7)] #for lacunarity = 2.0
        factors = [.5**i for i in range(6)]
        cafe_gen = self.torch_generator(0xcafe)
        beef_gen = self.torch_generator(0xbeef)
        cafe_fp = FractalPerlin2D(shape, resolutions, factors, generator=cafe_gen)
        beef_fp = FractalPerlin2D(shape, resolutions, factors, generator=beef_gen)
        cafe_noise1 = cafe_fp()
        cafe_noise2 = cafe_fp()
        self.assertFalse(torch.allclose(cafe_noise1, beef_fp()))
        self.assertFalse(torch.allclose(cafe_noise1, cafe_fp()))
        #reset cafe_gen
        cafe_gen = self.torch_generator(0xcafe)
        cafe_fp = FractalPerlin2D(shape, resolutions, factors, generator=cafe_gen)
        self.assertTrue(torch.allclose(cafe_noise1, cafe_fp()))
        
    def test_dynamic_batch_size(self):
        static_shape = (32,256,256)
        dynamic_shape = (256,256)
        resolutions = [(2**i,2**i) for i in range(1,7)] #for lacunarity = 2.0
        factors = [.5**i for i in range(6)]
        gen = self.torch_generator(0xcafe)
        fp = FractalPerlin2D(static_shape, resolutions, factors, generator=gen)
        static_noise = fp()
        
        #update batch size when using static shape (16 instead of 32)
        noise = fp(batch_size=16)
        self.assertEqual(len(noise), 16)
        
        #only using dynamic batch size
        gen.manual_seed(0xcafe) #reset gen
        fp = FractalPerlin2D(dynamic_shape, resolutions, factors, generator=gen)
        dynamic_noise = fp(32)
        self.assertTrue(torch.allclose(static_noise, dynamic_noise))
import torch
tau = 6.28318530718

class FractalPerlin2D(object):
    def __init__(self, shape, resolutions, factors, generator=torch.random.default_generator):
        shape = shape if len(shape)==3 else (None,)+shape
        self.shape = shape
        self.factors = factors
        self.generator = generator
        self.device = generator.device
        self.resolutions = resolutions
        self.grid_shapes = [(shape[1]//res[0], shape[2]//res[1]) for res in resolutions]
        
        #precomputed tensors
        self.linxs = [torch.linspace(0,1,gs[1],device=self.device) for gs in self.grid_shapes]
        self.linys = [torch.linspace(0,1,gs[0],device=self.device) for gs in self.grid_shapes]
        self.tl_masks = [self.fade(lx)[None,:]*self.fade(ly)[:,None] for lx, ly in zip(self.linxs, self.linys)]
        self.tr_masks = [torch.flip(tl_mask,dims=[1]) for tl_mask in self.tl_masks]
        self.bl_masks = [torch.flip(tl_mask,dims=[0]) for tl_mask in self.tl_masks]
        self.br_masks = [torch.flip(tl_mask,dims=[0,1]) for tl_mask in self.tl_masks]
        
    def fade(self, t):
        return 6 * t**5 - 15 * t**4 + 10 * t**3
    
    def perlin_noise(self, octave, batch_size):
        res = self.resolutions[octave]
        angles = torch.zeros((batch_size, res[0]+2, res[1]+2), device=self.device)
        angles.uniform_(0, tau, generator=self.generator)
        rx = torch.cos(angles)[:,:,:,None]*self.linxs[octave]
        ry = torch.sin(angles)[:,:,:,None]*self.linys[octave]
        prx, pry = rx[:,:,:,None,:], ry[:,:,:,:,None]
        nrx, nry = -torch.flip(prx, dims=[4]), -torch.flip(pry, dims=[3])
        br = prx[:,:-1,:-1] + pry[:,:-1,:-1]
        bl = nrx[:,:-1,1:] + pry[:,:-1,1:]
        tr = prx[:,1:,:-1] + nry[:,1:,:-1]
        tl = nrx[:,1:,1:] + nry[:,1:,1:]
        
        grid_shape = self.grid_shapes[octave]
        grids = self.br_masks[octave]*br + self.bl_masks[octave]*bl + self.tr_masks[octave]*tr + self.tl_masks[octave]*tl
        noise = grids.permute(0,1,3,2,4).reshape((batch_size, self.shape[1]+grid_shape[0], self.shape[2]+grid_shape[1]))

        A = torch.randint(0,grid_shape[0],(batch_size,), device=self.device, generator=self.generator)
        B = torch.randint(0,grid_shape[1],(batch_size,), device=self.device, generator=self.generator)
        noise = torch.stack([noise[n,a:a-grid_shape[0], b:b-grid_shape[1]] for n,(a,b) in enumerate(zip(A,B))])
        return noise
    
    def __call__(self, batch_size=None):
        batch_size = self.shape[0] if batch_size is None else batch_size
        shape = (batch_size,) + self.shape[1:]
        noise = torch.zeros(shape, device=self.device)
        for octave, factor in enumerate(self.factors):
            noise += factor*self.perlin_noise(octave, batch_size=batch_size)
        return noise
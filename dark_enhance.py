import torch
import torch.nn.functional as F
class DarkEnhance(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def get_dark_channel(self, image:torch.Tensor):
        patchSize = 7
        padSize = 3
        jDark = image.min(-1).values
        img = F.pad(jDark, ((padSize, padSize, padSize, padSize)), mode='constant', value=255)
        jDark = -F.max_pool2d(-img,kernel_size=patchSize,stride=1)
        return jDark

    def estimate_atmospheric_light(self, image:torch.Tensor, jDark:torch.Tensor):
        height, width = jDark.shape[1],jDark.shape[2]
        numpx = max(width * height / 1000, 1)
        jDarkVec = jDark.flatten(1,2)
        imgVec = image.flatten(1,2)
        _,indices = torch.sort(jDarkVec)
        brightest_indices = indices[:,int(width * height - numpx):]
        atmSum = imgVec[torch.arange(imgVec.shape[0]).unsqueeze(-1),brightest_indices[:,:int(numpx)]].sum(1)
        A = atmSum / int(numpx)
        return A[:,None,None,:]

    def estimate_transmission(self, image:torch.Tensor, A:torch.Tensor):
        omega = 0.98
        img = image/A
        transmission = 1 - omega * self.get_dark_channel(img)
        return transmission

    def dehaze(self, image:torch.Tensor, A:torch.Tensor, transmission:torch.Tensor):
        t0 = torch.tensor([0.15]).to(image.device)
        J= A + (image - A)/torch.maximum(transmission, t0).unsqueeze(-1)
        return J

    def forward(self, HazeImg:torch.Tensor):
        HazeImg_ = 255.0-HazeImg
        dark_image = self.get_dark_channel(HazeImg_)
        A = self.estimate_atmospheric_light(HazeImg_, dark_image)
        transmission = self.estimate_transmission(HazeImg_, A)
        dehaze_image = self.dehaze(HazeImg_, A, transmission)
        dehaze_image = 1-dehaze_image / torch.amax(dehaze_image,(1,2,3),True)
        return (dehaze_image*255).clamp(0,255)
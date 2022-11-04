import torch
import torch.nn.functional as F


class DarkEnhance(torch.nn.Module):
    def __init__(self, patch_size: int = 7, pad_size: int = 3, omega: float = 0.98, t0: float = 0.15) -> None:
        super().__init__()
        self.omega = omega
        self.t0 = torch.tensor([t0])
        self.pad = torch.nn.ConstantPad2d(
            (pad_size, pad_size, pad_size, pad_size), 255.0)
        self.max_filter = torch.nn.MaxPool2d(kernel_size=patch_size, stride=1)

    def get_dark_channel(self, image: torch.Tensor):
        jDark = image.min(-1).values
        img = self.pad(jDark)
        jDark = -self.max_filter(-img)
        return jDark

    def estimate_atmospheric_light(self, image: torch.Tensor, jDark: torch.Tensor):
        height, width = jDark.shape[1], jDark.shape[2]
        numpx = max(width * height // 1000, 1)
        jDarkVec = jDark.flatten(1, 2)
        imgVec = image.flatten(1, 2)
        #_, indices = torch.sort(jDarkVec, stable=True)
        #A = imgVec[torch.arange(imgVec.shape[0]).unsqueeze(-1),indices[:,int(width * height - numpx):-1]].mean(1)
        # A = torch.gather(imgVec, 1, indices[:, int(
        #     width * height - numpx):-1, None]).mean(1)
        _, indices = torch.topk(jDarkVec, k=int(
            width * height - numpx), dim=1, largest=False)
        A = torch.gather(imgVec, 1, indices[:, :, None]).mean(1)
        return A[:, None, None, :]

    def estimate_transmission(self, image: torch.Tensor, A: torch.Tensor):
        transmission = 1 - self.omega * self.get_dark_channel(image/A)
        return transmission

    def dehaze(self, image: torch.Tensor, A: torch.Tensor, transmission: torch.Tensor):
        J = A + (image - A)/torch.maximum(transmission, self.t0.to(transmission.device)).unsqueeze(-1)
        return J
        

    def forward(self, HazeImg: torch.Tensor):
        HazeImg_ = 255.0-HazeImg
        dark_image = self.get_dark_channel(HazeImg_)
        A = self.estimate_atmospheric_light(HazeImg_, dark_image)
        transmission = self.estimate_transmission(HazeImg_, A)
        dehaze_image = self.dehaze(HazeImg_, A, transmission)
        dehaze_image = 1-dehaze_image / \
            torch.amax(dehaze_image, (1, 2, 3), True)
        return dehaze_image*255

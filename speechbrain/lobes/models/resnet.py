from typing import List, Type, Union
from torch import Tensor
import torch.nn as nn
from torchvision.models.resnet import BasicBlock, Bottleneck
from torchvision.models import ResNet
import speechbrain as sb


class CustomResNet(ResNet):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]] = BasicBlock,
        layers: List[int] = [2, 2, 2, 2],
        n_classes: int = 10,
        in_channels: int = 1,
        **kwargs,
    ) -> None:
        self.original_inplanes: int = 64
        super(CustomResNet, self).__init__(block, layers, n_classes, **kwargs)
        self.conv1 = nn.Conv2d(
            in_channels,
            self.original_inplanes,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.log_sm = sb.nnet.activations.Softmax(apply_log=True)

    def forward(self, x: Tensor) -> Tensor:
        out = self._forward_impl(x)
        out = out.unsqueeze(1)
        return self.log_sm(out)


if __name__ == "__main__":
    model = CustomResNet()
    print(model)

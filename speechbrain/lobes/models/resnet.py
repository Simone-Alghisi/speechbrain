from typing import List, Type, Union
from torch import Tensor
import torch.nn as nn
from torchvision.models.resnet import BasicBlock, Bottleneck
from torchvision.models import ResNet
import speechbrain as sb


class CustomResNet(ResNet):
    """Custom implementation of the ResNet architecture available in
    TorchVision.

    The first convolutional layer conv1 has been modified in order to
    specify the number of channels in input. Moroever, the output of
    the network has been changed to its corresponding softmax application.

    Arguments
    ---------
    block: torchvision.models.resnet.BasicBlock or torchvision.models.resnet.Bottleneck
        Type of block used to make layers.
    layers: list of int
        Number of blocks for layer1, layer2, layer3, and layer4.
    in_classes: int
        Number of output classes.
    in_channels: int
        Number of input channels.
    **kwargs: any
        Additional parameters for TorchVision ResNet.


    Example
    -------
    >>> from speechbrain.lobes.models.resent import CustomResNet
    >>> import torch
    >>> model = CustomResNet()
    >>> x = torch.randn([64, 1, 40, 50])
    >>> y = model(x)
    >>> y.shape
    torch.Size([64, 1, 10])
    """

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
        """Computes the forward call. Firstly, it passes the input tensor
        x to TorchVision's _forward_impl, and then applies speechbrain's
        softmax to obtain the logits associated to the output. An unsqueeze
        is performed for compatibility reasons.

        Arguments
        ---------
        x : Tensor
            The input tensor with shape NxCxTxF, where N is the batch size,
            C is the number of channels, while T and F are the two remaining
            dimensions, e.g. time and features.

        Returns
        -------
        predictions : Tensor
            Tensor of probabilities with shape Nx1xO, where N is the batch size,
            while O is the number of output classes
        """
        out = self._forward_impl(x)
        out = out.unsqueeze(1)
        return self.log_sm(out)


if __name__ == "__main__":
    model = CustomResNet()
    print(model)

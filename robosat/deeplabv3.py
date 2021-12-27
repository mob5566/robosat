from modeling.deeplab import DeepLab

from collections import OrderedDict

class DeepLabV3P(DeepLab):
    def __init__(self, num_classes):
        super().__init__(num_classes=num_classes)

    def forward(self, x):
        result = OrderedDict()
        result["out"] = super().forward(x)

        return result

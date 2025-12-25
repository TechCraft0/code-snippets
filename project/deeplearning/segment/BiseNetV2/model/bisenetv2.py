import torch
from .common import *

class BisenetV2(nn.Module):
    def __init__(self, in_channels, out_channels, n_classes, mode: str="train"):
        super(BisenetV2, self).__init__()

        self.detail_branch = DetailEnhance()
        self.segment_branch = SegmentBranch(in_channels, out_channels)

        self.bga_block = BilateralGuidedAggregation(128)

        self.main_head = SegmentHead(128, 1024, n_classes)

        self.aux_head1 = SegmentHead(16, 128, n_classes)
        self.aux_head2 = SegmentHead(32, 128, n_classes)
        self.aux_head3 = SegmentHead(64, 128, n_classes)
        self.aux_head4 = SegmentHead(128, 128, n_classes)

        self.mode = mode

        self.init_weights()


    def forward(self, x):
        img_size = x.size()[2:]
        x1 = self.detail_branch(x)
        d1, d2, d3, d4, d5 = self.segment_branch(x)
        bga = self.bga_block(x1, d5)

        main_out = self.main_head(bga, img_size)

        if self.mode == 'train':

            aux_out1 = self.aux_head1(d1, img_size)
            aux_out2 = self.aux_head2(d2, img_size)
            aux_out3 = self.aux_head3(d3, img_size)
            aux_out4 = self.aux_head4(d4, img_size)

            return main_out, (aux_out1, aux_out2, aux_out3, aux_out4)
        elif self.mode in ['val', 'test']:
            return main_out
        elif self.mode == "export":
            pred = torch.argmax(main_out, dim=1, keepdim=True).int()  # shape: (1, 1, H, W)
            return pred
        else:
            raise ValueError(f"Unsupported model mode: {self.mode}")


    def init_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, mode='fan_out')
                if not module.bias is None: nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                if hasattr(module, 'last_bn') and module.last_bn:
                    nn.init.zeros_(module.weight)
                else:
                    nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

if __name__ == '__main__':
    data = torch.randn(1, 3, 1024, 512)
    model = BisenetV2(3, 16, 5)
    model.eval()
    out, aux_out = model(data)
    print(out.shape, aux_out[0].shape, aux_out[1].shape, aux_out[2].shape, aux_out[3].shape)

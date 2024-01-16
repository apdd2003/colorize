import torch
import torch.nn as nn
import numpy as np
# from IPython import embed

from .base_color import *


class ECCVGenerator(BaseColor):
    def __init__(self, norm_layer=nn.BatchNorm2d):
        super(ECCVGenerator, self).__init__()

        model1 = [nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=True), ]
        model1 += [nn.ReLU(True), ]
        model1 += [nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=True), ]
        model1 += [nn.ReLU(True), ]
        model1 += [norm_layer(64), ]

        model2 = [nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True), ]
        model2 += [nn.ReLU(True), ]
        model2 += [nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=True), ]
        model2 += [nn.ReLU(True), ]
        model2 += [norm_layer(128), ]

        model3 = [nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True), ]
        model3 += [nn.ReLU(True), ]
        model3 += [nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True), ]
        model3 += [nn.ReLU(True), ]
        model3 += [nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=True), ]
        model3 += [nn.ReLU(True), ]
        model3 += [norm_layer(256), ]

        model4 = [nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True), ]
        model4 += [nn.ReLU(True), ]
        model4 += [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True), ]
        model4 += [nn.ReLU(True), ]
        model4 += [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True), ]
        model4 += [nn.ReLU(True), ]
        model4 += [norm_layer(512), ]

        model5 = [nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True), ]
        model5 += [nn.ReLU(True), ]
        model5 += [nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True), ]
        model5 += [nn.ReLU(True), ]
        model5 += [nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True), ]
        model5 += [nn.ReLU(True), ]
        model5 += [norm_layer(512), ]

        model6 = [nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True), ]
        model6 += [nn.ReLU(True), ]
        model6 += [nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True), ]
        model6 += [nn.ReLU(True), ]
        model6 += [nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True), ]
        model6 += [nn.ReLU(True), ]
        model6 += [norm_layer(512), ]

        model7 = [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True), ]
        model7 += [nn.ReLU(True), ]
        model7 += [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True), ]
        model7 += [nn.ReLU(True), ]
        model7 += [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True), ]
        model7 += [nn.ReLU(True), ]
        model7 += [norm_layer(512), ]

        model8 = [nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=True), ]
        model8 += [nn.ReLU(True), ]
        model8 += [nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True), ]
        model8 += [nn.ReLU(True), ]
        model8 += [nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True), ]
        model8 += [nn.ReLU(True), ]

        model8 += [nn.Conv2d(256, 313, kernel_size=1, stride=1, padding=0, bias=True), ]

        self.model1 = nn.Sequential(*model1)
        self.model2 = nn.Sequential(*model2)
        self.model3 = nn.Sequential(*model3)
        self.model4 = nn.Sequential(*model4)
        self.model5 = nn.Sequential(*model5)
        self.model6 = nn.Sequential(*model6)
        self.model7 = nn.Sequential(*model7)
        self.model8 = nn.Sequential(*model8)

        self.softmax = nn.Softmax(dim=1)
        self.model_out = nn.Conv2d(313, 2, kernel_size=1, padding=0, dilation=1, stride=1, bias=False)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear')

    def forward(self, input_l):
        conv1_2 = self.model1(self.normalize_l(input_l))
        conv2_2 = self.model2(conv1_2)
        conv3_3 = self.model3(conv2_2)
        conv4_3 = self.model4(conv3_3)
        conv5_3 = self.model5(conv4_3)
        conv6_3 = self.model6(conv5_3)
        conv7_3 = self.model7(conv6_3)
        conv8_3 = self.model8(conv7_3)
        out_reg = self.model_out(self.softmax(conv8_3))

        return self.unnormalize_ab(self.upsample4(out_reg))


def eccv16(pretrained=True):
    model = ECCVGenerator()
    if pretrained:
        import torch.utils.model_zoo as model_zoo
        # model.load_state_dict(
        model.load_state_dict(
                # model_zoo.load_url('https://colorizers.s3.us-east-2.amazonaws.com/colorization_release_v2-9b330a0b.pth',
                #                    map_location='cpu', check_hash=True)))

        model_zoo.load_url(
            'https://samsk.s3.us-east-1.amazonaws.com/colorization_release_v2-9b330a0b.pth?response-content-disposition=inline&X-Amz-Security-Token=IQoJb3JpZ2luX2VjEAIaCmFwLXNvdXRoLTEiRjBEAiBchwsTFeDmLVWG%2F36Dc%2FihRK%2FW8eoPijeUGu0651%2BU6wIgevQ4tqAFIsaCbfCkpRVrC66DakeRm%2B0kuHCoEcbfI4kqhAMIq%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARAAGgw5NzUwNTAwNDI2NzkiDCf8R4Ghki2B6m199irYAsCBz33kEAily3MjhZFtxtCvdyOriwI6nR5aBHrG1lufBq74suPY27sQX7GIf4FW01wKgfsqC2G3TZK4zVpuisB3%2FSCkP7Qv%2FZA%2BDmd%2F%2FXy9asLXhHO17UEIuU45mnDBk7m1hChVY1W27fJvGNTALpxa86NDMjhIpWdvO%2BGMCdIDaRM%2FOjsFE8p13lGMuxjgGBUceHJTOPXpV76MxWhwndgX3o3F%2BEvGe6VssDgVEq241VSUWlRmlq3uojTcW7ULQb9vTtVzRfrNG5%2FdgojxbM%2FQKpG24JOAKQpBKvlaYeZGo0oX%2Fr9%2FU6377SO%2F3fJkIx8nfNNDsQzCsWINHJgZsXJBATo92mE9%2BH%2FUzJxi9P3Ie7G%2FQU2YPukXne1vWagHuvM85ExU9oGYwRPX3LFf345by7T6fCTsUZ6Pc133f%2B1pVDkACmXN%2FcqX87%2FoUDSfFJZaqTNvIAgOMM6Gm60GOrQCWE09gL39NbkR1RilySSbCbz1NuC5pfz0K2SDGmqZclfTNVRuBV66i%2FAeWdHcMadLvfq4Or1CGO74gie8s51wliGx4UM9vE8xiGh%2Bvyb1tPocoe3%2BBmmpT0LiW2r32TioPAtjUwsSmTwnbFBkMHmfycWT78pTqY6Pzpb1icgms0UqGm6Sfl8mcrwi8AfJ0de71o3qCJE4L8Tpz4ZKE%2BH1UjdprqZt%2F2zZ1qzMXBE9w0cYDhyzsKVkOX05WvHdEAoraY4L%2F6mV7OoVGp6MRlP1XqZqqeKeDIPG3KqSbrmu4yziuSDwctOnCxbOu4YpTL9m%2BXJWb%2BqmOEG1m2iSbc%2B%2FsdhFmUQszoTQzh1t1GLD65T7V3L8kdJOsxncdQPM%2Fuwh1WCQrjMbsk6JT5EqKZajG%2Fe0y%2Fw%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20240116T180834Z&X-Amz-SignedHeaders=host&X-Amz-Expires=3600&X-Amz-Credential=ASIA6GBMC5U334YINNMZ%2F20240116%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=42198766491be57207a69efe17afb5aa8b585e45b52c97fe9b89ad753ab65b0a', model_dir= 'colorizers',map_location='cpu', check_hash=True))
    return model

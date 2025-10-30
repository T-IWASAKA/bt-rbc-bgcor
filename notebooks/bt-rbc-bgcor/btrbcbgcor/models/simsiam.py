# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 12:09:08 2019

Barlow Twins

models file

@author: tadahaya
"""
import torch
import torch.nn as nn

def flatten(t):
    return t.reshape(t.shape[0], -1)


class NetWrapper(nn.Module):
    """ inspired by https://github.com/lucidrains/byol-pytorch """
    def __init__(self, net, layer=-2):
        super().__init__()
        self.net = net
        self.layer = layer
        self.hidden = None
        self.hook_registered = False


    def _find_layer(self):
        if type(self.layer)==str: # 名称で取得
            modules = dict([*self.net.named_modules()])
            return modules.get(self.layer, None)
        elif type(self.layer)==int: # indexで取得
            children = [*self.net.children()]
            return children[self.layer]
        return None


    def _hook(self, _, __, output):
        #print(output.shape)
        #self.hidden = flatten(output) # ViTのため平坦化せずそのまま流す！
        self.hidden = output


    def _register_hook(self):
        layer = self._find_layer()
        assert layer is not None, f"!! Hidden layer ({self.layer}) not found !!"
        handle = layer.register_forward_hook(self._hook)
        self.hook_registered = True


    def get_representation(self, x):
        if self.layer==-1:
            return self.net(x)
        if not self.hook_registered:
            self._register_hook()
        _ = self.net(x)
        hidden = self.hidden
        self.hidden = None # self.hiddenを初期化している
        assert hidden is not None, f"!! Hidden layer ({self.layer}) never emitted an output !!"
        return hidden


    def forward(self, x):
        representation = self.get_representation(x)
        #print(representation.shape)
        representation = representation[:, 0, :] # CLSトークンである一番上を取得

        return representation


class SimSiam(nn.Module):
    """
    single GPU version based on https://github.com/facebookresearch/barlowtwins

    """
    def __init__(self, backbone, latent_id, projection_sizes):
        """
        Parameters
        ----------
        backbone: Model

        latent_id: name or index of the layer to be fed to the projection

        projection_sizes: size of the hidden layers in the projection

        lambd: tradeoff function

        scale_factor: factor to scale loss by

        """
        super().__init__()
        self.backbone = NetWrapper(backbone, layer=latent_id)

        # projector
        sizes = projection_sizes
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False)) # BatchNorm入れるのでbias=False
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False)) # BatchNorm入れるのでbias=False
        self.projector = nn.Sequential(*layers)

        # predictor
        self.predictor = nn.Sequential(
            nn.Linear(sizes[-1], sizes[-1], bias=False),
            nn.BatchNorm1d(sizes[-1]),
            nn.ReLU(inplace=True),
            nn.Linear(sizes[-1], sizes[-1])
        )


    def forward(self, y1, y2): # 2つの画像を入力
        z1 = self.backbone(y1)
        z2 = self.backbone(y2)
        z1 = self.projector(z1)
        z2 = self.projector(z2)
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        # predictorを通らない側の逆伝播を固定
        z1 = z1.detach()
        z2 = z2.detach()

        # lossは双方向
        loss = -0.5 * (
            nn.functional.cosine_similarity(p1, z2, dim=-1).mean() +
            nn.functional.cosine_similarity(p2, z1, dim=-1).mean()
        )

        return loss
    

class LinearHead(nn.Module):
    def __init__(self, backbone, num_classes:int):
        super().__init__()
        self.backbone = backbone
        self.fc = nn.Linear(backbone.fc.in_features, num_classes)

    
    def forward(self, x):
        out = self.backbone(x)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out
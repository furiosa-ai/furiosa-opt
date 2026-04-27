#!/usr/bin/env python3
"""MNIST FC model: train, export, infer. See README.md for the full example.

Usage:
    python mnist.py         # train + export + infer (CPU)
    python mnist.py infer   # infer only (CPU)
"""

import argparse
from itertools import zip_longest
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from safetensors.torch import save_file, load_file

DIR = Path(__file__).parent
OUT = DIR.parents[1] / "data" / "mnist" / "mnist.safetensors"
TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])


KERNEL = DIR.parents[2] / "target" / "furiosa-opt" / "kernel" / "furiosa-visa-examples" / "mnist__forward.edf"


class Model(nn.Module):
    """784 → 256 (ReLU) → 10. Must match VISA kernel in mod.rs."""
    OP = "furiosa::mnist"
    EDF = None

    def __init__(self, device="cpu"):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)
        if device == "npu":
            from furiosa.torch._C import ir
            from furiosa.torch import EdfModule, set_fusion
            set_fusion(8)
            Model.EDF = EdfModule(ir.Edf.deserialize(KERNEL.read_bytes()))
            Model.EDF.to("furiosa:0")

    def forward(self, x):
        return torch.ops.furiosa.mnist(
            x, self.fc1.weight, self.fc1.bias, self.fc2.weight, self.fc2.bias,
        )

    @staticmethod
    def cpu(x, w1, b1, w2, b2):
        return F.linear(F.relu(F.linear(x.view(-1, 784), w1, b1)), w2, b2)

    @staticmethod
    def npu(x, w1, b1, w2, b2):
        x_pad  = F.pad(x.view(-1, 784), (0, 16)).bfloat16().squeeze(0)
        w1_pad = F.pad(w1, (0, 16)).bfloat16()
        w2_pad = F.pad(w2, (0, 0, 0, 6)).bfloat16()
        b2_pad = F.pad(b2, (0, 6)).bfloat16()
        return Model.EDF(x_pad, w1_pad, b1.bfloat16(), w2_pad, b2_pad)[0][:10].float().unsqueeze(0)


torch.library.define(Model.OP, "(Tensor x, Tensor w1, Tensor b1, Tensor w2, Tensor b2) -> Tensor")
torch.library.impl(Model.OP, "CompositeImplicitAutograd")(Model.cpu)
torch.library.impl(Model.OP, "PrivateUse1")(Model.npu)


def mnist(train: bool):
    return datasets.MNIST(str(DIR / "raw"), train=train, download=True, transform=TRANSFORM)


def train(epochs=3) -> Model:
    torch.manual_seed(42)
    model = Model()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(1, epochs + 1):
        for img, lbl in torch.utils.data.DataLoader(mnist(True), 64, shuffle=True):
            opt.zero_grad()
            F.cross_entropy(model(img), lbl).backward()
            opt.step()

    model.eval()
    correct = sum(
        (model(img).argmax(1) == lbl).sum().item()
        for img, lbl in torch.utils.data.DataLoader(mnist(False), 1000)
    )
    print(f"accuracy: {correct}/10000 ({correct / 100:.1f}%)")
    return model


def export(model: Model, n=10):
    test = mnist(False)
    tensors = {
        "fc1.weight": model.fc1.weight.detach().bfloat16(),
        "fc1.bias":   model.fc1.bias.detach().bfloat16(),
        "fc2.weight": model.fc2.weight.detach().bfloat16(),
        "fc2.bias":   model.fc2.bias.detach().bfloat16(),
    }
    # padded for VISA kernel (input 784→800, output 10→16)
    tensors["hw.fc1.weight"] = F.pad(model.fc1.weight, (0, 16)).detach().bfloat16()
    tensors["hw.fc2.weight"] = F.pad(model.fc2.weight, (0, 0, 0, 6)).detach().bfloat16()
    tensors["hw.fc2.bias"] = F.pad(model.fc2.bias, (0, 6)).detach().bfloat16()

    for i in range(n):
        img, lbl = test[i]
        tensors[f"image_{i}"] = img.view(784).bfloat16()
        tensors[f"hw.image_{i}"] = F.pad(img.view(784), (0, 16)).bfloat16()
        tensors[f"label_{i}"] = torch.tensor([lbl], dtype=torch.int32)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(OUT))
    print(f"exported: {OUT}")


def render(i, img, logits, lbl):
    blocks = {(0, 0): "  ", (1, 0): "▀▀", (0, 1): "▄▄", (1, 1): "██"}
    probs = F.softmax(logits, -1).squeeze(0)
    pred = int(probs.argmax().item())

    v = img.view(28, 28)
    v = (v - v.min()) / max(float(v.max() - v.min()), 1e-6)
    pix = (v > 0.25).int().tolist()
    digit = [
        "".join(blocks[pix[r][c], pix[r + 1][c]] for c in range(28))
        for r in range(0, 28, 2)
    ]

    bars = []
    for d in range(10):
        p = float(probs[d])
        fill = round(p * 20)
        mark = " ← pred" if d == pred else " (label)" if d == lbl else ""
        bars.append(f"{d}  {'█' * fill}{'·' * (20 - fill)} {p:5.1%}{mark}")

    print(f"\n[{i}] label={lbl} pred={pred} {'✓' if pred == lbl else '✗'}")
    for d, b in zip_longest(digit, bars, fillvalue=""):
        print(f"  {d:<56}  {b}")
    return pred


def infer(n=10, device="cpu"):
    tensors = load_file(str(OUT))
    model = Model(device=device)
    model.fc1.weight.data = tensors["fc1.weight"].float()
    model.fc1.bias.data   = tensors["fc1.bias"].float()
    model.fc2.weight.data = tensors["fc2.weight"].float()
    model.fc2.bias.data   = tensors["fc2.bias"].float()
    model.eval()

    if device == "npu":
        model = model.to("furiosa:0")

    correct = 0
    for i in range(n):
        img = tensors[f"image_{i}"].float()
        if device == "npu":
            img = img.to("furiosa:0")
        lbl = tensors[f"label_{i}"].item()
        logits = model(img.unsqueeze(0))
        if render(i, img, logits, lbl) == lbl:
            correct += 1
    print(f"\n{correct}/{n} correct")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", nargs="?", default="all", choices=["all", "infer"])
    p.add_argument("--device", default="cpu", choices=["cpu", "npu"])
    args = p.parse_args()

    if args.cmd == "all":
        model = train()
        export(model)
    infer(device=args.device)

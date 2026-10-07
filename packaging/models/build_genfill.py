"""Build generative fill's model files (modelstore.PACKS["genfill"]) from the
published weights, the way the release assets were made.

    uv run --python 3.12 --with torch==2.14.1 --with diffusers==0.41.0 \
        --with transformers==5.19.0 --with peft==0.21.2 --with accelerate \
        --with safetensors --with onnx==1.23.2 --with onnxruntime==1.30.0 \
        python packaging/models/build_genfill.py OUT_DIR

It downloads about 2.3 GB from Hugging Face, needs about 16 GB of memory and
takes a few minutes on a laptop. At the end it prints each file's size and
SHA-256, which go into modelstore.PACKS; upload every file in OUT_DIR as an
asset of the release modelstore.RELEASE names.

The steps, each for a reason recorded beside it:

1. Stable Diffusion 1.5 inpainting (fp16 safetensors) with the LCM-LoRA for
   SD 1.5 fused in at scale 1.0, so four LCM steps do the work of fifty.
2. The UNet's attention set to diffusers' classic processor before export.
   The default one exports scaled_dot_product_attention as a pattern
   onnxruntime's optimizer does not recognise; the classic one fuses into 32
   Attention and MultiHeadAttention nodes, which run on the CPU at the same
   speed and with about 0.8 GB less memory.
3. ONNX export (opset 17, the TorchScript exporter) of the UNet, the VAE
   encoder (to the latent distribution's mean, so the fill does not sample)
   and the VAE decoder, at the 512 square the app works at.
4. onnxruntime's transformer optimizer on the UNet, with every fusion that
   only has a CUDA kernel switched off (GroupNorm, NhwcConv, BiasAdd,
   BiasSplitGelu, packed QKV/KV), so the result runs on the CPU provider.
5. Every float weight of 1024 values or more stored as float16, followed by
   a Cast to float32, in one external data file per graph: half the download,
   and onnxruntime folds the casts at load, so it computes in float32.
6. The empty prompt's text embedding saved as an array, so the app needs no
   tokenizer and no text encoder.
7. Both licences and a NOTICE of these changes copied in, as section 4 of
   the licences requires of anyone passing the weights on.
"""
from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnx
import torch
from diffusers import StableDiffusionInpaintPipeline
from diffusers.models.attention_processor import AttnProcessor
from onnx import TensorProto, helper, numpy_helper

HERE = Path(__file__).resolve().parent
LICENCES = HERE.parents[1] / "src" / "pickapicka" / "web" / "licences"
BASE = "stable-diffusion-v1-5/stable-diffusion-inpainting"
LORA = "latent-consistency/lcm-lora-sdv1-5"
CPU_ONLY_FLAGS = ["--disable_group_norm", "--disable_skip_group_norm", "--disable_nhwc_conv",
                  "--disable_bias_splitgelu", "--disable_bias_add", "--disable_packed_qkv",
                  "--disable_packed_kv"]


class _Unet(torch.nn.Module):
    def __init__(self, u):
        super().__init__()
        self.u = u

    def forward(self, sample, timestep, encoder_hidden_states):
        return self.u(sample, timestep, encoder_hidden_states, return_dict=False)[0]


class _Encoder(torch.nn.Module):
    def __init__(self, v):
        super().__init__()
        self.v = v

    def forward(self, image):
        return self.v.encode(image).latent_dist.mean


class _Decoder(torch.nn.Module):
    def __init__(self, v):
        super().__init__()
        self.v = v

    def forward(self, latent):
        return self.v.decode(latent, return_dict=False)[0]


def export(work: Path) -> None:
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        BASE, variant="fp16", torch_dtype=torch.float32, safety_checker=None, requires_safety_checker=False)
    pipe.load_lora_weights(LORA)
    pipe.fuse_lora()
    pipe.unload_lora_weights()
    pipe.unet.set_attn_processor(AttnProcessor())
    unet, vae = pipe.unet.eval(), pipe.vae.eval()
    with torch.no_grad():
        ids = pipe.tokenizer("", padding="max_length", max_length=pipe.tokenizer.model_max_length,
                             truncation=True, return_tensors="pt").input_ids
        emb = pipe.text_encoder(ids)[0].numpy().astype(np.float32)
        np.save(work / "empty_prompt.npy", emb)
        torch.onnx.export(_Encoder(vae), (torch.randn(1, 3, 512, 512),), str(work / "vae_encoder.onnx"),
                          input_names=["image"], output_names=["latent"], opset_version=17, dynamo=False)
        torch.onnx.export(_Decoder(vae), (torch.randn(1, 4, 64, 64),), str(work / "vae_decoder.onnx"),
                          input_names=["latent"], output_names=["image"], opset_version=17, dynamo=False)
        (work / "unet").mkdir(exist_ok=True)
        torch.onnx.export(_Unet(unet), (torch.randn(1, 9, 64, 64), torch.tensor([999], dtype=torch.int64),
                                        torch.from_numpy(emb)),
                          str(work / "unet" / "unet.onnx"),
                          input_names=["sample", "timestep", "encoder_hidden_states"], output_names=["noise"],
                          opset_version=17, dynamo=False)


def optimize(work: Path) -> Path:
    out = work / "unet_opt" / "unet.onnx"
    out.parent.mkdir(exist_ok=True)
    subprocess.run([sys.executable, "-m", "onnxruntime.transformers.optimizer",
                    "--input", str(work / "unet" / "unet.onnx"), "--output", str(out),
                    "--model_type", "unet", "--use_multi_head_attention", *CPU_ONLY_FLAGS,
                    "--use_external_data_format", "--provider", "cpu", "--opt_level", "0"], check=True)
    return out


def pack(src: Path, name: str, dst: Path) -> None:
    m = onnx.load(str(src), load_external_data=True)
    g = m.graph
    inits, casts = [], []
    for init in g.initializer:
        if init.data_type == TensorProto.FLOAT and int(np.prod(init.dims)) >= 1024:
            half = numpy_helper.from_array(numpy_helper.to_array(init).astype(np.float16), init.name + "__fp16")
            inits.append(half)
            casts.append(helper.make_node("Cast", [half.name], [init.name], to=TensorProto.FLOAT,
                                          name=init.name + "__cast"))
        else:
            inits.append(init)
    del g.initializer[:]
    g.initializer.extend(inits)
    nodes = list(g.node)
    del g.node[:]
    g.node.extend(casts + nodes)
    # onnx appends external data to a file already there, so start clean.
    (dst / f"{name}.onnx.data").unlink(missing_ok=True)
    onnx.save_model(m, str(dst / f"{name}.onnx"), save_as_external_data=True, all_tensors_to_one_file=True,
                    location=f"{name}.onnx.data", size_threshold=1024)
    onnx.checker.check_model(str(dst / f"{name}.onnx"))


def main() -> None:
    out = Path(sys.argv[1]).resolve()
    work = out / "_work"
    work.mkdir(parents=True, exist_ok=True)
    export(work)
    pack(optimize(work), "unet", out)
    pack(work / "vae_encoder.onnx", "vae_encoder", out)
    pack(work / "vae_decoder.onnx", "vae_decoder", out)
    shutil.copy(work / "empty_prompt.npy", out / "empty_prompt.npy")
    shutil.copy(LICENCES / "CreativeML-OpenRAIL-M.txt", out / "LICENSE-CreativeML-OpenRAIL-M.txt")
    shutil.copy(LICENCES / "CreativeML-OpenRAIL++-M.txt", out / "LICENSE-CreativeML-OpenRAIL++-M.txt")
    shutil.copy(HERE / "genfill-NOTICE.txt", out / "NOTICE.txt")
    shutil.rmtree(work)
    for f in sorted(out.iterdir()):
        h = hashlib.sha256()
        with f.open("rb") as fh:
            while chunk := fh.read(1 << 20):
                h.update(chunk)
        print(f'("{f.name}", {f.stat().st_size}, "{h.hexdigest()}"),')


if __name__ == "__main__":
    main()

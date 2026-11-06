import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gguf_reader import GGUFFile

ARCHITECTURE = "lfm2moe"
ENGINE_TYPES = frozenset({"Q4_K", "Q5_K", "Q6_K", "Q8_0", "IQ3_S", "IQ4_XS", "F32", "F16"})
EXPERT_ROLES = ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps")
GIB = 1 << 30


@dataclass(frozen=True)
class Hyper:
    layers: int
    embd: int
    ffn: int
    heads: int
    kv_heads: tuple
    experts: int
    experts_used: int
    expert_ffn: int
    dense_lead: int
    conv_cache: int
    vocab: int
    key_length: int = 0
    value_length: int = 0

    @property
    def head_dim(self) -> int:
        return self.key_length or self.embd // self.heads

    @property
    def value_dim(self) -> int:
        return self.value_length or self.embd // self.heads

    def is_conv(self, layer: int) -> bool:
        return self.kv_heads[layer] == 0

    def is_moe(self, layer: int) -> bool:
        return layer >= self.dense_lead


def hyper_of(metadata: dict) -> Hyper:
    """The sizes of an lfm2moe file, read from its metadata; ValueError when the file is another architecture or lacks a key."""
    architecture = metadata.get("general.architecture")
    if architecture != ARCHITECTURE:
        raise ValueError(f"the architecture is {architecture!r}, not {ARCHITECTURE!r}")

    def need(key: str):
        value = metadata.get(f"{ARCHITECTURE}.{key}")
        if value is None:
            raise ValueError(f"the metadata lacks {ARCHITECTURE}.{key}")
        return value

    layers = need("block_count")
    kv_heads = need("attention.head_count_kv")
    if not isinstance(kv_heads, list) or len(kv_heads) != layers:
        raise ValueError("attention.head_count_kv must list one value per layer")
    return Hyper(layers, need("embedding_length"), need("feed_forward_length"), need("attention.head_count"), tuple(kv_heads),
                 need("expert_count"), need("expert_used_count"), need("expert_feed_forward_length"), need("leading_dense_block_count"),
                 need("shortconv.l_cache"), need("vocab_size"), metadata.get(f"{ARCHITECTURE}.attention.key_length", 0),
                 metadata.get(f"{ARCHITECTURE}.attention.value_length", 0))


def expected_shapes(h: Hyper) -> dict:
    """name -> shape of every tensor the file must hold, as GGUF stores them (the fastest index first); `output.weight` is optional
    (the embedding is used again when it is missing) and is handled apart."""
    shapes = {"token_embd.weight": [h.embd, h.vocab], "token_embd_norm.weight": [h.embd]}
    for layer in range(h.layers):
        prefix = f"blk.{layer}."
        shapes[prefix + "attn_norm.weight"] = [h.embd]
        shapes[prefix + "ffn_norm.weight"] = [h.embd]
        if h.is_conv(layer):
            shapes[prefix + "shortconv.conv.weight"] = [h.conv_cache, h.embd]
            shapes[prefix + "shortconv.in_proj.weight"] = [h.embd, 3 * h.embd]
            shapes[prefix + "shortconv.out_proj.weight"] = [h.embd, h.embd]
        else:
            shapes[prefix + "attn_q.weight"] = [h.embd, h.heads * h.head_dim]
            shapes[prefix + "attn_k.weight"] = [h.embd, h.kv_heads[layer] * h.head_dim]
            shapes[prefix + "attn_v.weight"] = [h.embd, h.kv_heads[layer] * h.value_dim]
            shapes[prefix + "attn_output.weight"] = [h.heads * h.value_dim, h.embd]
            shapes[prefix + "attn_q_norm.weight"] = [h.head_dim]
            shapes[prefix + "attn_k_norm.weight"] = [h.head_dim]
        if h.is_moe(layer):
            shapes[prefix + "ffn_gate_inp.weight"] = [h.embd, h.experts]
            shapes[prefix + "ffn_gate_exps.weight"] = [h.embd, h.expert_ffn, h.experts]
            shapes[prefix + "ffn_up_exps.weight"] = [h.embd, h.expert_ffn, h.experts]
            shapes[prefix + "ffn_down_exps.weight"] = [h.expert_ffn, h.embd, h.experts]
            shapes[prefix + "exp_probs_b.bias"] = [h.experts]
        else:
            shapes[prefix + "ffn_gate.weight"] = [h.embd, h.ffn]
            shapes[prefix + "ffn_up.weight"] = [h.embd, h.ffn]
            shapes[prefix + "ffn_down.weight"] = [h.ffn, h.embd]
    return shapes


def optional_shapes(h: Hyper, names: set) -> dict:
    """Tensors the loader of llama.cpp accepts but does not require: the output matrix, a fused query/key/value matrix instead of the
    three (the fused one then replaces them), their biases, and the extra output layer of the embedding variants."""
    shapes = {"output.weight": [h.embd, h.vocab]}
    for layer in range(h.layers):
        if h.is_conv(layer):
            continue
        prefix = f"blk.{layer}."
        q, k, v = h.heads * h.head_dim, h.kv_heads[layer] * h.head_dim, h.kv_heads[layer] * h.value_dim
        shapes[prefix + "attn_qkv.weight"] = [h.embd, q + k + v]
        shapes[prefix + "attn_qkv.bias"] = [q + k + v]
        shapes[prefix + "attn_q.bias"] = [q]
        shapes[prefix + "attn_k.bias"] = [k]
        shapes[prefix + "attn_v.bias"] = [v]
    if "dense_2.weight" in names:
        shapes["dense_2.weight"] = None
        shapes["dense_2.bias"] = None
    return shapes


def check(h: Hyper, tensors: list) -> list:
    """The problems of a tensor list against the layout of the architecture: missing, unknown and wrongly shaped tensors."""
    expected = expected_shapes(h)
    names = {tensor.name for tensor in tensors}
    optional = optional_shapes(h, names)
    for layer in range(h.layers):
        prefix = f"blk.{layer}."
        if prefix + "attn_qkv.weight" in names:
            for part in ("attn_q", "attn_k", "attn_v"):
                expected.pop(f"{prefix}{part}.weight", None)
    seen = {}
    problems = []
    for tensor in tensors:
        if tensor.name in seen:
            problems.append(f"{tensor.name}: listed twice")
        seen[tensor.name] = tensor
        if tensor.name in optional:
            want = optional[tensor.name]
            if want is not None and list(tensor.shape) != want:
                problems.append(f"{tensor.name}: shape {list(tensor.shape)}, expected {want}")
            if tensor.name == "dense_2.weight" and (len(tensor.shape) != 2 or tensor.shape[0] != h.embd):
                problems.append(f"dense_2.weight: shape {list(tensor.shape)}, expected [{h.embd}, n]")
        elif tensor.name not in expected:
            problems.append(f"{tensor.name}: not part of the {ARCHITECTURE} layout")
        elif list(tensor.shape) != expected[tensor.name]:
            problems.append(f"{tensor.name}: shape {list(tensor.shape)}, expected {expected[tensor.name]}")
    problems += [f"{name}: missing" for name in expected if name not in seen]
    problems += [f"blk.{layer}: has an attn_qkv.weight and also separate attn_q/k/v weights" for layer in range(h.layers)
                 if f"blk.{layer}.attn_qkv.weight" in names and f"blk.{layer}.attn_q.weight" in names]
    return problems


def role_of(name: str) -> str:
    return name.split(".", 2)[2].rsplit(".", 1)[0] if name.startswith("blk.") else name.rsplit(".", 1)[0]


def summarize(h: Hyper, tensors: list) -> dict:
    """Bytes by role and quantization, the layer pattern, and the bytes of experts one decoded token reads."""
    bytes_by_role: dict = {}
    types_by_role: dict = {}
    unsupported = set()
    for tensor in tensors:
        size = tensor.expected_bytes() or 0
        role = role_of(tensor.name)
        bytes_by_role[role] = bytes_by_role.get(role, 0) + size
        types_by_role.setdefault(role, set()).add(tensor.type_name)
        if tensor.type_name not in ENGINE_TYPES:
            unsupported.add(tensor.type_name)
    expert_bytes = sum(bytes_by_role.get(role, 0) for role in EXPERT_ROLES)
    moe_layers = h.layers - h.dense_lead
    return {
        "pattern": "".join("c" if h.is_conv(layer) else "A" for layer in range(h.layers)),
        "total_bytes": sum(bytes_by_role.values()),
        "expert_bytes": expert_bytes,
        "other_bytes": sum(bytes_by_role.values()) - expert_bytes,
        "expert_bytes_per_token": expert_bytes * h.experts_used // h.experts if moe_layers else 0,
        "expert_bytes_per_layer": expert_bytes // moe_layers if moe_layers else 0,
        "kv_bytes_per_token": sum(kv * (h.head_dim + h.value_dim) * 2 for kv in h.kv_heads),
        "bytes_by_role": bytes_by_role,
        "types_by_role": {role: sorted(types) for role, types in types_by_role.items()},
        "unsupported_types": sorted(unsupported),
    }


def report(path: Path) -> tuple:
    model = GGUFFile(path)
    h = hyper_of(model.metadata)
    problems = check(h, model.tensors)
    summary = summarize(h, model.tensors)
    lines = [f"{path.name}: {len(model.tensors)} tensors, {h.layers} layers, pattern {summary['pattern']} (c = short convolution, A = attention)",
             f"experts {h.experts_used} of {h.experts}, expert size {h.expert_ffn}, {h.dense_lead} dense layers first, vocabulary {h.vocab}",
             (f"file {summary['total_bytes'] / GIB:.2f} GiB: experts {summary['expert_bytes'] / GIB:.2f} GiB "
              f"({summary['expert_bytes'] / summary['total_bytes']:.1%}), the rest {summary['other_bytes'] / GIB:.2f} GiB"),
             (f"one decoded token reads {summary['expert_bytes_per_token'] / 1e6:.0f} MB of experts, one layer of experts is "
              f"{summary['expert_bytes_per_layer'] / (1 << 20):.0f} MiB, the cache of attention is {summary['kv_bytes_per_token'] / 1024:.0f} KiB per token (f16)")]
    for role, types in sorted(summary["types_by_role"].items()):
        lines.append(f"  {role:28} {'/'.join(types):12} {summary['bytes_by_role'][role] / (1 << 20):9.1f} MiB")
    if summary["unsupported_types"]:
        lines.append("types without a kernel of the engine: " + ", ".join(summary["unsupported_types"]))
    lines += [f"PROBLEM {problem}" for problem in problems]
    return lines, problems


def main(argv: list) -> int:
    parser = argparse.ArgumentParser(description="Check an lfm2moe GGUF file against the layout of the architecture and summarize it.")
    parser.add_argument("model", type=Path)
    args = parser.parse_args(argv)
    try:
        lines, problems = report(args.model)
    except (OSError, ValueError) as error:
        print(f"lfm-model-map: {error}", file=sys.stderr)
        return 2
    print("\n".join(lines))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

"""Megatron-Core baseline: train GPTModel with given TP/PP config, collect metrics."""

import os
import sys
import time
import argparse

import torch
import torch.distributed as dist

from megatron.core import parallel_state
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.distributed.finalize_model_grads import finalize_model_grads

MODEL_CONFIG = {
    "num_layers": 12,
    "hidden_size": 512,
    "num_attention_heads": 8,
    "ffn_hidden_size": 2048,
    "max_seq_len": 512,
    "vocab_size": 50304,  # pad to nearest multiple of 64 for TP divisibility
}


def init_distributed(tp, pp):
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=tp,
        pipeline_model_parallel_size=pp,
    )


def build_model(tp, pp):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    config = TransformerConfig(
        num_layers=MODEL_CONFIG["num_layers"],
        hidden_size=MODEL_CONFIG["hidden_size"],
        num_attention_heads=MODEL_CONFIG["num_attention_heads"],
        ffn_hidden_size=MODEL_CONFIG["ffn_hidden_size"],
        use_cpu_initialization=True,
        pipeline_dtype=torch.float32,
        bf16=False,
        fp16=False,
        sequence_parallel=False,
        tensor_model_parallel_size=tp,
        pipeline_model_parallel_size=pp,
        pipeline_model_parallel_comm_backend="nccl",
        tp_comm_overlap=True,
    )

    pp_rank = parallel_state.get_pipeline_model_parallel_rank()
    pp_size = parallel_state.get_pipeline_model_parallel_world_size()
    pre_process = pp_rank == 0
    post_process = pp_rank == pp_size - 1

    model = GPTModel(
        config=config,
        transformer_layer_spec=get_gpt_layer_local_spec(),
        vocab_size=MODEL_CONFIG["vocab_size"],
        max_sequence_length=MODEL_CONFIG["max_seq_len"],
        pre_process=pre_process,
        post_process=post_process,
        position_embedding_type="learned_absolute",
    )

    return model, config


def make_data_iterator(seq_len, micro_batch_size, vocab_size):
    """Yield random token batches indefinitely."""
    while True:
        tokens = torch.randint(0, vocab_size, (micro_batch_size, seq_len), device="cuda")
        labels = tokens.clone()
        loss_mask = torch.ones(micro_batch_size, seq_len, dtype=torch.float32, device="cuda")
        position_ids = torch.arange(seq_len, device="cuda").unsqueeze(0).expand(micro_batch_size, -1)
        # Causal mask: True = mask out (future), False = attend (past+current)
        att = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device="cuda"), diagonal=1)
        attention_mask = att.unsqueeze(0).unsqueeze(0)
        yield {
            "tokens": tokens,
            "labels": labels,
            "loss_mask": loss_mask,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
        }


def forward_step_func(data_iterator, model):
    def loss_func(loss_mask, output_tensor):
        losses = output_tensor.float()
        loss_mask = loss_mask.view(-1).float()
        loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()
        return loss, {"lm loss": loss}

    batch = next(data_iterator)
    out = model(
        batch["tokens"],
        batch["position_ids"],
        batch["attention_mask"],
        labels=batch["labels"],
    )
    return out, lambda out: loss_func(batch["loss_mask"], out)


def compute_mfu(model, seq_len, micro_batch_size, tp, pp, elapsed, num_steps):
    total_params = sum(p.numel() for p in model.parameters())
    # Approx FLOPs per step: 24 * L * h^2 * B * s * (4 + ...) per transformer layer
    # Simplified: ~ 96 * L * h^2 * B * s / (TP * PP) per GPU
    L = MODEL_CONFIG["num_layers"]
    h = MODEL_CONFIG["hidden_size"]
    B = micro_batch_size
    s = seq_len
    V = MODEL_CONFIG["vocab_size"]
    gpu_world = tp * pp
    dp_world = max(1, int(os.environ.get("WORLD_SIZE", "1")) // gpu_world)

    flops_per_step = (
        24 * L * h * h * B * s * (4 + 0.5)  # attention + FFN
        + 6 * L * h * s * V  # embedding / LM head
    ) * dp_world  # data parallel replicated

    total_flops = flops_per_step * num_steps
    mfu = total_flops / (elapsed * 96e12 * gpu_world)  # L20 FP16 peak ~96 TFLOPS
    return min(mfu, 1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--micro-batch-size", type=int, default=4)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--warmup-steps", type=int, default=10)
    args = parser.parse_args()

    tp = args.tp
    pp = args.pp
    B = args.micro_batch_size
    total_steps = args.num_steps
    warmup = args.warmup_steps
    S = MODEL_CONFIG["max_seq_len"]
    V = MODEL_CONFIG["vocab_size"]

    init_distributed(tp, pp)
    model_parallel_cuda_manual_seed(42)

    model, config = build_model(tp, pp)
    model.cuda()

    ddp = DistributedDataParallel(
        config=config,
        ddp_config=DistributedDataParallelConfig(
            grad_reduce_in_fp32=False,
            overlap_grad_reduce=False,
            use_distributed_optimizer=False,
        ),
        module=model,
    )

    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)
    forward_backward_func = get_forward_backward_func()

    rank = dist.get_rank()
    is_last = parallel_state.is_pipeline_last_stage()
    total_world = int(os.environ["WORLD_SIZE"])
    num_microbatches = max(1, int(pp * 2))  # need >= PP to fill pipeline

    # Warmup
    warmup_iter = make_data_iterator(S, B, V)
    for _ in range(warmup):
        optim.zero_grad()
        losses = forward_backward_func(
            forward_step_func=forward_step_func,
            data_iterator=warmup_iter,
            model=ddp,
            num_microbatches=num_microbatches,
            seq_length=S,
            micro_batch_size=B,
            decoder_seq_length=S,
            forward_only=False,
        )
        finalize_model_grads([ddp])
        optim.step()
    torch.cuda.synchronize()

    # Benchmark
    torch.cuda.reset_peak_memory_stats()
    data_iter = make_data_iterator(S, B, V)
    start = time.perf_counter()
    all_losses = []
    for step in range(total_steps):
        optim.zero_grad()
        losses = forward_backward_func(
            forward_step_func=forward_step_func,
            data_iterator=data_iter,
            model=ddp,
            num_microbatches=num_microbatches,
            seq_length=S,
            micro_batch_size=B,
            decoder_seq_length=S,
            forward_only=False,
        )
        finalize_model_grads([ddp])
        optim.step()
        torch.cuda.synchronize()

        if is_last and losses:
            loss_val = losses[0]["lm loss"].item()
            all_losses.append(loss_val)
            if rank == 0 and (step + 1) % 10 == 0:
                print(f"step {step+1:4d} | loss {loss_val:.4f}")
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    # Report
    peak_mem = torch.cuda.max_memory_allocated() / 1024**3
    dp_world = max(1, total_world // (tp * pp))
    tokens_per_step = B * S * dp_world  # Only DP multiplies tokens (TP splits model, not data)
    total_tokens = tokens_per_step * total_steps
    throughput = total_tokens / elapsed
    mfu = compute_mfu(model, S, B, tp, pp, elapsed, total_steps)

    if rank == 0:
        print(f"\n{'='*60}")
        print(f"Megatron-Core Baseline Results")
        print(f"{'='*60}")
        print(f"Config:          TP={tp} PP={pp} ({total_world} GPU, DP={dp_world})")
        print(f"Model:           {MODEL_CONFIG['hidden_size']}hid {MODEL_CONFIG['num_layers']}lay {MODEL_CONFIG['num_attention_heads']}head")
        print(f"Micro batch:     {B}  |  Seq len: {S}  |  Steps: {total_steps}")
        print(f"{'-'*60}")
        print(f"Throughput:      {throughput:,.0f} tok/s")
        print(f"Scaling eff:     {throughput / (gpu_count * B * S / 0.01):.2%}")
        print(f"Peak memory:     {peak_mem:.2f} GB/GPU")
        print(f"MFU:             {mfu:.2%}")
        print(f"Final loss:      {all_losses[-1]:.4f}" if all_losses else "")
        print(f"{'='*60}\n")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()

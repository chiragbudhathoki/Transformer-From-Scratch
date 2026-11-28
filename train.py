import argparse
import os

import numpy as np
import torch

from cs336_basics.linear import (
    transformer_lm,
    cross_entropy,
    AdamW,
    lr_cosine_scheduler,
    gradient_clipping,
    get_batch,
    save_checkpoint,
    load_checkpoint,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a language model.")

    # Data / output
    parser.add_argument("--train_data", type=str, required=True, help="Path to the training tokens (.npy or raw uint16).")
    parser.add_argument("--val_data", type=str, required=True, help="Path to the validation tokens (.npy or raw uint16).")
    parser.add_argument("--out_dir", type=str, required=True, help="Directory for checkpoints.")
    parser.add_argument("--resume_from", type=str, default=None, help="Checkpoint path to resume from.")

    # Model (vocab_size must match your tokenizer, so it is required)
    parser.add_argument("--vocab_size", type=int, required=True)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--rope_theta", type=float, default=10000.0)

    # Optimization
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_iters", type=int, default=10000)
    parser.add_argument("--max_lr", type=float, default=1e-3)
    parser.add_argument("--min_lr", type=float, default=1e-4)
    parser.add_argument("--warmup_iters", type=int, default=100)
    parser.add_argument(
        "--cosine_cycle_iters",
        type=int,
        default=None,
        help="Iteration at which cosine decay ends. Defaults to --max_iters.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    # Logging / eval / checkpointing
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--eval_iters", type=int, default=20)
    parser.add_argument("--checkpoint_interval", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    if args.cosine_cycle_iters is None:
        args.cosine_cycle_iters = args.max_iters
    return args


def load_tokens(path):
    if os.path.splitext(path)[1].lower() == ".npy":
        return np.load(path, mmap_mode="r")
  
    return np.memmap(path, dtype=np.uint16, mode="r")


@torch.no_grad()
def evaluate(model, data, eval_iters, batch_size, context_length, device):
    model.eval()
    losses = []
    for _ in range(eval_iters):
        x, y = get_batch(data, batch_size, context_length, device)
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            logits = model(x)
            losses.append(cross_entropy(logits, y).item())
    model.train()
    return float(np.mean(losses))


def main():
    args = parse_args()
    device = torch.device(args.device)

    train_data = load_tokens(args.train_data)
    val_data = load_tokens(args.val_data)

    model = transformer_lm(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        max_seq_len=args.context_length,
        theta=args.rope_theta,
    ).to(device)

    
    optimizer = AdamW(
        model.parameters(),
        lr=args.max_lr,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
    )

    os.makedirs(args.out_dir, exist_ok=True)

    # Resume (assumes load_checkpoint(src, model, optimizer) -> iteration)
    start_step = 0
    if args.resume_from is not None:
        start_step = load_checkpoint(args.resume_from, model, optimizer)
        print(f"Resumed from {args.resume_from} at step {start_step}")

    model.train()

    for step in range(start_step, args.max_iters):
        # Learning rate schedule
        lr = lr_cosine_scheduler(
            step,
            args.max_lr,
            args.min_lr,
            args.warmup_iters,
            args.cosine_cycle_iters,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        # Periodic validation
        if step % args.eval_interval == 0:
            val_loss = evaluate(
                model, val_data, args.eval_iters,
                args.batch_size, args.context_length, device,
            )
            print(f"step {step}: validation loss {val_loss:.4f}")

        # Training step
        x, y = get_batch(train_data, args.batch_size, args.context_length, device)
        optimizer.zero_grad()
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            loss = cross_entropy(model(x), y)
        loss.backward()
        gradient_clipping(model.parameters(), args.grad_clip)
        optimizer.step()
        
        if step % args.log_interval == 0:
            print(f"step {step}: train loss {loss.item():.4f}, lr {lr:.2e}")

        # Periodic checkpoint. Save the number of *completed* steps (step + 1)
        # so that resuming continues from the next unfinished step.
        completed = step + 1
        if completed % args.checkpoint_interval == 0:
            save_checkpoint(
                model, optimizer, completed,
                os.path.join(args.out_dir, f"checkpoint_{completed}.pt"),
            )

    # Final eval + checkpoint
    final_val = evaluate(
        model, val_data, args.eval_iters,
        args.batch_size, args.context_length, device,
    )
    print(f"final: validation loss {final_val:.4f}")
    save_checkpoint(
        model, optimizer, args.max_iters,
        os.path.join(args.out_dir, "checkpoint_final.pt")
    )


if __name__ == "__main__":
    main()




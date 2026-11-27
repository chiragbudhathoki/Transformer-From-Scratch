import argparse
import torch

from cs336_basics.linear import transformer_lm, softmax, load_tokenizer


@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_p: float = 0.9,
    context_length: int = 256,
    device: str = "cuda",
):
    model.eval()
    eos_id = tokenizer.encode("<|endoftext|>")[0]
    ids = tokenizer.encode(prompt)
    x = torch.tensor([ids], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        # Crop context to max_seq_len so RoPE buffers never overflow
        x_cond = x[:, -context_length:]

        with torch.amp.autocast(
            device_type=device, dtype=torch.bfloat16, enabled=(device == "cuda")
        ):
            logits = model(x_cond)[:, -1, :]  # (1, vocab_size)

        if temperature <= 0.0:
            next_id = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            probs = softmax(logits / temperature, dim=-1)

            # Top-p (nucleus) filtering
            sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
            cum_probs = torch.cumsum(sorted_probs, dim=-1)
            remove_mask = (cum_probs - sorted_probs) >= top_p
            sorted_probs = sorted_probs.masked_fill(remove_mask, 0.0)
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

            sampled_pos = torch.multinomial(sorted_probs, num_samples=1)
            next_id = torch.gather(sorted_idx, -1, sampled_pos)

        if next_id.item() == eos_id:
            break

        x = torch.cat([x, next_id], dim=-1)

    return tokenizer.decode(x[0].tolist())


def main():
    parser = argparse.ArgumentParser(description="Generate text from a trained LM.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--vocab", type=str, default="data/vocab.pkl")
    parser.add_argument("--merges", type=str, default="data/merges.pkl")
    parser.add_argument("--prompt", type=str, default="Once upon a time, there was a little")
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--d_ff", type=int, default=1344)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.vocab, args.merges, special_tokens=["<|endoftext|>"])

    model = transformer_lm(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        max_seq_len=args.context_length,
        theta=10000.0,
    ).to(args.device)

    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    print(f"Loaded checkpoint from iteration {ckpt.get('iteration', '?')}\n" + "-" * 50)

    story = generate(
        model,
        tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        context_length=args.context_length,
        device=args.device,
    )
    print(story)


if __name__ == "__main__":
    main()
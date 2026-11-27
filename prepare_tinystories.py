import os
import pickle
import time
import numpy as np
from cs336_basics.linear import run_train_bpe, load_tokenizer

train_txt = "data/TinyStoriesV2-GPT4-train.txt"
valid_txt = "data/TinyStoriesV2-GPT4-valid.txt"
vocab_path = "data/tinystories_vocab.pkl"
merges_path = "data/tinystories_merges.pkl"

t0 = time.time()
print("[1/4] Training 10k BPE tokenizer on TinyStories train...")
vocab, merges = run_train_bpe(train_txt, vocab_size=10000, special_tokens=["<|endoftext|>"])
print(f"      Done in {time.time() - t0:.1f}s! Vocab: {len(vocab)}, Merges: {len(merges)}")

with open(vocab_path, "wb") as f:
    pickle.dump(vocab, f)
with open(merges_path, "wb") as f:
    pickle.dump(merges, f)
print("[2/4] Saved vocab and merges .pkl files.")

tok = load_tokenizer(vocab_path, merges_path, special_tokens=["<|endoftext|>"])

def encode_to_bin(src_path, dst_path, label):
    t_start = time.time()
    buf = []
    total = 0
    with open(src_path, "r", encoding="utf-8") as f_in, open(dst_path, "wb") as f_out:
        for tid in tok.encode_iterable(f_in):
            buf.append(tid)
            if len(buf) >= 1_000_000:
                np.array(buf, dtype=np.uint16).tofile(f_out)
                total += len(buf)
                buf.clear()
                print(f"      [{label}] {total:,} tokens encoded ({time.time() - t_start:.1f}s)...")
        if buf:
            np.array(buf, dtype=np.uint16).tofile(f_out)
            total += len(buf)
    print(f"      [{label}] Finished! Total tokens: {total:,} in {time.time() - t_start:.1f}s")

print("[3/4] Encoding validation set...")
encode_to_bin(valid_txt, "data/tinystories_valid.bin", "valid")

print("[4/4] Encoding training set...")
encode_to_bin(train_txt, "data/tinystories_train.bin", "train")

print("All done! Ready to train.")
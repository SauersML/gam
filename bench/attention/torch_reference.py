"""The reference for attention_speed: PyTorch's causal scaled_dot_product_attention in f32 (TF32 off) at
the same shapes, forward and reverse, timed the same way (one warm call, then the mean over a loop
between synchronizations) and counted the same way (forward 4 w Σ T(T + 1)/2 per query head, reverse
2.5 times that). Grouped queries are given their key-value heads repeated beforehand. One line per shape:
model, sequences, length, forward and reverse seconds, their TFLOP/s."""
import sys, time, torch

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
dev = "cuda"
print(torch.cuda.get_device_name(), "torch", torch.__version__, flush=True)
reps = 20
for name, hq, hk, w, count in (("vpd4l", 6, 6, 128, 32), ("qwen3-0.6b", 16, 8, 128, 8)):
    for length in (256, 512, 1024, 2048):
        q = torch.randn(count, hq, length, w, device=dev, requires_grad=True)
        k = torch.randn(count, hk, length, w, device=dev).repeat_interleave(hq // hk, 1).requires_grad_()
        v = torch.randn(count, hk, length, w, device=dev).repeat_interleave(hq // hk, 1).requires_grad_()
        g = torch.randn(count, hq, length, w, device=dev)
        attend = lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)

        def timed(op):
            op()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(reps):
                op()
            torch.cuda.synchronize()
            return (time.perf_counter() - t0) / reps

        forward = timed(lambda: attend())
        out = attend()
        reverse = timed(lambda: torch.autograd.grad(out, (q, k, v), g, retain_graph=True))
        flops = 4.0 * w * hq * count * length * (length + 1) / 2
        print(f"{name}\t{count}\t{length}\t{forward:.6f}\t{reverse:.6f}\t{flops / forward / 1e12:.2f}\t{2.5 * flops / reverse / 1e12:.2f}", flush=True)
        del q, k, v, g, out

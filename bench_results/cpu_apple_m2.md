Environment: cpu; torch 2.14.1; Python 3.12.13; macOS-27.0.1-arm64-arm-64bit; 4 CPU threads; 2026-10-28.

| workload | params | baseline step | hooks every step | agent every step | agent governed | governor N | GPU memory growth | left after detach |
|---|---|---|---|---|---|---|---|---|
| ResNet-50, 8x3x224x224, SGD+momentum | 25.6M | 614 ms | +12.6% | +11.7% | **-0.8%** | 6 | n/a | n/a |
| Transformer LM (6 layers, d=512, 8 heads, vocab 32k), 4x255 tokens, AdamW | 51.7M | 506 ms | +10.0% | +6.6% | **+1.7%** | 4 | n/a | n/a |
| MLP 1024-4096-4096-4096-1000, batch 512, AdamW | 41.9M | 213 ms | +10.9% | +12.0% | **+2.7%** | 4 | n/a | n/a |

Budget: 5%. Overhead = mean step time vs. interleaved no-monitoring blocks (median of repetitions).

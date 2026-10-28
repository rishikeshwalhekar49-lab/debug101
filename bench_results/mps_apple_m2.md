Environment: Apple GPU (MPS), arm64; torch 2.14.1; Python 3.12.13; macOS-27.0.1-arm64-arm-64bit; 4 CPU threads; 2026-10-28.

| workload | params | baseline step | hooks every step | agent every step | agent governed | governor N | GPU memory growth | left after detach |
|---|---|---|---|---|---|---|---|---|
| ResNet-50, 16x3x224x224, SGD+momentum | 25.6M | 465 ms | +16.7% | +17.0% | **+2.1%** | 3 | 0 B | 0 B |
| Transformer LM (6 layers, d=512, 8 heads, vocab 32k), 8x255 tokens, AdamW | 51.7M | 582 ms | +2.7% | +5.7% | **-7.8%** | 6 | 0 B | 0 B |
| MLP 1024-4096-4096-4096-1000, batch 512, AdamW | 41.9M | 114 ms | +25.1% | +25.3% | **+4.7%** | 9 | 0 B | 0 B |

Budget: 5%. Overhead = mean step time vs. interleaved no-monitoring blocks (median of repetitions).

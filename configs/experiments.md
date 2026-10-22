# Recommended experiment configurations

| Stage | Layers | Width | Batch | Steps | Peak LR | Warmup | Seeds | Estimated purpose |
|---|---:|---:|---:|---:|---:|---:|---|---|
| simple/smoke | 2 | 16 | 8 | 2 | 0.003 | 1 | 7 | CI only |
| simple/educational | 16 | 64 | 64 | 500 | 0.003 | 50 | 7,19,31 | gradient and activation plots |
| simple/depth-sweep | 4,16,32 | 64 | 64 | 250 | 0.001,0.003,0.01 | 50 | 7,19,31 | failure boundaries |
| medium/recall | 8 | 128 | 64 | 1500 | 0.001 | 150 | 7,19,31 | attention learning |
| complex/wikitext2 | 12 | 256 | 16 | 3000 | 0.0003 | 300 | 7,19,31 | LM holdout curves |

**Read this first:** All listed configurations are *proposals*, not verified benchmark outcomes. Run the CPU smoke first; use CUDA for the larger conditions. Level 3 uses GPT-2 token IDs from the official dataset splits. Real GPU memory depends on sequence length, batch size, and the vocab-projection head.

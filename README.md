# ComfyUI_vaceFramepack

A comprehensive **ComfyUI custom node suite** for WanVideo / VACE long-form video generation with advanced context management.

## 🧩 Included Nodes

| Node | Category | Description |
|---|---|---|
| **WanVACE FramePack Sampler 2** | `framepackVACE` | Multi-section sampler using hierarchical latent compression (FramePack algorithm) for arbitrarily long videos. Supports multi-prompt scheduling, MoC / Sparse / Frame context modes, and custom sigma schedules. |
| **WanVideo Context Selector** | `WanVideo/Context` | Pixel-space context selector with CLIP-vision MoC retrieval, text-guided fusion, and Soft-NMS diversity sampling. Drop-in upgrade over simple sliding-window history. |

---

## 🚀 Features

### FramePack Sampler
- **Multi-prompt scheduling** — assign different prompts to different video sections
- **Hierarchical latent compression** — recent history stays high-res, distant history is avg-pooled
- **Multiple context modes** — `moc` (semantic retrieval), `sparse` (exponential backoff), `frame` (contiguous)
- **Built-in benchmarking** — per-section timing and quality reports

### WanVideo Context Selector
- **Hybrid context strategy** — combines a motion buffer (last *M* frames) with semantic memory (top-*K* relevant frames from entire history)
- **Multi-modal retrieval** — steer frame selection with visual similarity *and/or* text prompts via `text_weight`
- **Soft-NMS diversity** — prevents clumping of temporally adjacent frames via `diversity_radius`
- **Smart embedding cache** — only encodes new frames; auto-invalidates on model swap

---

## 📦 Installation

1. Navigate to your ComfyUI `custom_nodes/` directory:
   ```bash
   cd ComfyUI/custom_nodes
   ```
2. Clone the repo:
   ```bash
   git clone https://github.com/icogLabs/ComfyUI_vaceFramepack.git
   ```
3. Install dependencies:
   ```bash
   pip install -r ComfyUI_vaceFramepack/requirements.txt
   ```
4. Restart ComfyUI.

---

## 🔧 Usage

### WanVACE FramePack Sampler 2

Connect your **WanVideo model**, **VAE**, and prompts. Use `multi_prompts` (one prompt per line) for multi-section scheduling. Choose a context `mode`:

| Mode | Strategy |
|---|---|
| `moc` | Semantic retrieval via cosine similarity of latent chunks |
| `sparse` | Exponential backoff — always keeps frame 0 + recent half + log-spaced history |
| `frame` | Contiguous last-N (FramePack default) |
| `all` | Runs all three modes sequentially for comparison |

### WanVideo Context Selector

| Parameter | Default | Description |
|---|---|---|
| `selection_mode` | `contiguous` | `contiguous` = sliding window, `moc` = CLIP-vision retrieval |
| `context_size` | 16 | Total frames to output |
| `contiguous_size` | 4 | Recent frames always kept for motion continuity |
| `text_weight` | 0.0 | 0 = visual only, 1 = text only |
| `diversity_radius` | 16 | Soft-NMS suppression window |
| `similarity_threshold` | 0.0 | Minimum score to include a frame |

> **Tip:** For MoC mode, connect a **CLIP Vision** model. Optionally connect a **Text Encoder** and **prompt** for text-guided retrieval.

---

## 🏗️ Architecture

```
ComfyUI_vaceFramepack/
├── __init__.py                  # Node registration
├── model_utils.py               # Shared model utilities
│
├── nodes/
│   ├── samplers.py              # WanVACE FramePack Sampler 2
│   └── wan_context_node.py      # WanVideo Context Selector
│
├── context/
│   ├── builder.py               # Hierarchical context building & masks
│   ├── strategies.py            # Latent-space strategies (Sparse, MoC, Compressor)
│   ├── cache_manager.py         # CLIP embedding cache (singleton)
│   ├── selector_factory.py      # Pixel strategy factory
│   └── pixel_strategies/        # Pixel-space frame selection
│       ├── base_strategy.py     # Abstract ContextStrategy
│       ├── contiguous.py        # Last-N-frames
│       └── moc.py               # CLIP-vision MoC + text fusion + Soft-NMS
│
├── sampler/                     # FramePack generation loop
├── utils/                       # Benchmarking, prompts, VAE, tensor ops
├── wanvideo/                    # WanVideo model definitions
├── wan/                         # FramePack VACE core
└── ...                          # Additional sub-packages
```

## License

MIT

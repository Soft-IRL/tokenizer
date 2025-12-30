# OFT Motion Projector + Tokenizer Adapter
# This file is a runnable script / notebook-style collection of functions and
# example cells to:
#  - provide a BaseMotionTokenizer API
#  - implement Classic (clustering) and VQ-VAE tokenizers (minimal placeholders)
#  - implement a MotionProjector PyTorch module compatible with OFT-style pipelines
#  - show a minimal fine-tuning loop with LoRA (using peft) to adapt the projector
#
# NOTE:
#  - This is a blueprint. Replace placeholder implementations (e.g., segmentation,
#    VQ-VAE encoder) with your real data / models.
#  - Requires: torch, sklearn, peft, transformers (for LLM wrapper), einops (optional)
#  - To run on CPU for small toy tests; use GPU for real runs.

# ======= Imports =======
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans

# Optional imports for LoRA
try:
    from peft import get_peft_model, LoraConfig, TaskType
    PEFT_AVAILABLE = True
except Exception:
    PEFT_AVAILABLE = False



# ======= VQ-VAE Tokenizer (minimal placeholder) =======
class DummyVQVaeEncoder(nn.Module):
    def __init__(self, input_dim=3, latent_dim=64):
        super().__init__()
        # simple 1D conv encoder placeholder
        self.net = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, latent_dim)
        )
    def forward(self, x):
        # x: (B, L, input_dim) -> reorder
        x = x.permute(0,2,1)
        return self.net(x)  # (B, latent_dim)

class VQVaeMotionTokenizer(BaseMotionTokenizer):
    def __init__(self, encoder, codebook_vectors):
        self.encoder = encoder
        self._codebook = np.asarray(codebook_vectors).astype(np.float32)

    def encode(self, trajectory):
        # segment into fixed length segments for encoder
        seg_len = 16
        ids = []
        for s in range(0, max(1, trajectory.shape[0] - seg_len + 1), seg_len):
            seg = trajectory[s:s+seg_len]
            with torch.no_grad():
                t = torch.from_numpy(seg.astype(np.float32)).unsqueeze(0)  # (1,L,3)
                z = self.encoder(t).cpu().numpy()[0]  # (latent_dim,)
            # nearest neighbor to codebook
            d = np.linalg.norm(self._codebook - z[None,:], axis=1)
            k = int(np.argmin(d))
            ids.append(k)
        return ids

    @property
    def codebook(self):
        return self._codebook

# ======= MotionProjector (OFT-compatible) =======
class MotionProjector(nn.Module):
    """
    Maps codebook vectors (K, D) to VLM embedding space (d_vlm).
    Usage:
      projector = MotionProjector(codebook=np_array_KD, d_vlm=4096)
      embeddings = projector(ids_tensor)  # (T, d_vlm)

    The codebook is stored inside the module for convenience but can be updated.
    """
    def __init__(self, codebook: np.ndarray, d_vlm: int = 4096, hidden_mult: int = 4,
                 codebook_trainable: bool = False):
        super().__init__()
        K, D = codebook.shape
        self.register_buffer('codebook', torch.tensor(codebook))
        if codebook_trainable:
            # make it trainable param
            self.codebook = nn.Parameter(self.codebook)
        hidden = max(D * hidden_mult, 128)
        self.proj = nn.Sequential(
            nn.Linear(D, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_vlm),
            nn.LayerNorm(d_vlm)
        )

    def forward(self, ids: torch.LongTensor):
        # ids: (T,) or (B, T)
        single = False
        if ids.dim() == 1:
            single = True
            ids = ids.unsqueeze(0)
        emb = F.embedding(ids, self.codebook)  # (B, T, D)
        B, T, D = emb.shape
        emb = emb.view(B*T, D)
        out = self.proj(emb)
        out = out.view(B, T, -1)
        if single:
            out = out.squeeze(0)
        return out

    def set_codebook(self, new_codebook: np.ndarray):
        # replace buffer
        self.register_buffer('codebook', torch.tensor(new_codebook.astype(np.float32)))

# ======= LoRA integration helper (for single linear layers) =======
# This is a very small convenience wrapper if peft is unavailable
class TinyLoRA(nn.Module):
    def __init__(self, module: nn.Module, r=8, alpha=16):
        super().__init__()
        # only support linear module
        assert isinstance(module, nn.Linear)
        self.module = module
        in_dim, out_dim = module.in_features, module.out_features
        self.r = r
        self.alpha = alpha
        if r > 0:
            self.A = nn.Parameter(torch.zeros((r, in_dim)))
            self.B = nn.Parameter(torch.zeros((out_dim, r)))
            nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
            nn.init.zeros_(self.B)
            self.scale = alpha / r
        else:
            self.register_parameter('A', None)
            self.register_parameter('B', None)

    def forward(self, x):
        out = self.module(x)
        if self.r > 0:
            delta = (x @ self.A.T) @ self.B.T  # (B, out_dim)
            out = out + self.scale * delta
        return out


# ======= Base Tokenizer API =======
class BaseMotionTokenizer:
    def encode(self, trajectory):
        """Return list[int] token ids for given trajectory (np.ndarray T x Dpoints)
        """
        raise NotImplementedError

    def decode(self, ids):
        """Optional: reconstruct sequence from ids
        """
        raise NotImplementedError

    @property
    def codebook(self):
        """Return numpy array (K, D) of codebook vectors
        """
        raise NotImplementedError

    @property
    def vocab_size(self):
        return int(self.codebook.shape[0])

# ======= Classic (DP + clustering) tokenizer =======
# Placeholder DP segmentation function - replace with your actual implementation

def douglas_peucker_segmentation(trajectory, epsilon=0.01):
    # Very naive segmentation: split into fixed-length chunks as placeholder
    T = trajectory.shape[0]
    seg_len = 16
    segments = []
    for s in range(0, max(1, T - seg_len + 1), seg_len):
        segments.append(trajectory[s:s+seg_len])
    return segments


def compute_handcrafted_features(segment):
    # segment: (L, 3) e.g. x,y,z
    # returns simple features: lin, L, d, ux, uy, uz, cx, cy, cz
    pts = np.asarray(segment)
    diffs = np.diff(pts, axis=0)
    step_lengths = np.linalg.norm(diffs, axis=1)
    L = float(np.sum(step_lengths))
    start = pts[0]
    end = pts[-1]
    d = float(np.linalg.norm(end - start))
    lin = float(d / (L + 1e-12))
    u = np.zeros(3)
    if d > 1e-9:
        u = (end - start) / (d + 1e-12)
    centroid = pts.mean(axis=0)
    feat = np.concatenate(([lin, L, d], u, centroid))
    return feat


class ClassicMotionTokenizer(BaseMotionTokenizer):
    def __init__(self, n_clusters=64, seg_len=16, dp_epsilon=0.01):
        self.n_clusters = n_clusters
        self.seg_len = seg_len
        self.dp_epsilon = dp_epsilon
        self.kmeans = None
        self._codebook = None

    def fit(self, list_of_trajectories):
        feats = []
        for traj in list_of_trajectories:
            segs = douglas_peucker_segmentation(traj, epsilon=self.dp_epsilon)
            for seg in segs:
                feats.append(compute_handcrafted_features(seg))
        F = np.vstack(feats)
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=0)
        self.kmeans.fit(F)
        self._codebook = self.kmeans.cluster_centers_.astype(np.float32)
        return self

    def encode(self, trajectory):
        segs = douglas_peucker_segmentation(trajectory, epsilon=self.dp_epsilon)
        ids = []
        for seg in segs:
            f = compute_handcrafted_features(seg).reshape(1, -1)
            ids.append(int(self.kmeans.predict(f)[0]))
        return ids

    def decode(self, ids):
        # decode as centroids (not a trajectory, just proto-feature)
        return self._codebook[ids]

    @property
    def codebook(self):
        return self._codebook



# TOKENIZER:
# Toy dataset: random trajectories + text
N = 200
trajs = [np.cumsum(np.random.randn(160,3)*0.05, axis=0) for _ in range(N)]
texts = ["pick up object" if i%2==0 else "move forward" for i in range(N)]

# Fit classic tokenizer
classic_tok = ClassicMotionTokenizer(n_clusters=32, seg_len=16)
classic_tok.fit(trajs)
ids0 = classic_tok.encode(trajs[0])
print('example ids classic:', ids0[:10])


# Build projector
codebook = classic_tok.codebook  # (K, D)
projector = MotionProjector(codebook, d_vlm=512)

# Toy LLM embedding stub (simulate frozen VLM)
class DummyVLM(nn.Module):
    def __init__(self, d_model=512):
        super().__init__()
        self.enc = nn.Linear(d_model, d_model)
    def forward(self, x):
        # x: (B, T, d_model)
        return x.mean(dim=1)
vlm = DummyVLM(d_model=512)

# If peft is available, attach LoRA to a dummy linear inside vlm (illustration):
if PEFT_AVAILABLE:
    lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=['enc'], task_type=TaskType.SEQ_2_SEQ_LM)
    vlm = get_peft_model(vlm, lora_config)

# Trainable params: projector.proj + optional LoRA inside vlm
opt = torch.optim.AdamW(list(projector.proj.parameters()) + ([] if not PEFT_AVAILABLE else list(vlm.parameters())), lr=1e-4)

# Minimal training: map motion embeddings to a toy target (e.g., text embedding simulation)
for epoch in range(3):
    total = 0.0
    for i in range(len(trajs)):
        ids = classic_tok.encode(trajs[i])
        ids_t = torch.tensor(ids, dtype=torch.long)
        emb = projector(ids_t)  # (T, d_vlm)
        if emb.dim() == 2:
            emb = emb.unsqueeze(0)
        out = vlm(emb)  # (B, d_vlm)
        # toy target
        target = torch.randn_like(out)
        loss = F.mse_loss(out, target)
        opt.zero_grad(); loss.backward(); opt.step()
        total += loss.item()
    print('epoch', epoch, 'loss', total/len(trajs))

print('Done toy fine-tune')


# prismatic/models/projectors.py

class MotionProjector(nn.Module):
    def __init__(self, input_dim, llm_dim):
        super().__init__()
        # dim input_dim: dimension of codebook vectors
        self.proj = nn.Sequential(
            nn.Linear(input_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim)
        )

    def forward(self, x):
        # x shape: (batch, motion_len, input_dim)
        return self.proj(x)

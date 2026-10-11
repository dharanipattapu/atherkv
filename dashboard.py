import streamlit as st
import torch
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time

from transformers import AutoTokenizer
from aetherkv.core.profiler import EntropyProfiler
from aetherkv.core.quant_kernel import AdaptiveCacheAllocator
from aetherkv.core.offloader import AsyncPCIeOffloader
from aetherkv.core.scheduler import EntropyAwareScheduler
from aetherkv.utils.metrics import AetherKVMetrics

# Page Setup
st.set_page_config(
    page_title="AetherKV - Multi-User Concurrent Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark Glassmorphism Styling Override
st.markdown("""
<style>
    /* Global Page Dark Theme */
    header[data-testid="stHeader"] {
        background-color: #050608 !important;
        background: #050608 !important;
    }
    .stApp {
        background-color: #050608 !important;
        color: #e2e8f0 !important;
    }
    .main .block-container {
        padding-top: 1.5rem !important;
    }

    /* Transparent Dark Glassmorphism Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(10, 14, 20, 0.85) !important;
        backdrop-filter: blur(16px) saturate(180%) !important;
        border-right: 1px solid rgba(0, 255, 128, 0.2) !important;
    }

    /* Input Box Full Dark Glassmorphism Override */
    div[data-testid="stTextArea"] > div {
        background-color: #0d131d !important;
        border: 1px solid rgba(0, 255, 128, 0.4) !important;
        border-radius: 10px !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5) !important;
    }

    div[data-testid="stTextArea"] textarea {
        background-color: #0d131d !important;
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 14px !important;
        font-weight: 500 !important;
        line-height: 1.5 !important;
    }

    div[data-testid="stTextArea"] textarea:focus {
        border-color: #00ff80 !important;
        box-shadow: 0 0 12px rgba(0, 255, 128, 0.5) !important;
    }

    /* Glass Buttons */
    div.stButton > button {
        background: linear-gradient(135deg, rgba(0, 255, 128, 0.2), rgba(0, 136, 255, 0.2)) !important;
        border: 1px solid #00ff80 !important;
        color: #00ff80 !important;
        font-weight: 700 !important;
        border-radius: 10px !important;
        padding: 12px 24px !important;
        box-shadow: 0 0 15px rgba(0, 255, 128, 0.2) !important;
        transition: all 0.3s ease !important;
    }
    div.stButton > button:hover {
        background: linear-gradient(135deg, rgba(0, 255, 128, 0.4), rgba(0, 136, 255, 0.4)) !important;
        color: #ffffff !important;
        box-shadow: 0 0 25px rgba(0, 255, 128, 0.5) !important;
        transform: translateY(-2px);
    }

    /* Glass Telemetry Cards */
    .glass-card {
        background: rgba(13, 19, 28, 0.7);
        border: 1px solid rgba(0, 255, 128, 0.25);
        backdrop-filter: blur(12px);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        margin-bottom: 12px;
    }
    .metric-title {
        color: #94a3b8;
        font-size: 12px;
        font-weight: 500;
        text-transform: uppercase;
    }
    .metric-value {
        color: #00ff80;
        font-size: 28px;
        font-weight: 700;
        text-shadow: 0 0 12px rgba(0, 255, 128, 0.4);
    }

    /* Step Pipeline Cards */
    .step-card {
        background: rgba(16, 24, 38, 0.85);
        border-left: 4px solid #00ff80;
        border-radius: 8px;
        padding: 14px;
        margin: 10px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.4);
    }

    /* Token Badges */
    .token-box {
        display: inline-block;
        padding: 5px 9px;
        margin: 3px;
        border-radius: 6px;
        font-size: 13px;
        font-family: 'JetBrains Mono', monospace;
    }
    .token-sink { background: rgba(0, 255, 128, 0.2); border: 1px solid #00ff80; color: #00ff80; }
    .token-recent { background: rgba(0, 136, 255, 0.2); border: 1px solid #0088ff; color: #0088ff; }
    .token-passive { background: rgba(255, 255, 255, 0.05); border: 1px solid #334155; color: #64748b; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_tokenizer():
    return AutoTokenizer.from_pretrained("gpt2")

tokenizer = load_tokenizer()

st.title("⚡ AetherKV: Multi-User Concurrent Batch Engine")
st.caption("Real-Time Pipeline Walkthrough across Continuous Multi-User Requests & Multi-Head Attention Matrices")

# Transparent Sidebar Controls
st.sidebar.markdown("### ⚙️ Engine Parameters")
batch_size = st.sidebar.slider("Concurrent Batch Size (Users)", min_value=1, max_value=4, value=2)
num_heads = st.sidebar.slider("Attention Heads per Model", min_value=2, max_value=16, value=8)
entropy_threshold = st.sidebar.slider("Entropy Threshold", min_value=1.0, max_value=4.0, value=2.5, step=0.1)
step_delay = st.sidebar.slider("Visual Step Delay (seconds)", min_value=0.2, max_value=2.0, value=0.6, step=0.2)

# Sample Default Prompts
sample_prompts = [
    "The deployment of large language models in production environments faces significant hardware bottlenecks due to key-value cache memory growth during long-context autoregressive decoding.",
    "AetherKV introduces a dynamic entropy-based memory abstraction layer that intercepts raw attention weights during forward pass decoding steps.",
    "Critical sink and recent tokens remain protected in high-precision BF16 format, whereas passive blocks undergo dynamic block-wise INT4 quantization via custom Triton CUDA kernels.",
    "In Needle In A Haystack benchmark evaluations, this multi-tiered architecture achieves approximately 70 percent VRAM footprint reduction while retaining factual accuracy."
]

# User Prompts Layout
st.subheader(f"💬 Concurrent User Prompts (Batch Size = {batch_size})")
prompts = []
cols = st.columns(batch_size)
for i in range(batch_size):
    with cols[i]:
        txt = st.text_area(f"User Request #{i+1}", value=sample_prompts[i % len(sample_prompts)], height=130)
        prompts.append(txt)

run_pipeline = st.button("🚀 Process Batch Requests & Evaluate Memory", use_container_width=True)

if run_pipeline:
    st.markdown("---")
    
    status_box = st.empty()
    progress_bar = st.progress(0)
    
    # STEP 1: TOKENIZATION
    status_box.markdown("""
    <div class="step-card">
        🎯 <b>STEP 1: Tokenization & Sequence Ingestion</b><br>
        <span style="color: #94a3b8;">Processing batch raw strings into LLM sub-token arrays...</span>
    </div>
    """, unsafe_allow_html=True)
    progress_bar.progress(25)
    time.sleep(step_delay)

    batch_tokens = [tokenizer.encode(p) for p in prompts]
    max_len = max(len(t) for t in batch_tokens)
    
    padded_tokens = []
    for t in batch_tokens:
        padded = t + [tokenizer.pad_token_id or 0] * (max_len - len(t))
        padded_tokens.append(padded)
        
    seq_len = max_len

    # STEP 2: ENTROPY PROFILING
    status_box.markdown("""
    <div class="step-card">
        🧠 <b>STEP 2: Attention Weight Interception & Shannon Entropy Profiling (Module 1)</b><br>
        <span style="color: #94a3b8;">Calculating H(X) = -Σ P log2(P) to categorize SINK, RECENT, and PASSIVE tiers...</span>
    </div>
    """, unsafe_allow_html=True)
    progress_bar.progress(50)
    time.sleep(step_delay)

    profiler = EntropyProfiler(sink_size=16, recent_size=32, entropy_threshold=entropy_threshold)
    allocator = AdaptiveCacheAllocator(block_size=128)
    offloader = AsyncPCIeOffloader()
    scheduler = EntropyAwareScheduler(total_vram_mb=24576.0)

    for idx, p in enumerate(prompts):
        scheduler.add_request(f"user_{idx+1}", p, len(batch_tokens[idx]))

    sched_res = scheduler.schedule_step()

    torch.manual_seed(42)
    raw_attn = torch.randn(batch_size, num_heads, seq_len, seq_len).softmax(dim=-1)
    kv_cache = torch.randn(batch_size, num_heads, seq_len, 64, dtype=torch.float32)

    start_time = time.perf_counter()
    entropy, masks = profiler.profile_attention(raw_attn)

    # STEP 3: INT4 QUANTIZATION
    status_box.markdown("""
    <div class="step-card">
        📦 <b>STEP 3: Block-wise INT4 Dynamic Quantization (Module 2)</b><br>
        <span style="color: #94a3b8;">Compressing Passive blocks down to INT4 while preserving Sinks & Recent in BF16...</span>
    </div>
    """, unsafe_allow_html=True)
    progress_bar.progress(75)
    time.sleep(step_delay)

    q_kv, scales = allocator.quantize_passive_blocks(kv_cache, masks)

    # STEP 4: PCIE OFFLOADING
    status_box.markdown("""
    <div class="step-card">
        ⚡ <b>STEP 4: Asynchronous CUDA PCIe Host Streaming (Module 3)</b><br>
        <span style="color: #94a3b8;">Offloading low-entropy compressed blocks to pinned CPU Host RAM...</span>
    </div>
    """, unsafe_allow_html=True)
    progress_bar.progress(100)

    passive_indices = (masks == 2).nonzero(as_tuple=False)
    for idx in passive_indices[:10]:
        offloader.offload_block_async(f"b_{idx[0]}_{idx[1]}", q_kv[idx[0], :, idx[1]], scales[idx[0], :, idx[1]])

    end_time = time.perf_counter()
    latency_ms = (end_time - start_time) * 1000.0

    time.sleep(0.3)
    status_box.success("✅ Execution Pipeline Completed Successfully!")

    # TELEMETRY MATH
    baseline_vram_kb = (kv_cache.element_size() * kv_cache.nelement()) / 1024.0
    passive_count = (masks == 2).sum().item()
    uncompressed_count = (masks != 2).sum().item()
    compressed_vram_kb = ((uncompressed_count * 64 * num_heads * 4) + (passive_count * 64 * num_heads * 1)) / 1024.0
    vram_stats = AetherKVMetrics.calculate_vram_reduction(baseline_vram_kb, compressed_vram_kb)

    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="glass-card"><div class="metric-title">Baseline Batch VRAM</div><div class="metric-value">{baseline_vram_kb:.1f} KB</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="glass-card"><div class="metric-title">AetherKV Batch VRAM</div><div class="metric-value">{compressed_vram_kb:.1f} KB</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="glass-card"><div class="metric-title">Total VRAM Saved</div><div class="metric-value">{vram_stats["reduction_percentage"]}%</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="glass-card"><div class="metric-title">Active Batch Requests</div><div class="metric-value">{sched_res["running_count"]} / {batch_size}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # PER-USER TOKEN CLASSIFICATION
    st.subheader("🔍 Concurrent Batch Token Tiers (Per-User Breakdown)")
    st.caption("🟢 **Sink Tokens (BF16 Protected)** | 🔵 **Recent Window (BF16 Protected)** | ⚪ **Passive Tokens (INT4 Compressed)**")

    for u_idx in range(batch_size):
        u_words = [tokenizer.decode([t]) for t in batch_tokens[u_idx]]
        u_mask = masks[u_idx].cpu().numpy().tolist()

        token_html = ""
        for word, tier in zip(u_words, u_mask):
            clean_word = word.replace("<", "&lt;").replace(">", "&gt;")
            if tier == 0:
                token_html += f'<span class="token-box token-sink">{clean_word}</span>'
            elif tier == 1:
                token_html += f'<span class="token-box token-recent">{clean_word}</span>'
            else:
                token_html += f'<span class="token-box token-passive">{clean_word}</span>'

        st.markdown(f"**User #{u_idx+1} Prompt Pipeline Matrix:**")
        st.markdown(f'<div style="background: rgba(10, 15, 24, 0.85); border: 1px solid rgba(0,255,128,0.25); padding: 14px; border-radius: 10px; margin-bottom: 15px; line-height: 2.2;">{token_html}</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # GRAPHICAL HEATMAP & METRICS
    l_col, r_col = st.columns(2)
    with l_col:
        st.subheader("📊 Batch Token Tiers Heatmap Matrix")
        fig_heat = px.imshow(
            masks.cpu().numpy(),
            labels=dict(x="Token Sequence Index", y="User Request Index", color="Tier"),
            color_continuous_scale=[[0, "#00ff80"], [0.5, "#0088ff"], [1.0, "#1f2937"]],
            aspect="auto"
        )
        fig_heat.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#e2e8f0'), coloraxis_showscale=False)
        st.plotly_chart(fig_heat, use_container_width=True)

    with r_col:
        st.subheader("⚡ Continuous Batching Telemetry")
        st.write(f"• **Attention Heads Evaluated:** `{num_heads} Heads`")
        st.write(f"• **Total Parallel Tokens:** `{seq_len * batch_size} Tokens`")
        st.write(f"• **Parallel Passive Tokens Compressed:** `{passive_count} Tokens`")
        st.write(f"• **Batch Execution Step Time:** `{latency_ms:.2f} ms`")
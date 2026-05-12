import streamlit as st
import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ──────────────────────────────────────────────
# Page config & styling
# ──────────────────────────────────────────────
st.set_page_config(page_title="SpectrumCraft", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .block-container { padding-top: 1rem; }
    .main-title {
        font-size: 2.4rem; font-weight: 700; text-align: center;
        background: linear-gradient(135deg, #f59e42, #e05e3a, #c74bdb);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .sub-title {
        text-align: center; color: #888; font-size: 1rem; margin-top: 0;
    }
    .info-box {
        background: #0e1117; border: 1px solid #333; border-radius: 8px;
        padding: 14px 18px; margin: 10px 0; font-size: 0.9rem; line-height: 1.7;
    }
    .section-label {
        font-size: 0.85rem; font-weight: 600; color: #f59e42;
        letter-spacing: 0.5px; margin-bottom: 4px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">📊 SpectrumCraft</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Understand how frequencies build images</p>', unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────
def to_gray(uploaded_bytes):
    """Load uploaded image as float64 grayscale [0, 255]."""
    img = Image.open(io.BytesIO(uploaded_bytes)).convert("L")
    # Resize to power of 2 for FFT efficiency (max 256 for speed)
    size = min(256, min(img.size))
    # Snap to nearest power of 2
    size = 2 ** int(np.log2(size))
    img = img.resize((size, size), Image.LANCZOS)
    return np.array(img, dtype=np.float64)


def forward_fft(image):
    """Compute 2-D FFT and return shifted spectrum."""
    return np.fft.fftshift(np.fft.fft2(image))


def inverse_fft(shifted_spectrum):
    """Inverse FFT from shifted spectrum → real image normalised to [0, 1]."""
    raw = np.abs(np.fft.ifft2(np.fft.ifftshift(shifted_spectrum)))
    lo, hi = raw.min(), raw.max()
    if hi - lo == 0:
        return np.zeros_like(raw)
    return (raw - lo) / (hi - lo)


def magnitude_spectrum(shifted):
    return np.log1p(np.abs(shifted))


def phase_spectrum(shifted):
    return np.angle(shifted)


def circular_mask(shape, radius_frac):
    """Return a binary mask: 1 inside radius_frac of half-diagonal, 0 outside."""
    rows, cols = shape
    cy, cx = rows // 2, cols // 2
    Y, X = np.ogrid[:rows, :cols]
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    max_r = min(cy, cx)
    return (dist <= radius_frac * max_r).astype(np.float64)


def bandpass_mask(shape, lo, hi):
    rows, cols = shape
    cy, cx = rows // 2, cols // 2
    Y, X = np.ogrid[:rows, :cols]
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    max_r = min(cy, cx)
    return ((dist >= lo * max_r) & (dist <= hi * max_r)).astype(np.float64)


def energy_retained(original_spectrum, mask):
    total = np.sum(np.abs(original_spectrum) ** 2)
    kept = np.sum(np.abs(original_spectrum * mask) ** 2)
    if total == 0:
        return 0.0
    return kept / total * 100


def psnr(original_01, reconstructed_01):
    mse = np.mean((original_01 - reconstructed_01) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * np.log10(1.0 / mse)


def png_kb(arr_01):
    """PNG file size in KB for a [0,1] float image."""
    u8 = (np.clip(arr_01, 0, 1) * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(u8).save(buf, format="PNG")
    return len(buf.getvalue()) / 1024


def plot_gray(data, title="", cmap="gray"):
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(data, cmap=cmap)
    ax.set_title(title, fontsize=9)
    ax.axis("off")
    fig.tight_layout(pad=0.3)
    return fig


def plot_spectrum_with_mask(mag, mask=None, title=""):
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(mag, cmap="inferno")
    if mask is not None:
        overlay = np.zeros((*mask.shape, 4))
        overlay[mask == 0] = [0, 0, 0.3, 0.55]  # dim outside the mask
        ax.imshow(overlay)
        # Draw mask boundary
        from matplotlib.patches import Circle
        rows, cols = mask.shape
        cy, cx = rows // 2, cols // 2
        # Find radius from mask
        r = np.sqrt(mask.sum() / np.pi) if mask.sum() > 0 else 0
        circle = Circle((cx, cy), r, fill=False, edgecolor="#f59e42",
                         linewidth=1.2, linestyle="--")
        ax.add_patch(circle)
    ax.set_title(title, fontsize=9)
    ax.axis("off")
    fig.tight_layout(pad=0.3)
    return fig


# ──────────────────────────────────────────────
# Synthetic test images
# ──────────────────────────────────────────────
def make_test_image(name, N=128):
    if name == "Circle":
        img = np.zeros((N, N))
        Y, X = np.ogrid[:N, :N]
        img[((X - N//2)**2 + (Y - N//2)**2) < (N//5)**2] = 255
    elif name == "Square":
        img = np.zeros((N, N))
        s = N // 5
        img[N//2-s:N//2+s, N//2-s:N//2+s] = 255
    elif name == "Stripes":
        X = np.arange(N)
        img = ((np.sin(2 * np.pi * 4 * X / N) * 0.5 + 0.5) * 255)
        img = np.tile(img, (N, 1))
    elif name == "Diagonal":
        Y, X = np.mgrid[:N, :N]
        img = ((np.sin(2 * np.pi * 5 * (X + Y) / N) * 0.5 + 0.5) * 255)
    elif name == "Cross":
        img = np.zeros((N, N))
        t = N // 16
        img[N//2-t:N//2+t, :] = 255
        img[:, N//2-t:N//2+t] = 255
    elif name == "Checkerboard":
        Y, X = np.mgrid[:N, :N]
        img = ((np.floor(X / (N/8)).astype(int) + np.floor(Y / (N/8)).astype(int)) % 2) * 255.0
    else:
        img = np.random.rand(N, N) * 255
    return img.astype(np.float64)


# ──────────────────────────────────────────────
# Sidebar navigation
# ──────────────────────────────────────────────
page = st.sidebar.radio(
    "Navigate",
    [
        "1 – Wave Synthesis",
        "2 – Frequency Window",
        "3 – Phase vs Magnitude",
        "4 – Spatial Convolution",
        "5 – Frequency Bandpass",
        "6 – How It Works",
    ],
)


# ══════════════════════════════════════════════
# PAGE 1 – WAVE SYNTHESIS
# ══════════════════════════════════════════════
if page == "1 – Wave Synthesis":
    st.header("Wave Synthesis")
    st.markdown("""
    <div class="info-box">
    <strong>Core idea:</strong> every image is a weighted sum of 2-D sinusoidal waves at
    different frequencies, amplitudes, and phases.  Add waves below and watch
    the combined image update.  This is exactly what the inverse Fourier
    Transform does — it sums up all the component waves to reconstruct an image.
    </div>
    """, unsafe_allow_html=True)

    N = 128

    if "waves" not in st.session_state:
        st.session_state.waves = [
            {"fx": 3, "fy": 0, "amp": 1.0, "phase": 0.0},
            {"fx": 0, "fy": 5, "amp": 0.7, "phase": 0.0},
        ]

    # Preset buttons
    presets = {
        "Horizontal Bars": [{"fx": 0, "fy": 6, "amp": 1.0, "phase": 0.0}],
        "Checkerboard-ish": [
            {"fx": 4, "fy": 0, "amp": 1.0, "phase": 0.0},
            {"fx": 0, "fy": 4, "amp": 1.0, "phase": 0.0},
        ],
        "Diagonal": [{"fx": 4, "fy": 4, "amp": 1.0, "phase": 0.0}],
        "Complex Mix": [
            {"fx": 2, "fy": 1, "amp": 1.0, "phase": 0.0},
            {"fx": 5, "fy": 3, "amp": 0.6, "phase": 1.2},
            {"fx": 0, "fy": 8, "amp": 0.3, "phase": 0.5},
            {"fx": 7, "fy": 2, "amp": 0.4, "phase": 2.1},
        ],
    }
    cols_preset = st.columns(len(presets) + 1)
    for i, (name, waves) in enumerate(presets.items()):
        if cols_preset[i].button(name, key=f"preset_{i}"):
            st.session_state.waves = [w.copy() for w in waves]
    if cols_preset[-1].button("➕ Add wave"):
        st.session_state.waves.append({"fx": 1, "fy": 1, "amp": 0.5, "phase": 0.0})

    # Wave controls & individual previews
    wave_cols_left, wave_col_right = st.columns([3, 2])

    with wave_cols_left:
        to_remove = None
        for idx, w in enumerate(st.session_state.waves):
            with st.expander(f"Wave {idx + 1}  (fx={w['fx']}, fy={w['fy']})", expanded=True):
                c1, c2, c3, c4, c5 = st.columns([2, 2, 2, 2, 1])
                w["fx"] = c1.slider("fx", -15, 15, int(w["fx"]), key=f"fx_{idx}")
                w["fy"] = c2.slider("fy", -15, 15, int(w["fy"]), key=f"fy_{idx}")
                w["amp"] = c3.slider("amp", 0.0, 2.0, float(w["amp"]), 0.05, key=f"amp_{idx}")
                w["phase"] = c4.slider("φ", 0.0, 6.28, float(w["phase"]), 0.1, key=f"ph_{idx}")
                if c5.button("✕", key=f"rm_{idx}"):
                    to_remove = idx
        if to_remove is not None:
            st.session_state.waves.pop(to_remove)
            st.rerun()

    # Generate combined image
    combined = np.zeros((N, N))
    Y, X = np.mgrid[:N, :N]
    for w in st.session_state.waves:
        combined += w["amp"] * np.cos(
            2 * np.pi * (w["fx"] * X / N + w["fy"] * Y / N) + w["phase"]
        )

    with wave_col_right:
        st.markdown('<p class="section-label">Combined result</p>', unsafe_allow_html=True)
        fig = plot_gray(combined, f"{len(st.session_state.waves)} wave(s) summed")
        st.pyplot(fig); plt.close(fig)

        # Show individual wave thumbnails
        if len(st.session_state.waves) <= 6:
            thumb_cols = st.columns(min(len(st.session_state.waves), 3))
            for i, w in enumerate(st.session_state.waves):
                single = w["amp"] * np.cos(
                    2 * np.pi * (w["fx"] * X / N + w["fy"] * Y / N) + w["phase"]
                )
                with thumb_cols[i % 3]:
                    fig2 = plot_gray(single, f"Wave {i+1}")
                    st.pyplot(fig2); plt.close(fig2)

    st.info(
        "**Try this:** start with the 'Horizontal Bars' preset (1 wave), then add "
        "waves one by one.  Notice how each new frequency adds detail in a specific "
        "direction.  A real photograph needs thousands of such waves at precisely "
        "tuned amplitudes and phases — that's why random tweaking gives noise, not "
        "meaningful images."
    )


# ══════════════════════════════════════════════
# PAGE 2 – FREQUENCY WINDOW (PROGRESSIVE RECON)
# ══════════════════════════════════════════════
elif page == "2 – Frequency Window":
    st.header("Frequency Window – Progressive Reconstruction")
    st.markdown("""
    <div class="info-box">
    Place a circular window at the center of the magnitude spectrum.  Only
    frequencies inside the window survive → the rest are zeroed out.
    A tiny window keeps low frequencies (blurry shape).  Expand it to add detail.
    </div>
    """, unsafe_allow_html=True)

    source = st.radio("Image source", ["Test pattern", "Upload your own"], horizontal=True)

    if source == "Test pattern":
        pat_name = st.selectbox("Pattern", ["Circle", "Square", "Stripes", "Diagonal", "Cross", "Checkerboard"])
        image = make_test_image(pat_name)
    else:
        uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key="win_upload")
        if uploaded is None:
            st.stop()
        image = to_gray(uploaded.read())

    original_01 = image / 255.0
    spectrum = forward_fft(image)
    mag = magnitude_spectrum(spectrum)

    st.markdown("---")
    radius = st.slider("Window radius (% of max frequency)", 1, 100, 25, 1)
    r_frac = radius / 100.0

    mask = circular_mask(image.shape, r_frac)
    filtered = spectrum * mask
    recon = inverse_fft(filtered)
    eng = energy_retained(spectrum, mask)
    quality = psnr(original_01, recon)

    c1, c2, c3 = st.columns(3)
    with c1:
        fig = plot_gray(original_01, "Original")
        st.pyplot(fig); plt.close(fig)
    with c2:
        fig = plot_spectrum_with_mask(mag, mask, f"Spectrum (window {radius}%)")
        st.pyplot(fig); plt.close(fig)
    with c3:
        fig = plot_gray(recon, f"Reconstructed")
        st.pyplot(fig); plt.close(fig)

    m1, m2, m3 = st.columns(3)
    m1.metric("Window radius", f"{radius}%")
    m2.metric("Energy retained", f"{eng:.1f}%")
    m3.metric("PSNR", f"{quality:.1f} dB" if quality < 200 else "∞")

    # Progressive gallery
    st.markdown("---")
    st.subheader("Progressive Reconstruction Gallery")
    levels = [5, 15, 30, 50, 75, 100]
    prog_cols = st.columns(len(levels))
    for i, lv in enumerate(levels):
        m = circular_mask(image.shape, lv / 100)
        r_img = inverse_fft(spectrum * m)
        with prog_cols[i]:
            fig = plot_gray(r_img, f"{lv}%")
            st.pyplot(fig); plt.close(fig)
            e = energy_retained(spectrum, m)
            st.caption(f"Energy: {e:.0f}%")


# ══════════════════════════════════════════════
# PAGE 3 – PHASE VS MAGNITUDE
# ══════════════════════════════════════════════
elif page == "3 – Phase vs Magnitude":
    st.header("Phase vs Magnitude")
    st.markdown("""
    <div class="info-box">
    The FFT produces two components: <strong>magnitude</strong> (how strong each
    frequency is) and <strong>phase</strong> (where each wave is positioned).
    Most people focus on magnitude, but phase carries the structural
    information — edges, shapes, positions.  This demo proves it.
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Upload two images to swap their phase and magnitude")
    col_up1, col_up2 = st.columns(2)
    with col_up1:
        f1 = st.file_uploader("Image A", type=["jpg", "jpeg", "png"], key="phase_a")
    with col_up2:
        f2 = st.file_uploader("Image B", type=["jpg", "jpeg", "png"], key="phase_b")

    if f1 and f2:
        imgA = to_gray(f1.read())
        imgB = to_gray(f2.read())
        # Make same size
        sz = min(imgA.shape[0], imgB.shape[0])
        imgA = np.array(Image.fromarray(imgA.astype(np.uint8)).resize((sz, sz)), dtype=np.float64)
        imgB = np.array(Image.fromarray(imgB.astype(np.uint8)).resize((sz, sz)), dtype=np.float64)

        specA = forward_fft(imgA)
        specB = forward_fft(imgB)

        magA, phaseA = np.abs(specA), np.angle(specA)
        magB, phaseB = np.abs(specB), np.angle(specB)

        # Swap: magnitude A + phase B,  magnitude B + phase A
        combo1 = magA * np.exp(1j * phaseB)  # mag A, phase B
        combo2 = magB * np.exp(1j * phaseA)  # mag B, phase A

        recon1 = inverse_fft(combo1)
        recon2 = inverse_fft(combo2)

        st.markdown("---")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            fig = plot_gray(imgA / 255, "Image A")
            st.pyplot(fig); plt.close(fig)
        with c2:
            fig = plot_gray(imgB / 255, "Image B")
            st.pyplot(fig); plt.close(fig)
        with c3:
            fig = plot_gray(recon1, "Mag A + Phase B")
            st.pyplot(fig); plt.close(fig)
        with c4:
            fig = plot_gray(recon2, "Mag B + Phase A")
            st.pyplot(fig); plt.close(fig)

        st.success(
            "Notice how the **phase** dominates the structure. 'Mag A + Phase B' looks "
            "like Image B, not A. This proves that phase carries the spatial layout "
            "(edges, shapes), while magnitude mostly carries texture and contrast."
        )
    else:
        # Demo with built-in patterns
        st.info("Upload two images above, or see the demo with test patterns below.")
        imgA = make_test_image("Circle")
        imgB = make_test_image("Checkerboard")
        specA, specB = forward_fft(imgA), forward_fft(imgB)

        combo1 = np.abs(specA) * np.exp(1j * np.angle(specB))
        combo2 = np.abs(specB) * np.exp(1j * np.angle(specA))

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            fig = plot_gray(imgA/255, "Circle"); st.pyplot(fig); plt.close(fig)
        with c2:
            fig = plot_gray(imgB/255, "Checkerboard"); st.pyplot(fig); plt.close(fig)
        with c3:
            fig = plot_gray(inverse_fft(combo1), "Mag Circle\n+ Phase Checker"); st.pyplot(fig); plt.close(fig)
        with c4:
            fig = plot_gray(inverse_fft(combo2), "Mag Checker\n+ Phase Circle"); st.pyplot(fig); plt.close(fig)

        st.success("The result looks like whichever image donated its **phase** — phase carries the structure.")


# ══════════════════════════════════════════════
# PAGE 4 – SPATIAL CONVOLUTION (ACTUAL SPATIAL)
# ══════════════════════════════════════════════
elif page == "4 – Spatial Convolution":
    st.header("Spatial Domain Filtering (Convolution)")
    st.markdown("""
    <div class="info-box">
    Spatial filtering slides a small kernel across the image and computes a
    weighted sum at each pixel.  This is <strong>convolution</strong> — it
    happens in the pixel domain, not the frequency domain.  Common kernels:
    averaging (blur), Gaussian (smooth blur), Sobel (edge detection),
    sharpening.
    </div>
    """, unsafe_allow_html=True)

    source = st.radio("Image source", ["Test pattern", "Upload"], horizontal=True, key="sp_src")
    if source == "Test pattern":
        pat = st.selectbox("Pattern", ["Circle", "Square", "Cross", "Checkerboard"], key="sp_pat")
        image = make_test_image(pat)
    else:
        up = st.file_uploader("Upload", type=["jpg","jpeg","png"], key="sp_up")
        if up is None: st.stop()
        image = to_gray(up.read())

    original_01 = image / 255.0

    kernel_name = st.selectbox("Preset kernel", [
        "Box blur 3×3", "Box blur 5×5", "Gaussian 3×3",
        "Sharpen", "Edge detect (Laplacian)", "Sobel-X", "Sobel-Y",
        "Emboss", "Custom"
    ])

    kernels = {
        "Box blur 3×3": np.ones((3,3)) / 9,
        "Box blur 5×5": np.ones((5,5)) / 25,
        "Gaussian 3×3": np.array([[1,2,1],[2,4,2],[1,2,1]], dtype=float) / 16,
        "Sharpen": np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], dtype=float),
        "Edge detect (Laplacian)": np.array([[0,1,0],[1,-4,1],[0,1,0]], dtype=float),
        "Sobel-X": np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=float),
        "Sobel-Y": np.array([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=float),
        "Emboss": np.array([[-2,-1,0],[-1,1,1],[0,1,2]], dtype=float),
    }

    if kernel_name == "Custom":
        st.write("Enter a 3×3 kernel (editable):")
        import pandas as pd
        default = pd.DataFrame(np.zeros((3,3)))
        edited = st.data_editor(default, num_rows="fixed", key="custom_kern")
        kernel = edited.to_numpy().astype(np.float64)
    else:
        kernel = kernels[kernel_name]

    st.markdown("**Kernel values:**")
    st.dataframe(
        kernel.round(4),
        use_container_width=False,
        hide_index=True,
    )

    # Convolve via FFT (fast, equivalent to spatial convolution)
    padded_kernel = np.zeros_like(image)
    kh, kw = kernel.shape
    padded_kernel[:kh, :kw] = kernel
    # Shift kernel so its center is at (0,0)
    padded_kernel = np.roll(padded_kernel, -(kh // 2), axis=0)
    padded_kernel = np.roll(padded_kernel, -(kw // 2), axis=1)

    F_img = np.fft.fft2(image)
    F_kern = np.fft.fft2(padded_kernel)
    result_raw = np.real(np.fft.ifft2(F_img * F_kern))
    # Normalise for display
    lo, hi = result_raw.min(), result_raw.max()
    if hi - lo == 0:
        result_01 = np.zeros_like(result_raw)
    else:
        result_01 = (result_raw - lo) / (hi - lo)

    c1, c2 = st.columns(2)
    with c1:
        fig = plot_gray(original_01, "Original")
        st.pyplot(fig); plt.close(fig)
        st.caption(f"PNG size: {png_kb(original_01):.1f} KB")
    with c2:
        fig = plot_gray(result_01, f"After {kernel_name}")
        st.pyplot(fig); plt.close(fig)
        st.caption(f"PNG size: {png_kb(result_01):.1f} KB")

    st.markdown("""
    <div class="info-box">
    <strong>Spatial vs Frequency filtering:</strong> spatial convolution slides
    a kernel across pixels.  Frequency filtering multiplies a mask in the
    spectrum.  They are mathematically equivalent (convolution theorem) but
    offer different intuitions.  Use the <em>Frequency Bandpass</em> page to
    compare.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# PAGE 5 – FREQUENCY BANDPASS
# ══════════════════════════════════════════════
elif page == "5 – Frequency Bandpass":
    st.header("Frequency Domain – Bandpass Filter")
    st.markdown("""
    <div class="info-box">
    Keep only frequencies between an inner and outer radius.  Low-pass (outer
    only) keeps broad shapes.  High-pass (inner only) keeps edges.  Bandpass
    keeps a specific band.  The mask is shown overlaid on the spectrum so you
    can see exactly which frequencies survive.
    </div>
    """, unsafe_allow_html=True)

    source = st.radio("Image source", ["Test pattern", "Upload"], horizontal=True, key="bp_src")
    if source == "Test pattern":
        pat = st.selectbox("Pattern", ["Circle", "Square", "Stripes", "Diagonal", "Cross", "Checkerboard"], key="bp_pat")
        image = make_test_image(pat)
    else:
        up = st.file_uploader("Upload", type=["jpg","jpeg","png"], key="bp_up")
        if up is None: st.stop()
        image = to_gray(up.read())

    original_01 = image / 255.0
    spectrum = forward_fft(image)
    mag = magnitude_spectrum(spectrum)

    # Preset buttons write to a non-widget key; the slider reads it as default
    if "bp_lo" not in st.session_state:
        st.session_state.bp_lo = 0
        st.session_state.bp_hi = 100

    def _set_bp(lo_val, hi_val):
        st.session_state.bp_lo = lo_val
        st.session_state.bp_hi = hi_val

    # Quick presets (placed above the slider so they update it on next render)
    st.markdown("**Quick presets:**")
    pc1, pc2, pc3, pc4 = st.columns(4)
    pc1.button("Low-pass (0–30%)",  on_click=_set_bp, args=(0, 30))
    pc2.button("High-pass (30–100%)", on_click=_set_bp, args=(30, 100))
    pc3.button("Mid-band (20–60%)", on_click=_set_bp, args=(20, 60))
    pc4.button("All (0–100%)",      on_click=_set_bp, args=(0, 100))

    bp_values = st.slider(
        "Frequency range (inner → outer, % of max)",
        0, 100,
        (st.session_state.bp_lo, st.session_state.bp_hi),
        1,
    )
    lo, hi = bp_values
    # Keep state in sync when the user drags the slider manually
    st.session_state.bp_lo = lo
    st.session_state.bp_hi = hi

    mask = bandpass_mask(image.shape, lo / 100, hi / 100)
    filtered = spectrum * mask
    recon = inverse_fft(filtered)
    eng = energy_retained(spectrum, mask)
    quality = psnr(original_01, recon)

    c1, c2, c3 = st.columns(3)
    with c1:
        fig = plot_gray(original_01, "Original")
        st.pyplot(fig); plt.close(fig)
    with c2:
        # Show mask overlaid on spectrum
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.imshow(mag, cmap="inferno")
        overlay = np.zeros((*mask.shape, 4))
        overlay[mask == 0] = [0, 0, 0.3, 0.6]
        ax.imshow(overlay)
        ax.set_title(f"Bandpass {lo}%-{hi}%", fontsize=9)
        ax.axis("off")
        fig.tight_layout(pad=0.3)
        st.pyplot(fig); plt.close(fig)
    with c3:
        fig = plot_gray(recon, "Filtered Result")
        st.pyplot(fig); plt.close(fig)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Band", f"{lo}% – {hi}%")
    m2.metric("Energy retained", f"{eng:.1f}%")
    m3.metric("PSNR", f"{quality:.1f} dB" if quality < 200 else "∞")
    m4.metric("PNG size", f"{png_kb(recon):.1f} KB")


# ══════════════════════════════════════════════
# PAGE 6 – HOW IT WORKS
# ══════════════════════════════════════════════
elif page == "6 – How It Works":
    st.header("How It Works")

    st.subheader("1 – What is the Fourier Transform?")
    st.write(
        "Any signal — audio, image, anything — can be decomposed into a sum of "
        "pure sine/cosine waves.  The Fourier Transform tells you *which* "
        "frequencies are present and *how strong* each one is.  For a 2-D "
        "image, the waves run in every direction (horizontal, vertical, "
        "diagonal) at every possible spatial frequency."
    )

    st.subheader("2 – The Magnitude Spectrum")
    st.write(
        "The bright dot at the center = DC component (average brightness).  "
        "Moving outward = higher frequencies = finer detail.  Bright spots far "
        "from center mean that specific frequency is prominent.  Vertical "
        "stripes in the image produce horizontal bright dots in the spectrum."
    )

    st.subheader("3 – Phase Matters More Than You'd Think")
    st.write(
        "The spectrum has two parts: magnitude (strength) and phase (position "
        "of each wave).  Most visual structure lives in the phase.  Swapping "
        "the phase of two images makes the result look like whichever image "
        "donated its phase — see the Phase vs Magnitude page."
    )

    st.subheader("4 – The Window / Filtering Concept")
    st.write(
        "Imagine a circular window at the center of the spectrum.  Keep only "
        "what's inside → low-pass filter → blurry but recognisable.  Keep only "
        "outside → high-pass filter → edges only.  Keep a ring → bandpass → a "
        "specific scale of detail.  This is how JPEG compression works: discard "
        "high frequencies your eye can't distinguish."
    )

    st.subheader("5 – Spatial vs Frequency Filtering")
    st.write(
        "Spatial filtering (convolution) slides a small kernel across pixels.  "
        "Frequency filtering multiplies a mask in the spectrum.  They are "
        "mathematically equivalent (the convolution theorem), just two views "
        "of the same operation.  The original Streamlit code confused these — "
        "it called frequency-domain masking 'spatial filtering'."
    )

    st.subheader("6 – Why You Can't Generate Novel Images This Way")
    st.write(
        "Every possible NxN image is some unique combination of N² frequencies.  "
        "But the space of all combinations is astronomically vast, and the "
        "fraction that produces anything meaningful is essentially zero.  "
        "Randomly tweaking frequencies gives noise.  This is why neural "
        "networks (diffusion models, GANs) exist — they learn which specific "
        "frequency recipes correspond to real-world content."
    )

    st.subheader("7 – The Math (Simplified)")
    st.latex(r"F(u,v) = \sum_{x}\sum_{y} f(x,y)\, e^{-j2\pi(ux/M + vy/N)}")
    st.write(
        "For each frequency (u, v), multiply every pixel by a corresponding "
        "wave and sum up.  The inverse transform sums all the waves back to "
        "rebuild the image."
    )

    st.subheader("8 – Real-World Uses of FFT")
    st.write(
        "JPEG/HEIF compression, image denoising, edge detection, texture "
        "analysis, MRI reconstruction (MRI scanners directly capture frequency "
        "data), radar processing, audio equalisation, and as building blocks "
        "inside modern deep learning architectures."
    )

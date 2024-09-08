import streamlit as st
import numpy as np
from PIL import Image
import io
import pandas as pd
import matplotlib.pyplot as plt

# Custom CSS to reduce subheader font size
st.markdown("""
<style>
    .small-font {
        font-size:16px !important;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Function to create smaller subheaders
def small_subheader(text):
    st.markdown(f'<p class="small-font">{text}</p>', unsafe_allow_html=True)

def process_image(contents):
    image = Image.open(io.BytesIO(contents)).convert('L')
    return np.array(image)

def compute_spectrum(image):
    f = np.fft.fft2(image)
    return np.fft.fftshift(f)

def reconstruct_image(spectrum):
    img_back = np.fft.ifft2(np.fft.ifftshift(spectrum))
    img_back = np.abs(img_back)
    return (img_back - np.min(img_back)) / (np.max(img_back) - np.min(img_back))

def plot_spectrum(spectrum):
    magnitude_spectrum = np.log(np.abs(spectrum) + 1)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(magnitude_spectrum, cmap='gray')
    ax.axis('off')
    return fig

def get_file_size(image):
    with io.BytesIO() as output:
        Image.fromarray((image * 255).astype(np.uint8)).save(output, format="PNG")
        return len(output.getvalue()) / 1024  # Size in KB

def apply_bandpass_filter(spectrum, low_freq, high_freq):
    rows, cols = spectrum.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), np.uint8)
    r_outer = int(high_freq * min(crow, ccol))
    r_inner = int(low_freq * min(crow, ccol))
    center = [crow, ccol]
    x, y = np.ogrid[:rows, :cols]
    mask_area = np.logical_and(((x - center[0])**2 + (y - center[1])**2 >= r_inner**2),
                               ((x - center[0])**2 + (y - center[1])**2 <= r_outer**2))
    mask[mask_area] = 1
    return spectrum * mask

heading_styles = '''
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Bungee+Shade&display=swap');
        @import url('https://fonts.googleapis.com/css2?family=Indie+Flower&display=swap');

        .glowing-heading {
            font-family: 'Poppins', sans-serif;
            font-size: 48px;
            text-align: center;
            animation: glowing 2s infinite;
            color: #FF5733; /* Orange color */
            text-shadow: 2px 2px 4px #333;
        }

        .sub-heading {
            font-family: 'Quicksand', cursive;
            font-size: 32px;
            text-align: center;
            animation: colorChange 4s infinite;
            text-shadow: 1px 1px 2px #333;
            color: #0099CC; /* Blue color */
        }

        @keyframes glowing {
            0% { color: #FF5733; } /* Orange color */
            25% { color: #FFFFFF; } /* White color */
            50% { color: #128807; } /* Green color */
            75% { color: #0000FF; } /* Blue color */
            100% { color: #FF5733; } /* Orange color */
        }

        @keyframes colorChange {
            0% { color: #0099CC; } /* Blue color */
            25% { color: #FF5733; } /* Orange color */
            50% { color: #66FF66; } /* Light Green color */
            75% { color: #FFCC00; } /* Yellow color */
            100% { color: #0099CC; } /* Blue color */
        }
    </style>
'''

st.markdown(heading_styles, unsafe_allow_html=True)
st.markdown(f'<p class="glowing-heading">📊 SpectrumCraft 📊</p>', unsafe_allow_html=True)
st.markdown(f'<p class="sub-heading">Custom Filters & Frequency Tuning</p>', unsafe_allow_html=True)

# Create a sidebar for navigation
page = st.sidebar.selectbox("Choose a page", ["About", "How to Use", "Main Application"])

if page == "About":
    st.write("""
    SpectrumCraft is a powerful web application designed for image processing enthusiasts, 
    researchers, and students interested in exploring the fascinating world of frequency domain 
    manipulation and spatial filtering.

    Key Features:
    1. Upload and process various image formats
    2. Visualize image spectrums using Fourier Transform
    3. Apply custom spatial domain filters
    4. Experiment with frequency domain bandpass filters
    5. Compare original and processed images
    6. Analyze file size changes after applying filters

    Whether you're a beginner looking to understand image processing concepts or an expert 
    seeking a convenient tool for quick experiments, SpectrumCraft offers an intuitive 
    interface to explore the power of image filtering techniques.
    """)

elif page == "How to Use":
    st.header("How to Use SpectrumCraft")
    st.write("Watch the tutorial video below to learn how to use SpectrumCraft:")
    
    # Add your .mov file here
    st.video("reference.mov")
    
    st.write("""
    1. Upload an image using the file uploader
    2. Observe the original image, its spectrum, and reconstructed version
    3. Experiment with spatial domain filters by adjusting the filter matrix
    4. Try frequency domain filtering by setting custom bandpass ranges
    5. Compare the results and file sizes of processed images
    """)

elif page == "Main Application":
    st.header("SpectrumCraft Main Application")
    
    @st.cache_data
    def apply_bandpass_filter_cached(spectrum, low_freq, high_freq):
        filtered_spectrum = apply_bandpass_filter(spectrum, low_freq, high_freq)
        freq_filtered_image = reconstruct_image(filtered_spectrum)
        return freq_filtered_image

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image_bytes = uploaded_file.read()
        original_image = process_image(image_bytes)
        spectrum = compute_spectrum(original_image)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            small_subheader("Original Image")
            st.image(original_image, use_column_width=True)
            st.caption(f"Memory usage: {get_file_size(original_image):.2f} KB")

        with col2:
            small_subheader("Magnitude Spectrum")
            fig = plot_spectrum(spectrum)
            st.pyplot(fig)

        with col3:
            small_subheader("Reconstructed Image")
            reconstructed = reconstruct_image(spectrum)
            st.image(reconstructed, use_column_width=True)
            st.caption(f"Memory usage: {get_file_size(reconstructed):.2f} KB")

        st.info("""
            **Note on Reconstruction:**
            When reconstructing the image using all frequencies, you may notice some noise or slight differences compared to the original image. 
            This is due to:
            1. Numerical precision limitations in computations
            2. Rounding errors in the Fourier transform and inverse transform processes
            3. Potential loss of information in the phase component
            
            These factors can introduce small artifacts or noise in the reconstructed image, even when using all available frequency information.
            """)
        col1, col2 = st.columns(2)

        with col1:
            small_subheader("Spatial Domain Filter")
            filter_size = st.slider("Filter Size", min_value=3, max_value=15, value=7, step=2)
            st.write("Editable Filter Matrix")
            filter_matrix = np.ones((filter_size, filter_size))
            filter_df = pd.DataFrame(filter_matrix)
            edited_filter_df = st.data_editor(filter_df, num_rows="dynamic")
            
            if st.button("Apply Spatial Filter"):
                edited_filter = edited_filter_df.to_numpy()
                full_size_filter = np.zeros_like(original_image, dtype=float)
                center = np.array(full_size_filter.shape) // 2
                start = center - np.array(edited_filter.shape) // 2
                end = start + np.array(edited_filter.shape)
                full_size_filter[start[0]:end[0], start[1]:end[1]] = edited_filter
                
                filtered_spectrum = spectrum * full_size_filter
                spatial_filtered_image = reconstruct_image(filtered_spectrum)
                
                st.session_state.spatial_filtered_image = spatial_filtered_image
                st.session_state.spatial_original_size = get_file_size(original_image)
                st.session_state.spatial_new_size = get_file_size(spatial_filtered_image)

            if 'spatial_filtered_image' in st.session_state:
                st.subheader("Spatially Filtered Image")
                st.image(st.session_state.spatial_filtered_image, use_column_width=True)
                st.write(f"Original size: {st.session_state.spatial_original_size:.2f} KB")
                st.write(f"New size (after spatial filtering): {st.session_state.spatial_new_size:.2f} KB")

        with col2:
            small_subheader("Frequency Domain Filter")
            st.write("Custom Frequency Range (Bandpass)")
            low_freq, high_freq = st.slider("Frequency Range", 0.0, 1.0, (0.0, 1.0), 0.01)
            
            # Apply frequency filter
            freq_filtered_image = apply_bandpass_filter_cached(spectrum, low_freq, high_freq)
            
            st.subheader("Frequency Filtered Image")
            st.image(freq_filtered_image, use_column_width=True)
            st.write(f"Original size: {get_file_size(original_image):.2f} KB")
            st.write(f"New size (after frequency filtering): {get_file_size(freq_filtered_image):.2f} KB")
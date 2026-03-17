"""
Sach-AI — Multimodal Deepfake Detection System
Detects AI-generated or manipulated content across images, video, and audio.
Provides confidence scores, Grad-CAM explainability, spectrogram visualization,
and source/metadata analysis.
"""

from flask import Flask, render_template, request, url_for
import os
import numpy as np
import cv2
import time
import traceback
from PIL import Image as pImage
from PIL.ExifTags import TAGS
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend — must be before pyplot import
import matplotlib.pyplot as plt
from datetime import datetime

# ─── Conditional imports with graceful fallbacks ───
try:
    import tensorflow as tf
    from keras.models import load_model
    import keras.utils as ima
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False
    print("[!] Keras/TensorFlow not available — image & audio detection disabled")

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    print("[!] Librosa not available — audio detection disabled")

try:
    import face_recognition
    import torch
    import torchvision
    from torchvision import transforms
    from torch import nn
    import torchvision.models as models
    from torch.utils.data import DataLoader
    from torch.utils.data.dataset import Dataset
    TORCH_AVAILABLE = True
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[i] PyTorch device: {DEVICE}")
except ImportError:
    TORCH_AVAILABLE = False
    DEVICE = None
    print("[!] PyTorch not available — video detection disabled")


# ═══════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
EXPLAINABILITY_FOLDER = os.path.join('static', 'explainability')
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'bmp', 'webp',
                      'mp4', 'avi', 'mov', 'mkv',
                      'wav', 'mp3', 'flac', 'ogg', 'm4a'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB limit

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(EXPLAINABILITY_FOLDER, exist_ok=True)

# Image detection settings
IMG_WIDTH, IMG_HEIGHT = 224, 224

# Video detection settings
im_size = 112
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
sequence_length = 100

if TORCH_AVAILABLE:
    sm = nn.Softmax(dim=1)
    inv_normalize = transforms.Normalize(
        mean=-1 * np.divide(mean, std),
        std=np.divide([1, 1, 1], std)
    )
    train_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((im_size, im_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])


# ═══════════════════════════════════════════════════════════════
#  VIDEO MODEL ARCHITECTURE (ResNeXt50 + LSTM)
# ═══════════════════════════════════════════════════════════════

if TORCH_AVAILABLE:
    class Model(nn.Module):
        def __init__(self, num_classes, latent_dim=2048, lstm_layers=1,
                     hidden_dim=2048, bidirectional=False):
            super(Model, self).__init__()
            model = models.resnext50_32x4d(pretrained=True)
            self.model = nn.Sequential(*list(model.children())[:-2])
            self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)
            self.relu = nn.LeakyReLU()
            self.dp = nn.Dropout(0.4)
            self.linear1 = nn.Linear(2048, num_classes)
            self.avgpool = nn.AdaptiveAvgPool2d(1)

        def forward(self, x):
            batch_size, seq_length, c, h, w = x.shape
            x = x.view(batch_size * seq_length, c, h, w)
            fmap = self.model(x)
            x = self.avgpool(fmap)
            x = x.view(batch_size, seq_length, 2048)
            x_lstm, _ = self.lstm(x, None)
            return fmap, self.dp(self.linear1(x_lstm[:, -1, :]))

    class validation_dataset(Dataset):
        def __init__(self, video_names, sequence_length, transform=None):
            self.video_names = video_names
            self.transform = transform
            self.count = sequence_length

        def __len__(self):
            return len(self.video_names)

        def __getitem__(self, idx):
            video_path = self.video_names[idx]
            frames = []
            for i, frame in enumerate(self.frame_extract(video_path)):
                faces = face_recognition.face_locations(frame)
                try:
                    top, right, bottom, left = faces[0]
                    frame = frame[top:bottom, left:right, :]
                except:
                    pass
                frames.append(self.transform(frame))
                if len(frames) == self.count:
                    break
            frames = torch.stack(frames)
            frames = frames[:self.count]
            return frames.unsqueeze(0)

        def frame_extract(self, path):
            vidObj = cv2.VideoCapture(path)
            success = 1
            while success:
                success, image = vidObj.read()
                if success:
                    yield image
            vidObj.release()


# ═══════════════════════════════════════════════════════════════
#  MODEL LOADING (at startup — not per-request)
# ═══════════════════════════════════════════════════════════════

image_model = None
audio_model = None
video_model = None


def load_all_models():
    """Load all detection models once at startup."""
    global image_model, audio_model, video_model

    if KERAS_AVAILABLE:
        try:
            image_model = load_model('model/completed_augmented_trained_model.h5', compile=False)
            print("[✓] Image deepfake model loaded successfully")
        except Exception as e:
            print(f"[✗] Image model not loaded: {e}")

        try:
            audio_model = load_model('model/audio_classifier.h5', compile=False)
            print("[✓] Audio deepfake model loaded successfully")
        except Exception as e:
            print(f"[✗] Audio model not loaded: {e}")

    if TORCH_AVAILABLE:
        try:
            video_model = Model(2).to(DEVICE)
            path_to_model = 'model/model_97_acc_100_frames_FF_data.pt'
            video_model.load_state_dict(
                torch.load(path_to_model, map_location=DEVICE)
            )
            video_model.eval()
            print("[✓] Video deepfake model loaded successfully")
        except Exception as e:
            print(f"[✗] Video model not loaded: {e}")
            video_model = None


load_all_models()


# ═══════════════════════════════════════════════════════════════
#  METADATA / SOURCE EXTRACTION
# ═══════════════════════════════════════════════════════════════

def extract_image_metadata(filepath):
    """Extract EXIF and file metadata from an image. Returns (metadata_dict, flags_list)."""
    metadata = {}
    flags = []
    try:
        img = pImage.open(filepath)
        metadata['Format'] = img.format or 'Unknown'
        metadata['Resolution'] = f"{img.size[0]} × {img.size[1]} px"
        metadata['Color Mode'] = img.mode
        metadata['File Size'] = f"{os.path.getsize(filepath) / 1024:.1f} KB"

        exif_data = img._getexif()
        if exif_data:
            interesting_tags = {
                'Make': 'Camera Make',
                'Model': 'Camera Model',
                'Software': 'Software',
                'DateTime': 'Date Taken',
                'ExifImageWidth': 'Original Width',
                'ExifImageHeight': 'Original Height',
                'Flash': 'Flash',
                'FocalLength': 'Focal Length',
                'ISOSpeedRatings': 'ISO',
            }
            for tag_id, value in exif_data.items():
                tag_name = TAGS.get(tag_id, str(tag_id))
                if tag_name in interesting_tags:
                    metadata[interesting_tags[tag_name]] = str(value)

            # Check for editing software
            if 'Software' in metadata:
                sw = metadata['Software'].lower()
                if any(x in sw for x in ['photoshop', 'gimp', 'lightroom',
                                          'stable diffusion', 'midjourney',
                                          'dall-e', 'comfyui']):
                    flags.append({
                        'text': f"Image edited/created with: {metadata['Software']}",
                        'type': 'warning'
                    })
        else:
            flags.append({
                'text': 'No EXIF metadata found — common in AI-generated or heavily processed images',
                'type': 'warning'
            })

        # Check for unusual dimensions (powers of 2 = common in GANs)
        w, h = img.size
        if w == h and w in [256, 512, 1024, 2048]:
            flags.append({
                'text': f'Square {w}×{h} resolution — common in GAN-generated images',
                'type': 'warning'
            })

    except Exception as e:
        flags.append({'text': f'Metadata extraction error: {str(e)}', 'type': 'info'})

    return metadata, flags


def extract_audio_metadata(filepath):
    """Extract audio file metadata. Returns (metadata_dict, flags_list)."""
    metadata = {}
    flags = []
    try:
        metadata['File Size'] = f"{os.path.getsize(filepath) / 1024:.1f} KB"
        metadata['File Type'] = os.path.splitext(filepath)[1].upper().replace('.', '')

        if LIBROSA_AVAILABLE:
            y, sr = librosa.load(filepath, sr=None)
            duration = librosa.get_duration(y=y, sr=sr)
            metadata['Sample Rate'] = f"{sr:,} Hz"
            metadata['Duration'] = f"{duration:.2f} seconds"
            metadata['Total Samples'] = f"{len(y):,}"
            metadata['Channels'] = 'Mono (converted by librosa)'

            # Compute basic audio features for source analysis
            rms = np.sqrt(np.mean(y ** 2))
            metadata['RMS Energy'] = f"{rms:.4f}"
            zcr = np.mean(librosa.feature.zero_crossing_rate(y))
            metadata['Zero-Crossing Rate'] = f"{zcr:.4f}"

            if duration < 1.0:
                flags.append({
                    'text': 'Very short audio clip (< 1s) — may be a spliced segment',
                    'type': 'warning'
                })
            if sr < 8000:
                flags.append({
                    'text': f'Unusually low sample rate ({sr} Hz) — potential quality degradation or re-encoding',
                    'type': 'warning'
                })
            if rms < 0.001:
                flags.append({
                    'text': 'Extremely low audio energy — may contain silence or synthetic padding',
                    'type': 'warning'
                })
    except Exception as e:
        flags.append({'text': f'Audio metadata error: {str(e)}', 'type': 'info'})

    return metadata, flags


def extract_video_metadata(filepath):
    """Extract video file metadata. Returns (metadata_dict, flags_list)."""
    metadata = {}
    flags = []
    try:
        cap = cv2.VideoCapture(filepath)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
            codec = "".join([chr((fourcc_int >> (8 * i)) & 0xFF) for i in range(4)])

            metadata['Resolution'] = f"{w} × {h} px"
            metadata['FPS'] = f"{fps:.1f}"
            metadata['Total Frames'] = f"{frame_count:,}"
            metadata['Duration'] = f"{duration:.2f} seconds"
            metadata['Codec'] = codec
            metadata['File Size'] = f"{os.path.getsize(filepath) / (1024 * 1024):.2f} MB"

            # Count faces in first frame for analysis
            ret, first_frame = cap.read()
            if ret:
                face_locations = face_recognition.face_locations(first_frame) if TORCH_AVAILABLE else []
                metadata['Faces in First Frame'] = str(len(face_locations))

            cap.release()

            if duration < 2.0:
                flags.append({
                    'text': 'Very short video (< 2s) — may be a manipulated clip',
                    'type': 'warning'
                })
            if fps > 60:
                flags.append({
                    'text': f'Unusually high FPS ({fps:.0f}) — uncommon for authentic recordings',
                    'type': 'warning'
                })
        else:
            flags.append({'text': 'Could not open video file', 'type': 'danger'})
    except Exception as e:
        flags.append({'text': f'Video metadata error: {str(e)}', 'type': 'info'})

    return metadata, flags


# ═══════════════════════════════════════════════════════════════
#  EXPLAINABILITY: Grad-CAM for Images
# ═══════════════════════════════════════════════════════════════

def generate_gradcam(model, img_array, original_img_path, save_filename):
    """
    Generate Grad-CAM heatmap overlay for a Keras image classification model.
    Returns the URL path to the saved visualization, or None on failure.
    """
    try:
        # Find the last Conv2D layer
        last_conv_layer_name = None
        for layer in reversed(model.layers):
            if 'conv' in layer.name.lower() and hasattr(layer, 'output'):
                last_conv_layer_name = layer.name
                break

        if last_conv_layer_name is None:
            print("[!] No convolutional layer found for Grad-CAM")
            return None

        # Build gradient model
        grad_model = tf.keras.Model(
            inputs=model.input,
            outputs=[model.get_layer(last_conv_layer_name).output, model.output]
        )

        # Compute gradients
        img_tensor = tf.cast(img_array, tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(img_tensor)
            conv_output, predictions = grad_model(img_tensor)
            # For binary sigmoid output
            if predictions.shape[-1] == 1:
                loss = predictions[0, 0]
            else:
                pred_index = tf.argmax(predictions[0])
                loss = predictions[0, pred_index]

        grads = tape.gradient(loss, conv_output)
        if grads is None:
            print("[!] Grad-CAM: gradients are None")
            return None

        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_output = conv_output[0]
        heatmap = conv_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
        heatmap = heatmap.numpy()

        # ─── Create side-by-side visualization ───
        original = cv2.imread(original_img_path)
        if original is None:
            return None
        original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

        heatmap_resized = cv2.resize(heatmap, (original.shape[1], original.shape[0]))
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        superimposed = cv2.addWeighted(original, 0.6, heatmap_colored, 0.4, 0)
        superimposed_rgb = cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB)

        # Create a 3-panel figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.patch.set_facecolor('#0d1117')

        axes[0].imshow(original_rgb)
        axes[0].set_title('Original Image', color='white', fontsize=12, fontweight='bold')
        axes[0].axis('off')

        axes[1].imshow(heatmap_resized, cmap='jet')
        axes[1].set_title('Attention Heatmap', color='white', fontsize=12, fontweight='bold')
        axes[1].axis('off')

        axes[2].imshow(superimposed_rgb)
        axes[2].set_title('Overlay (Suspicious Regions)', color='white', fontsize=12, fontweight='bold')
        axes[2].axis('off')

        plt.tight_layout()
        save_path = os.path.join(EXPLAINABILITY_FOLDER, save_filename)
        plt.savefig(save_path, bbox_inches='tight', facecolor='#0d1117', dpi=120)
        plt.close(fig)

        return url_for('static', filename=f'explainability/{save_filename}')

    except Exception as e:
        print(f"[!] Grad-CAM error: {traceback.format_exc()}")
        return None


# ═══════════════════════════════════════════════════════════════
#  EXPLAINABILITY: Mel-Spectrogram Visualization for Audio
# ═══════════════════════════════════════════════════════════════

def generate_spectrogram_viz(filepath, save_filename, prediction_label, confidence):
    """
    Generate a mel-spectrogram visualization of the analyzed audio.
    Returns the URL path to the saved image, or None on failure.
    """
    try:
        y, sr = librosa.load(filepath, sr=16000, duration=5)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        fig, axes = plt.subplots(2, 1, figsize=(12, 7), gridspec_kw={'height_ratios': [3, 1]})
        fig.patch.set_facecolor('#0d1117')

        # Top panel: Mel-spectrogram
        img = librosa.display.specshow(
            mel_spec_db, sr=sr, x_axis='time', y_axis='mel',
            ax=axes[0], cmap='magma'
        )
        axes[0].set_title('Mel-Spectrogram Analysis', color='white', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Time (s)', color='#aaa')
        axes[0].set_ylabel('Frequency (Hz)', color='#aaa')
        axes[0].tick_params(colors='#aaa')
        fig.colorbar(img, ax=axes[0], format='%+2.0f dB', label='Power (dB)')

        # Bottom panel: Waveform
        times = np.linspace(0, len(y) / sr, len(y))
        color = '#00e676' if 'Real' in prediction_label or 'Bonafide' in prediction_label else '#ff1744'
        axes[1].plot(times, y, color=color, linewidth=0.5, alpha=0.8)
        axes[1].fill_between(times, y, alpha=0.2, color=color)
        axes[1].set_title(f'Waveform — {prediction_label} ({confidence:.1f}%)',
                          color='white', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Time (s)', color='#aaa')
        axes[1].set_ylabel('Amplitude', color='#aaa')
        axes[1].set_facecolor('#0d1117')
        axes[1].tick_params(colors='#aaa')
        axes[1].set_xlim(0, len(y) / sr)

        plt.tight_layout()
        save_path = os.path.join(EXPLAINABILITY_FOLDER, save_filename)
        plt.savefig(save_path, bbox_inches='tight', facecolor='#0d1117', dpi=120)
        plt.close(fig)

        return url_for('static', filename=f'explainability/{save_filename}')

    except Exception as e:
        print(f"[!] Spectrogram viz error: {traceback.format_exc()}")
        return None


# ═══════════════════════════════════════════════════════════════
#  EXPLAINABILITY: Video Frame Analysis Visualization
# ═══════════════════════════════════════════════════════════════

def generate_video_frame_analysis(filepath, save_filename, prediction_label, confidence):
    """
    Extract key frames from the video and show face detection + analysis.
    Returns the URL path to the saved image, or None on failure.
    """
    try:
        cap = cv2.VideoCapture(filepath)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Sample 6 evenly-spaced frames
        n_samples = min(6, total_frames)
        sample_indices = np.linspace(0, total_frames - 1, n_samples, dtype=int)
        sampled_frames = []
        face_counts = []

        for idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Detect faces and draw boxes
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if TORCH_AVAILABLE:
                    faces = face_recognition.face_locations(rgb_frame)
                    face_counts.append(len(faces))
                    for (top, right, bottom, left) in faces:
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                else:
                    face_counts.append(0)
                sampled_frames.append((idx, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))

        cap.release()

        if not sampled_frames:
            return None

        # Create visualization
        n_cols = min(3, len(sampled_frames))
        n_rows = (len(sampled_frames) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        fig.patch.set_facecolor('#0d1117')

        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
            axes = [axes]
        for row_axes in axes:
            if not isinstance(row_axes, np.ndarray):
                row_axes = [row_axes]
            for ax in row_axes:
                ax.set_facecolor('#0d1117')
                ax.axis('off')

        for i, (frame_idx, frame) in enumerate(sampled_frames):
            r, c = i // n_cols, i % n_cols
            ax = axes[r][c] if isinstance(axes[r], (list, np.ndarray)) else axes[r]
            ax.imshow(frame)
            time_sec = frame_idx / fps if fps > 0 else 0
            ax.set_title(
                f'Frame #{frame_idx} (t={time_sec:.1f}s) — {face_counts[i]} face(s)',
                color='white', fontsize=10, fontweight='bold'
            )

        fig.suptitle(
            f'Video Frame Analysis — {prediction_label} ({confidence:.1f}% confidence)',
            color='white', fontsize=14, fontweight='bold', y=1.02
        )
        plt.tight_layout()
        save_path = os.path.join(EXPLAINABILITY_FOLDER, save_filename)
        plt.savefig(save_path, bbox_inches='tight', facecolor='#0d1117', dpi=100)
        plt.close(fig)

        return url_for('static', filename=f'explainability/{save_filename}')

    except Exception as e:
        print(f"[!] Video frame analysis error: {traceback.format_exc()}")
        return None


# ═══════════════════════════════════════════════════════════════
#  DETECTION: Image Deepfake
# ═══════════════════════════════════════════════════════════════

def detect_deepfake(file_path):
    """
    Detect deepfake in an image file.
    Returns dict with: result, confidence, is_real, explainability_image,
                        explainability_caption, findings, metadata
    """
    findings = []
    metadata, meta_flags = extract_image_metadata(file_path)
    findings.extend(meta_flags)

    if image_model is None:
        return {
            'result': 'Model not loaded — cannot analyze',
            'confidence': 0,
            'is_real': None,
            'explainability_image': None,
            'explainability_caption': '',
            'findings': findings + [{'text': 'Image detection model is not available. '
                                             'Please ensure model/completed_augmented_trained_model.h5 exists.', 'type': 'danger'}],
            'metadata': metadata,
        }

    try:
        # Preprocess
        img = ima.load_img(file_path, target_size=(IMG_WIDTH, IMG_HEIGHT))
        x = ima.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = x / 255.0

        # Predict
        result = image_model.predict(x)
        raw_score = float(result[0][0]) if result.shape[-1] == 1 else float(result[0][0])

        if raw_score < 0.50:
            result_string = "FAKE — AI-Generated or Manipulated Image"
            is_real = False
            confidence = (1 - raw_score) * 100  # Higher = more confident it's fake
        else:
            result_string = "REAL — Authentic Image"
            is_real = True
            confidence = raw_score * 100

        # Generate Grad-CAM explainability
        timestamp = int(time.time())
        gradcam_filename = f"gradcam_{timestamp}.png"
        explainability_url = generate_gradcam(image_model, x, file_path, gradcam_filename)

        # Findings based on model output
        findings.append({
            'text': f'Model raw output score: {raw_score:.4f} (threshold: 0.50)',
            'type': 'info'
        })
        if confidence > 90:
            findings.append({
                'text': f'High confidence ({confidence:.1f}%) — model is very certain about this classification',
                'type': 'success' if is_real else 'danger'
            })
        elif confidence < 60:
            findings.append({
                'text': f'Low confidence ({confidence:.1f}%) — result may be unreliable, manual review recommended',
                'type': 'warning'
            })

        caption = ('Grad-CAM heatmap showing regions the model focused on. '
                   'Red/warm areas indicate high attention — possible manipulation artifacts.')

        return {
            'result': result_string,
            'confidence': round(confidence, 1),
            'is_real': is_real,
            'explainability_image': explainability_url,
            'explainability_caption': caption,
            'findings': findings,
            'metadata': metadata,
        }

    except Exception as e:
        findings.append({'text': f'Detection error: {str(e)}', 'type': 'danger'})
        return {
            'result': 'Error during analysis',
            'confidence': 0,
            'is_real': None,
            'explainability_image': None,
            'explainability_caption': '',
            'findings': findings,
            'metadata': metadata,
        }


# ═══════════════════════════════════════════════════════════════
#  DETECTION: Audio Deepfake
# ═══════════════════════════════════════════════════════════════

def detect_voicefake(filepath):
    """
    Detect deepfake/synthetic audio.
    Returns dict with: result, confidence, is_real, explainability_image,
                        explainability_caption, findings, metadata
    """
    findings = []
    metadata, meta_flags = extract_audio_metadata(filepath)
    findings.extend(meta_flags)

    if audio_model is None:
        return {
            'result': 'Model not loaded — cannot analyze',
            'confidence': 0,
            'is_real': None,
            'explainability_image': None,
            'explainability_caption': '',
            'findings': findings + [{'text': 'Audio detection model is not available. '
                                             'Please ensure model/audio_classifier.h5 exists.', 'type': 'danger'}],
            'metadata': metadata,
        }

    try:
        SAMPLE_RATE = 16000
        N_MELS = 128
        DURATION = 5
        max_time_steps = 109

        audio, _ = librosa.load(filepath, sr=SAMPLE_RATE, duration=DURATION)
        mel_spectro = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_mels=N_MELS)
        mel_spectro = librosa.power_to_db(mel_spectro, ref=np.max)

        if mel_spectro.shape[1] < max_time_steps:
            mel_spectro_padded = np.pad(
                mel_spectro,
                ((0, 0), (0, max_time_steps - mel_spectro.shape[1])),
                mode='constant'
            )
        else:
            mel_spectro_padded = mel_spectro[:, :max_time_steps]

        mel_spec = np.expand_dims(np.array(mel_spectro_padded), axis=0)

        # Predict
        result = audio_model.predict(mel_spec)
        classes = np.argmax(result, axis=1)[0]
        probabilities = result[0]

        if classes == 0:
            result_string = "FAKE — Spoofed / Synthetic Voice Detected"
            is_real = False
            confidence = float(probabilities[0]) * 100
        else:
            result_string = "REAL — Bonafide / Authentic Voice"
            is_real = True
            confidence = float(probabilities[1]) * 100

        # Generate spectrogram visualization
        timestamp = int(time.time())
        spec_filename = f"spectrogram_{timestamp}.png"
        explainability_url = generate_spectrogram_viz(
            filepath, spec_filename, result_string.split('—')[0].strip(), confidence
        )

        # Findings
        findings.append({
            'text': f'Model output probabilities: Fake={probabilities[0]:.4f}, Real={probabilities[1]:.4f}',
            'type': 'info'
        })
        if confidence > 90:
            findings.append({
                'text': f'High confidence ({confidence:.1f}%) — model is very certain',
                'type': 'success' if is_real else 'danger'
            })
        elif confidence < 60:
            findings.append({
                'text': f'Low confidence ({confidence:.1f}%) — result may be unreliable',
                'type': 'warning'
            })

        # Source analysis for audio
        if not is_real:
            findings.append({
                'text': 'Voice may have been generated using TTS (Text-to-Speech) or voice cloning technology',
                'type': 'warning'
            })
            findings.append({
                'text': 'Common deepfake audio sources: Voice cloning APIs, TTS engines, vocoder-based synthesis',
                'type': 'info'
            })

        caption = ('Mel-spectrogram showing the frequency analysis of the audio. '
                   'Top: spectral energy over time. Bottom: waveform amplitude. '
                   'Synthetic audio often shows unnatural smoothness or repetitive patterns.')

        return {
            'result': result_string,
            'confidence': round(confidence, 1),
            'is_real': is_real,
            'explainability_image': explainability_url,
            'explainability_caption': caption,
            'findings': findings,
            'metadata': metadata,
        }

    except Exception as e:
        findings.append({'text': f'Detection error: {str(e)}', 'type': 'danger'})
        return {
            'result': 'Error during analysis',
            'confidence': 0,
            'is_real': None,
            'explainability_image': None,
            'explainability_caption': '',
            'findings': findings,
            'metadata': metadata,
        }


# ═══════════════════════════════════════════════════════════════
#  DETECTION: Video Deepfake
# ═══════════════════════════════════════════════════════════════

def im_convert(tensor):
    """Convert a tensor to a numpy image."""
    image = tensor.to("cpu").clone().detach()
    image = image.squeeze()
    image = inv_normalize(image)
    image = image.numpy()
    image = image.transpose(1, 2, 0)
    image = image.clip(0, 1)
    return image


def predict_video(model, img):
    """Run inference on a video tensor. Returns (prediction_class, confidence)."""
    fmap, logits = model(img.to(DEVICE))
    logits = sm(logits)
    _, prediction = torch.max(logits, 1)
    confidence = logits[:, int(prediction.item())].item() * 100
    return int(prediction.item()), confidence


def detect_video_deepfake(filepath):
    """
    Detect deepfake in a video file.
    Returns dict with: result, confidence, is_real, explainability_image,
                        explainability_caption, findings, metadata
    """
    findings = []
    metadata, meta_flags = extract_video_metadata(filepath)
    findings.extend(meta_flags)

    if video_model is None:
        return {
            'result': 'Model not loaded — cannot analyze',
            'confidence': 0,
            'is_real': None,
            'explainability_image': None,
            'explainability_caption': '',
            'findings': findings + [{'text': 'Video detection model is not available. '
                                             'Please ensure model/model_97_acc_100_frames_FF_data.pt exists.', 'type': 'danger'}],
            'metadata': metadata,
        }

    try:
        video_dataset = validation_dataset(
            [filepath], sequence_length=sequence_length, transform=train_transforms
        )

        prediction_class, confidence = predict_video(video_model, video_dataset[0])

        if prediction_class == 1:
            result_string = "REAL — Authentic Video"
            is_real = True
        else:
            result_string = "FAKE — Deepfake Manipulation Detected"
            is_real = False

        # Generate frame analysis visualization
        timestamp = int(time.time())
        frame_filename = f"video_frames_{timestamp}.png"
        explainability_url = generate_video_frame_analysis(
            filepath, frame_filename,
            result_string.split('—')[0].strip(), confidence
        )

        # Findings
        findings.append({
            'text': f'Analyzed {sequence_length} frames using ResNeXt50 + LSTM temporal model',
            'type': 'info'
        })
        findings.append({
            'text': f'Prediction confidence: {confidence:.1f}%',
            'type': 'info'
        })
        if confidence > 90:
            findings.append({
                'text': f'High confidence ({confidence:.1f}%) — model is very certain',
                'type': 'success' if is_real else 'danger'
            })
        elif confidence < 60:
            findings.append({
                'text': f'Low confidence ({confidence:.1f}%) — result may be unreliable',
                'type': 'warning'
            })

        if not is_real:
            findings.append({
                'text': 'Possible manipulation types: Face swap, face reenactment, or full synthesis',
                'type': 'warning'
            })
            findings.append({
                'text': 'Common deepfake video sources: FaceSwap, DeepFaceLab, First Order Motion Model',
                'type': 'info'
            })

        caption = ('Frame-by-frame analysis with face detection boxes. '
                   'Green boxes highlight detected faces. '
                   'The model analyzes temporal consistency across frames to detect manipulation.')

        return {
            'result': result_string,
            'confidence': round(confidence, 1),
            'is_real': is_real,
            'explainability_image': explainability_url,
            'explainability_caption': caption,
            'findings': findings,
            'metadata': metadata,
        }

    except Exception as e:
        findings.append({'text': f'Detection error: {str(e)}', 'type': 'danger'})
        return {
            'result': 'Error during analysis',
            'confidence': 0,
            'is_real': None,
            'explainability_image': None,
            'explainability_caption': '',
            'findings': findings,
            'metadata': metadata,
        }


# ═══════════════════════════════════════════════════════════════
#  FLASK ROUTES
# ═══════════════════════════════════════════════════════════════

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def home():
    return render_template('Home.html')


@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/image')
def image():
    return render_template('image.html')


@app.route('/audio')
def audio():
    return render_template('audio.html')


@app.route('/video')
def video():
    return render_template('video.html')


@app.route('/upload-image', methods=['POST'])
def upload_image():
    if request.method == 'POST':
        file = request.files.get('file')
        if not file or not allowed_file(file.filename):
            return render_template('result.html',
                                   result='Invalid file format',
                                   confidence=0, is_real=None, option='image',
                                   explainability_image=None, explainability_caption='',
                                   findings=[{'text': 'Please upload a valid image file (JPG, PNG, etc.)', 'type': 'danger'}],
                                   metadata={})

        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)

        analysis = detect_deepfake(filename)
        return render_template('result.html', option='image', **analysis)


@app.route('/upload-audio', methods=['POST'])
def upload_audio():
    if request.method == 'POST':
        file = request.files.get('file')
        if not file or not allowed_file(file.filename):
            return render_template('result.html',
                                   result='Invalid file format',
                                   confidence=0, is_real=None, option='audio',
                                   explainability_image=None, explainability_caption='',
                                   findings=[{'text': 'Please upload a valid audio file (WAV, MP3, etc.)', 'type': 'danger'}],
                                   metadata={})

        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)

        analysis = detect_voicefake(filename)
        return render_template('result.html', option='audio', **analysis)


@app.route('/upload-video', methods=['POST'])
def upload_video():
    if request.method == 'POST':
        file = request.files.get('file')
        if not file or not allowed_file(file.filename):
            return render_template('result.html',
                                   result='Invalid file format',
                                   confidence=0, is_real=None, option='video',
                                   explainability_image=None, explainability_caption='',
                                   findings=[{'text': 'Please upload a valid video file (MP4, AVI, etc.)', 'type': 'danger'}],
                                   metadata={})

        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)

        analysis = detect_video_deepfake(filename)
        return render_template('result.html', option='video', **analysis)


# ═══════════════════════════════════════════════════════════════
#  START SERVER
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("   Sach-AI — Deepfake Detection System")
    print("   Beyond the mask, lies the real.")
    print("=" * 60)
    print(f"   Image Model: {'✓ Loaded' if image_model else '✗ Not loaded'}")
    print(f"   Audio Model: {'✓ Loaded' if audio_model else '✗ Not loaded'}")
    print(f"   Video Model: {'✓ Loaded' if video_model else '✗ Not loaded'}")
    print("=" * 60 + "\n")
    app.run(debug=True)

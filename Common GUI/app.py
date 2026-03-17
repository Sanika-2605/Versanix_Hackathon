"""
Sach-AI — Multimodal Deepfake Detection System
Detects AI-generated or manipulated content across images, video, and audio.
Provides confidence scores, Grad-CAM explainability, spectrogram visualization,
metadata/source analysis, and JSON API endpoints.

When trained models are not available, falls back to heuristic-based analysis
(Error Level Analysis, noise analysis, spectral analysis).
"""

from flask import Flask, render_template, request, url_for, jsonify
import os
import numpy as np
import cv2
import time
import traceback
from PIL import Image as pImage
from PIL.ExifTags import TAGS
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from scipy import stats as sp_stats

# ─── Conditional imports ───
try:
    import tensorflow as tf
    from keras.models import load_model
    import keras.utils as ima
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False
    print("[!] Keras/TensorFlow not available")

try:
    import librosa
    import librosa.display
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    print("[!] Librosa not available")

try:
    import face_recognition
    FACE_RECOG_AVAILABLE = True
except ImportError:
    FACE_RECOG_AVAILABLE = False
    print("[!] face_recognition not available — video face detection disabled")

try:
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
    print("[!] PyTorch not available")


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
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(EXPLAINABILITY_FOLDER, exist_ok=True)

IMG_WIDTH, IMG_HEIGHT = 224, 224
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
#  VIDEO MODEL ARCHITECTURE
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
                if FACE_RECOG_AVAILABLE:
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
#  MODEL LOADING
# ═══════════════════════════════════════════════════════════════

image_model = None
audio_model = None
video_model = None


def load_all_models():
    global image_model, audio_model, video_model

    if KERAS_AVAILABLE:
        try:
            image_model = load_model('model/completed_augmented_trained_model.h5', compile=False)
            print("[✓] Image model loaded")
        except Exception as e:
            print(f"[✗] Image model not loaded: {e}")
        try:
            audio_model = load_model('model/audio_classifier.h5', compile=False)
            print("[✓] Audio model loaded")
        except Exception as e:
            print(f"[✗] Audio model not loaded: {e}")

    if TORCH_AVAILABLE:
        try:
            video_model = Model(2).to(DEVICE)
            video_model.load_state_dict(
                torch.load('model/model_97_acc_100_frames_FF_data.pt', map_location=DEVICE)
            )
            video_model.eval()
            print("[✓] Video model loaded")
        except Exception as e:
            print(f"[✗] Video model not loaded: {e}")
            video_model = None


load_all_models()


# ═══════════════════════════════════════════════════════════════
#  HEURISTIC DETECTION — No model needed
# ═══════════════════════════════════════════════════════════════

def error_level_analysis(filepath, quality=90):
    """
    Error Level Analysis (ELA) for images.
    Resaves at a known quality and compares — manipulated regions
    show higher error because they were saved at a different quality.
    Returns (ela_image, mean_error, max_error).
    """
    original = cv2.imread(filepath)
    if original is None:
        return None, 0, 0

    # Resave at known quality
    temp_path = os.path.join(UPLOAD_FOLDER, '_ela_temp.jpg')
    cv2.imwrite(temp_path, original, [cv2.IMWRITE_JPEG_QUALITY, quality])
    resaved = cv2.imread(temp_path)

    # Compute difference
    ela = cv2.absdiff(original, resaved)
    ela = ela * 10  # Amplify differences for visibility

    mean_error = float(np.mean(ela))
    max_error = float(np.max(ela))

    try:
        os.remove(temp_path)
    except:
        pass

    return ela, mean_error, max_error


def analyze_noise_patterns(filepath):
    """
    Analyze noise patterns in an image. AI-generated images tend to
    have more uniform noise; real photos have sensor-specific noise.
    Returns dict of noise metrics.
    """
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return {}

    # Laplacian variance (focus sharpness)
    laplacian_var = float(cv2.Laplacian(img, cv2.CV_64F).var())

    # Local noise estimation using median filter
    median = cv2.medianBlur(img, 5)
    noise = cv2.absdiff(img, median)
    noise_mean = float(np.mean(noise))
    noise_std = float(np.std(noise))

    # Frequency domain analysis
    f = np.fft.fft2(img.astype(float))
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    spectral_centroid = float(np.mean(magnitude))

    return {
        'laplacian_variance': round(laplacian_var, 2),
        'noise_mean': round(noise_mean, 4),
        'noise_std': round(noise_std, 4),
        'spectral_centroid': round(spectral_centroid, 2),
    }


def heuristic_image_detection(filepath):
    """
    Heuristic-based image deepfake detection using ELA + noise analysis.
    Works without any trained model.
    """
    findings = []
    metadata, meta_flags = extract_image_metadata(filepath)
    findings.extend(meta_flags)

    # 1. Error Level Analysis
    ela_img, mean_error, max_error = error_level_analysis(filepath)

    # 2. Noise analysis
    noise_metrics = analyze_noise_patterns(filepath)

    # 3. Scoring heuristic
    score = 50.0  # Start neutral
    reasons = []

    # ELA scoring
    if mean_error > 15:
        score -= 20
        reasons.append("High ELA error — regions may have been re-compressed at different qualities")
    elif mean_error > 8:
        score -= 10
        reasons.append("Moderate ELA inconsistency detected")
    else:
        score += 10
        reasons.append("ELA levels appear consistent across the image")

    # Noise uniformity scoring
    if noise_metrics.get('noise_std', 0) < 3:
        score -= 15
        reasons.append("Very uniform noise pattern — common in AI-generated images")
    elif noise_metrics.get('noise_std', 0) > 12:
        score += 10
        reasons.append("Natural noise variation consistent with camera sensor")

    # Laplacian (sharpness) scoring
    lap_var = noise_metrics.get('laplacian_variance', 0)
    if lap_var < 50:
        score -= 5
        reasons.append("Low frequency detail — image may be synthetically smoothed")
    elif lap_var > 1500:
        score += 5
        reasons.append("Rich detail and texture variation — consistent with real photography")

    # Meta-based scoring
    has_exif = any('EXIF' not in f.get('text', '') for f in meta_flags if f.get('type') == 'warning')
    no_exif_flag = any('No EXIF' in f.get('text', '') for f in meta_flags)
    if no_exif_flag:
        score -= 10

    # Clamp score
    score = max(5, min(95, score))
    is_real = score >= 50
    confidence = score if is_real else (100 - score)

    for r in reasons:
        findings.append({'text': r, 'type': 'info' if is_real else 'warning'})

    findings.append({
        'text': f'Heuristic analysis score: {score:.1f}/100 (ELA + noise + metadata)',
        'type': 'info'
    })
    findings.append({
        'text': '⚠ Using heuristic analysis (no trained model loaded). '
                'Results are indicative — load a trained model for higher accuracy.',
        'type': 'warning'
    })

    # Generate ELA visualization
    explainability_url = None
    if ela_img is not None:
        explainability_url = _save_ela_visualization(filepath, ela_img, noise_metrics)

    result_string = "REAL — Image appears authentic (heuristic)" if is_real \
        else "FAKE — Potential manipulation detected (heuristic)"

    return {
        'result': result_string,
        'confidence': round(confidence, 1),
        'is_real': is_real,
        'explainability_image': explainability_url,
        'explainability_caption': (
            'Error Level Analysis (ELA): Differences in compression levels reveal '
            'tampered regions. Bright areas in the ELA image indicate potential manipulation. '
            'Noise analysis shows the distribution of sensor noise across the image.'
        ),
        'findings': findings,
        'metadata': metadata,
    }


def _save_ela_visualization(filepath, ela_img, noise_metrics):
    """Save ELA + noise analysis visualization."""
    try:
        original = cv2.imread(filepath)
        original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        ela_rgb = cv2.cvtColor(ela_img, cv2.COLOR_BGR2RGB)

        # Noise map
        gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        median = cv2.medianBlur(gray, 5)
        noise_map = cv2.absdiff(gray, median)

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.patch.set_facecolor('#0d1117')

        axes[0].imshow(original_rgb)
        axes[0].set_title('Original Image', color='white', fontsize=12, fontweight='bold')
        axes[0].axis('off')

        axes[1].imshow(ela_rgb)
        axes[1].set_title('Error Level Analysis', color='white', fontsize=12, fontweight='bold')
        axes[1].axis('off')

        im = axes[2].imshow(noise_map, cmap='hot')
        axes[2].set_title(f'Noise Map (σ={noise_metrics.get("noise_std", 0):.2f})',
                          color='white', fontsize=12, fontweight='bold')
        axes[2].axis('off')

        plt.tight_layout()
        filename = f"ela_{int(time.time())}.png"
        save_path = os.path.join(EXPLAINABILITY_FOLDER, filename)
        plt.savefig(save_path, bbox_inches='tight', facecolor='#0d1117', dpi=120)
        plt.close(fig)
        return url_for('static', filename=f'explainability/{filename}')
    except Exception as e:
        print(f"[!] ELA viz error: {e}")
        return None


def heuristic_audio_detection(filepath):
    """
    Heuristic-based audio deepfake detection using spectral analysis.
    Works without any trained model.
    """
    findings = []
    metadata, meta_flags = extract_audio_metadata(filepath)
    findings.extend(meta_flags)

    if not LIBROSA_AVAILABLE:
        return {
            'result': 'Cannot analyze — librosa not installed',
            'confidence': 0, 'is_real': None,
            'explainability_image': None, 'explainability_caption': '',
            'findings': findings + [{'text': 'Install librosa to enable audio analysis', 'type': 'danger'}],
            'metadata': metadata,
        }

    try:
        y, sr = librosa.load(filepath, sr=16000, duration=10)
        duration = len(y) / sr

        # Feature extraction
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85))
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        rms = np.mean(librosa.feature.rms(y=y))
        spectral_flatness = np.mean(librosa.feature.spectral_flatness(y=y))
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_var = np.mean(np.var(mfccs, axis=1))

        # Scoring heuristic
        score = 50.0
        reasons = []

        # Spectral flatness — synthetic audio tends to be more spectrally flat
        if spectral_flatness > 0.15:
            score -= 15
            reasons.append(f"High spectral flatness ({spectral_flatness:.3f}) — characteristic of synthetic audio")
        elif spectral_flatness < 0.02:
            score += 10
            reasons.append(f"Low spectral flatness ({spectral_flatness:.3f}) — natural harmonic content")

        # MFCC variance — real speech has more MFCC variation
        if mfcc_var < 50:
            score -= 10
            reasons.append(f"Low MFCC variance ({mfcc_var:.1f}) — limited tonal variation, possible TTS")
        elif mfcc_var > 200:
            score += 10
            reasons.append(f"High MFCC variance ({mfcc_var:.1f}) — rich natural speech characteristics")

        # Zero crossing rate
        if zcr < 0.03:
            score -= 5
            reasons.append("Very low zero-crossing rate — unnaturally smooth signal")
        elif zcr > 0.15:
            score += 5
            reasons.append("Natural zero-crossing rate consistent with real speech")

        # RMS energy variation
        rms_values = librosa.feature.rms(y=y)[0]
        rms_variability = float(np.std(rms_values) / (np.mean(rms_values) + 1e-8))
        if rms_variability < 0.3:
            score -= 10
            reasons.append("Very consistent energy levels — unnatural for human speech")
        elif rms_variability > 0.8:
            score += 5
            reasons.append("Natural energy variation consistent with real speech")

        # Bandwidth
        if spectral_bandwidth < 1000:
            score -= 5
            reasons.append("Narrow spectral bandwidth — possible voice compression or synthesis")

        score = max(5, min(95, score))
        is_real = score >= 50
        confidence = score if is_real else (100 - score)

        for r in reasons:
            findings.append({'text': r, 'type': 'info' if is_real else 'warning'})

        findings.append({
            'text': f'Heuristic analysis score: {score:.1f}/100 (spectral + MFCC + energy)',
            'type': 'info'
        })
        findings.append({
            'text': '⚠ Using heuristic analysis (no trained model loaded).',
            'type': 'warning'
        })

        # Source indicators
        if not is_real:
            findings.append({
                'text': 'Possible sources: TTS engines, voice cloning APIs, vocoder-based synthesis',
                'type': 'info'
            })

        # Generate spectrogram visualization
        explainability_url = _save_audio_heuristic_viz(y, sr, score, is_real, {
            'spectral_centroid': spectral_centroid,
            'spectral_flatness': spectral_flatness,
            'mfcc_var': mfcc_var,
            'zcr': zcr,
        })

        result_string = "REAL — Voice appears authentic (heuristic)" if is_real \
            else "FAKE — Possible synthetic/cloned voice (heuristic)"

        return {
            'result': result_string,
            'confidence': round(confidence, 1),
            'is_real': is_real,
            'explainability_image': explainability_url,
            'explainability_caption': (
                'Top: Mel-spectrogram showing frequency energy over time. '
                'Middle: Waveform. Bottom: MFCC coefficients — the primary features '
                'used for voice analysis. Synthetic audio often shows unnatural smoothness.'
            ),
            'findings': findings,
            'metadata': metadata,
        }

    except Exception as e:
        findings.append({'text': f'Analysis error: {str(e)}', 'type': 'danger'})
        return {
            'result': 'Error during analysis',
            'confidence': 0, 'is_real': None,
            'explainability_image': None, 'explainability_caption': '',
            'findings': findings, 'metadata': metadata,
        }


def _save_audio_heuristic_viz(y, sr, score, is_real, features):
    """Save audio spectral analysis visualization."""
    try:
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

        fig, axes = plt.subplots(3, 1, figsize=(14, 10),
                                  gridspec_kw={'height_ratios': [3, 1.5, 2]})
        fig.patch.set_facecolor('#0d1117')

        # Mel spectrogram
        librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel',
                                  ax=axes[0], cmap='magma')
        axes[0].set_title('Mel-Spectrogram', color='white', fontsize=13, fontweight='bold')
        axes[0].set_xlabel('')
        axes[0].set_ylabel('Frequency (Hz)', color='#aaa')
        axes[0].tick_params(colors='#aaa')

        # Waveform
        color = '#00e676' if is_real else '#ff1744'
        times = np.linspace(0, len(y) / sr, len(y))
        axes[1].plot(times, y, color=color, linewidth=0.4, alpha=0.8)
        axes[1].fill_between(times, y, alpha=0.15, color=color)
        verdict = 'REAL' if is_real else 'FAKE'
        axes[1].set_title(f'Waveform — Score: {score:.0f}/100 ({verdict})',
                          color='white', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Amplitude', color='#aaa')
        axes[1].set_facecolor('#0d1117')
        axes[1].tick_params(colors='#aaa')
        axes[1].set_xlim(0, len(y) / sr)

        # MFCCs
        librosa.display.specshow(mfccs, sr=sr, x_axis='time', ax=axes[2], cmap='coolwarm')
        axes[2].set_title(f'MFCC Coefficients (variance={features["mfcc_var"]:.1f})',
                          color='white', fontsize=12, fontweight='bold')
        axes[2].set_ylabel('MFCC', color='#aaa')
        axes[2].set_xlabel('Time (s)', color='#aaa')
        axes[2].tick_params(colors='#aaa')

        plt.tight_layout()
        filename = f"audio_analysis_{int(time.time())}.png"
        save_path = os.path.join(EXPLAINABILITY_FOLDER, filename)
        plt.savefig(save_path, bbox_inches='tight', facecolor='#0d1117', dpi=120)
        plt.close(fig)
        return url_for('static', filename=f'explainability/{filename}')
    except Exception as e:
        print(f"[!] Audio viz error: {e}")
        return None


def heuristic_video_detection(filepath):
    """
    Heuristic-based video deepfake detection using frame analysis.
    """
    findings = []
    metadata, meta_flags = extract_video_metadata(filepath)
    findings.extend(meta_flags)

    try:
        cap = cv2.VideoCapture(filepath)
        if not cap.isOpened():
            return {
                'result': 'Error — could not open video',
                'confidence': 0, 'is_real': None,
                'explainability_image': None, 'explainability_caption': '',
                'findings': [{'text': 'Failed to open video file', 'type': 'danger'}],
                'metadata': metadata,
            }

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Sample frames for analysis
        n_samples = min(12, total_frames)
        sample_indices = np.linspace(0, total_frames - 1, n_samples, dtype=int)

        face_counts = []
        frame_brightnesses = []
        frame_blurs = []
        sampled_frames = []
        ela_scores = []

        for idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Face detection
            if FACE_RECOG_AVAILABLE:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                faces = face_recognition.face_locations(rgb)
                face_counts.append(len(faces))
                for (top, right, bottom, left) in faces:
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            else:
                # Use Haar cascade as fallback
                face_cascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                )
                faces_haar = face_cascade.detectMultiScale(gray, 1.3, 5)
                face_counts.append(len(faces_haar))
                for (x, y, w, h) in faces_haar:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            frame_brightnesses.append(float(np.mean(gray)))
            frame_blurs.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))

            # Mini ELA per frame
            temp_path = os.path.join(UPLOAD_FOLDER, '_frame_ela.jpg')
            cv2.imwrite(temp_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            resaved = cv2.imread(temp_path)
            if resaved is not None:
                ela_diff = float(np.mean(cv2.absdiff(frame, resaved)))
                ela_scores.append(ela_diff)

            sampled_frames.append((idx, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))

        cap.release()
        try:
            os.remove(os.path.join(UPLOAD_FOLDER, '_frame_ela.jpg'))
        except:
            pass

        # Scoring
        score = 50.0
        reasons = []

        # Face consistency
        if len(face_counts) > 2:
            face_consistency = np.std(face_counts)
            if face_consistency > 0.5:
                score -= 15
                reasons.append(f"Inconsistent face detection ({face_consistency:.2f} std) — faces appear/disappear")
            else:
                score += 10
                reasons.append("Consistent face detection across frames")

        # Brightness consistency
        if len(frame_brightnesses) > 2:
            brightness_std = np.std(frame_brightnesses)
            if brightness_std > 30:
                score -= 10
                reasons.append(f"High brightness variation ({brightness_std:.1f}) — possible splicing")
            else:
                score += 5
                reasons.append("Consistent lighting across frames")

        # Blur consistency
        if len(frame_blurs) > 2:
            blur_std = np.std(frame_blurs)
            blur_mean = np.mean(frame_blurs)
            if blur_std / (blur_mean + 1e-8) > 0.5:
                score -= 10
                reasons.append("Inconsistent sharpness across frames — possible manipulation")

        # ELA consistency
        if len(ela_scores) > 2:
            ela_mean = np.mean(ela_scores)
            if ela_mean > 10:
                score -= 10
                reasons.append(f"High video ELA score ({ela_mean:.1f}) — re-compression artifacts detected")
            else:
                score += 5
                reasons.append("Low ELA variance — consistent compression")

        score = max(5, min(95, score))
        is_real = score >= 50
        confidence = score if is_real else (100 - score)

        for r in reasons:
            findings.append({'text': r, 'type': 'info' if is_real else 'warning'})

        findings.append({
            'text': f'Analyzed {len(sampled_frames)} frames from {total_frames} total',
            'type': 'info'
        })
        findings.append({
            'text': f'Heuristic score: {score:.1f}/100 (face consistency + ELA + temporal)',
            'type': 'info'
        })
        findings.append({
            'text': '⚠ Using heuristic analysis (no trained model loaded).',
            'type': 'warning'
        })

        if not is_real:
            findings.append({
                'text': 'Possible sources: FaceSwap, DeepFaceLab, First Order Motion Model',
                'type': 'info'
            })

        # Visualization
        explainability_url = _save_video_heuristic_viz(sampled_frames, fps, score, is_real,
                                                        face_counts, frame_blurs)

        result_string = "REAL — Video appears authentic (heuristic)" if is_real \
            else "FAKE — Potential deepfake detected (heuristic)"

        return {
            'result': result_string,
            'confidence': round(confidence, 1),
            'is_real': is_real,
            'explainability_image': explainability_url,
            'explainability_caption': (
                'Sampled frames with face detection overlays and temporal analysis. '
                'Green boxes indicate detected faces. Inconsistencies in face detection, '
                'brightness, or sharpness across frames may indicate manipulation.'
            ),
            'findings': findings,
            'metadata': metadata,
        }

    except Exception as e:
        findings.append({'text': f'Analysis error: {str(e)}', 'type': 'danger'})
        return {
            'result': 'Error during analysis',
            'confidence': 0, 'is_real': None,
            'explainability_image': None, 'explainability_caption': '',
            'findings': findings, 'metadata': metadata,
        }


def _save_video_heuristic_viz(sampled_frames, fps, score, is_real,
                                face_counts, frame_blurs):
    """Save video frame analysis visualization."""
    try:
        n_show = min(6, len(sampled_frames))
        step = max(1, len(sampled_frames) // n_show)
        frames_to_show = sampled_frames[::step][:n_show]

        n_cols = min(3, len(frames_to_show))
        n_rows = max(1, (len(frames_to_show) + n_cols - 1) // n_cols)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        fig.patch.set_facecolor('#0d1117')

        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = np.array([axes])
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)

        for ax_row in axes:
            for ax in ax_row:
                ax.set_facecolor('#0d1117')
                ax.axis('off')

        for i, (frame_idx, frame) in enumerate(frames_to_show):
            r, c = i // n_cols, i % n_cols
            axes[r][c].imshow(frame)
            time_sec = frame_idx / fps if fps > 0 else 0
            fc = face_counts[i * step] if i * step < len(face_counts) else 0
            axes[r][c].set_title(
                f'Frame #{frame_idx} (t={time_sec:.1f}s) — {fc} face(s)',
                color='white', fontsize=10, fontweight='bold'
            )

        verdict = 'REAL' if is_real else 'FAKE'
        fig.suptitle(
            f'Video Frame Analysis — {verdict} (Score: {score:.0f}/100)',
            color='#00e676' if is_real else '#ff1744',
            fontsize=14, fontweight='bold', y=1.02
        )
        plt.tight_layout()
        filename = f"video_analysis_{int(time.time())}.png"
        save_path = os.path.join(EXPLAINABILITY_FOLDER, filename)
        plt.savefig(save_path, bbox_inches='tight', facecolor='#0d1117', dpi=100)
        plt.close(fig)
        return url_for('static', filename=f'explainability/{filename}')
    except Exception as e:
        print(f"[!] Video viz error: {e}")
        return None


# ═══════════════════════════════════════════════════════════════
#  METADATA EXTRACTION
# ═══════════════════════════════════════════════════════════════

def extract_image_metadata(filepath):
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
            tag_map = {
                'Make': 'Camera Make', 'Model': 'Camera Model',
                'Software': 'Software', 'DateTime': 'Date Taken',
                'ISOSpeedRatings': 'ISO',
            }
            for tag_id, value in exif_data.items():
                tag_name = TAGS.get(tag_id, str(tag_id))
                if tag_name in tag_map:
                    metadata[tag_map[tag_name]] = str(value)
            if 'Software' in metadata:
                sw = metadata['Software'].lower()
                if any(x in sw for x in ['photoshop', 'gimp', 'stable diffusion', 'midjourney', 'dall-e']):
                    flags.append({'text': f"Image edited/created with: {metadata['Software']}", 'type': 'warning'})
        else:
            flags.append({'text': 'No EXIF metadata found — common in AI-generated or heavily processed images', 'type': 'warning'})

        w, h = img.size
        if w == h and w in [256, 512, 1024, 2048]:
            flags.append({'text': f'Square {w}×{h} resolution — common in GAN-generated images', 'type': 'warning'})
    except Exception as e:
        flags.append({'text': f'Metadata error: {str(e)}', 'type': 'info'})
    return metadata, flags


def extract_audio_metadata(filepath):
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
            rms = np.sqrt(np.mean(y ** 2))
            metadata['RMS Energy'] = f"{rms:.4f}"
            if duration < 1.0:
                flags.append({'text': 'Very short audio clip (< 1s)', 'type': 'warning'})
            if sr < 8000:
                flags.append({'text': f'Low sample rate ({sr} Hz)', 'type': 'warning'})
    except Exception as e:
        flags.append({'text': f'Audio metadata error: {str(e)}', 'type': 'info'})
    return metadata, flags


def extract_video_metadata(filepath):
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
            cap.release()
            if duration < 2.0:
                flags.append({'text': 'Very short video (< 2s)', 'type': 'warning'})
    except Exception as e:
        flags.append({'text': f'Video metadata error: {str(e)}', 'type': 'info'})
    return metadata, flags


# ═══════════════════════════════════════════════════════════════
#  GRAD-CAM (when model IS available)
# ═══════════════════════════════════════════════════════════════

def generate_gradcam(model, img_array, original_img_path, save_filename):
    try:
        last_conv = None
        for layer in reversed(model.layers):
            if 'conv' in layer.name.lower():
                last_conv = layer.name
                break
        if not last_conv:
            return None

        grad_model = tf.keras.Model(
            inputs=model.input,
            outputs=[model.get_layer(last_conv).output, model.output]
        )
        img_tensor = tf.cast(img_array, tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(img_tensor)
            conv_output, predictions = grad_model(img_tensor)
            loss = predictions[0, 0] if predictions.shape[-1] == 1 else predictions[0, tf.argmax(predictions[0])]

        grads = tape.gradient(loss, conv_output)
        if grads is None:
            return None
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap = conv_output[0] @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
        heatmap = heatmap.numpy()

        original = cv2.imread(original_img_path)
        original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        heatmap_resized = cv2.resize(heatmap, (original.shape[1], original.shape[0]))
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        superimposed = cv2.addWeighted(original, 0.6, heatmap_colored, 0.4, 0)
        superimposed_rgb = cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.patch.set_facecolor('#0d1117')
        for ax, img_data, title in zip(axes,
                                         [original_rgb, heatmap_resized, superimposed_rgb],
                                         ['Original', 'Attention Heatmap', 'Overlay']):
            if len(img_data.shape) == 2:
                ax.imshow(img_data, cmap='jet')
            else:
                ax.imshow(img_data)
            ax.set_title(title, color='white', fontsize=12, fontweight='bold')
            ax.axis('off')
        plt.tight_layout()
        save_path = os.path.join(EXPLAINABILITY_FOLDER, save_filename)
        plt.savefig(save_path, bbox_inches='tight', facecolor='#0d1117', dpi=120)
        plt.close(fig)
        return url_for('static', filename=f'explainability/{save_filename}')
    except Exception as e:
        print(f"[!] Grad-CAM error: {e}")
        return None


# ═══════════════════════════════════════════════════════════════
#  MAIN DETECTION DISPATCHERS
# ═══════════════════════════════════════════════════════════════

def detect_deepfake_image(file_path):
    """Image detection — uses model if available, else heuristic fallback."""
    if image_model is None:
        return heuristic_image_detection(file_path)

    # Model-based detection
    findings = []
    metadata, meta_flags = extract_image_metadata(file_path)
    findings.extend(meta_flags)
    try:
        img = ima.load_img(file_path, target_size=(IMG_WIDTH, IMG_HEIGHT))
        x = ima.img_to_array(img)
        x = np.expand_dims(x, axis=0) / 255.0

        result = image_model.predict(x)
        raw_score = float(result[0][0]) if result.shape[-1] == 1 else float(result[0][0])

        if raw_score < 0.50:
            result_string = "FAKE — AI-Generated or Manipulated Image"
            is_real = False
            confidence = (1 - raw_score) * 100
        else:
            result_string = "REAL — Authentic Image"
            is_real = True
            confidence = raw_score * 100

        gradcam_url = generate_gradcam(image_model, x, file_path, f"gradcam_{int(time.time())}.png")
        findings.append({'text': f'Model score: {raw_score:.4f} (threshold: 0.50)', 'type': 'info'})
        if confidence > 90:
            findings.append({'text': f'High confidence ({confidence:.1f}%)', 'type': 'success' if is_real else 'danger'})
        elif confidence < 60:
            findings.append({'text': f'Low confidence ({confidence:.1f}%) — manual review recommended', 'type': 'warning'})

        return {
            'result': result_string,
            'confidence': round(confidence, 1),
            'is_real': is_real,
            'explainability_image': gradcam_url,
            'explainability_caption': 'Grad-CAM heatmap showing regions the model focused on. Red/warm areas indicate high attention.',
            'findings': findings,
            'metadata': metadata,
        }
    except Exception as e:
        return heuristic_image_detection(file_path)


def detect_deepfake_audio(filepath):
    """Audio detection — uses model if available, else heuristic fallback."""
    if audio_model is None:
        return heuristic_audio_detection(filepath)

    findings = []
    metadata, meta_flags = extract_audio_metadata(filepath)
    findings.extend(meta_flags)
    try:
        SAMPLE_RATE, N_MELS, DURATION, max_time_steps = 16000, 128, 5, 109
        audio, _ = librosa.load(filepath, sr=SAMPLE_RATE, duration=DURATION)
        mel_spectro = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_mels=N_MELS)
        mel_spectro = librosa.power_to_db(mel_spectro, ref=np.max)

        if mel_spectro.shape[1] < max_time_steps:
            mel_spectro = np.pad(mel_spectro, ((0, 0), (0, max_time_steps - mel_spectro.shape[1])), mode='constant')
        else:
            mel_spectro = mel_spectro[:, :max_time_steps]

        mel_spec = np.expand_dims(mel_spectro, axis=0)
        result = audio_model.predict(mel_spec)
        classes = np.argmax(result, axis=1)[0]
        probs = result[0]

        if classes == 0:
            result_string = "FAKE — Spoofed / Synthetic Voice Detected"
            is_real = False
            confidence = float(probs[0]) * 100
        else:
            result_string = "REAL — Bonafide / Authentic Voice"
            is_real = True
            confidence = float(probs[1]) * 100

        explainability_url = _save_audio_heuristic_viz(audio, SAMPLE_RATE, confidence, is_real,
                                                        {'mfcc_var': float(np.mean(np.var(librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=13), axis=1)))})
        findings.append({'text': f'Probabilities: Fake={probs[0]:.4f}, Real={probs[1]:.4f}', 'type': 'info'})
        if not is_real:
            findings.append({'text': 'Possible sources: TTS engines, voice cloning, vocoder synthesis', 'type': 'info'})

        return {
            'result': result_string,
            'confidence': round(confidence, 1),
            'is_real': is_real,
            'explainability_image': explainability_url,
            'explainability_caption': 'Mel-spectrogram, waveform, and MFCC analysis of the audio.',
            'findings': findings,
            'metadata': metadata,
        }
    except Exception as e:
        return heuristic_audio_detection(filepath)


def detect_deepfake_video(filepath):
    """Video detection — uses model if available, else heuristic fallback."""
    if video_model is None:
        return heuristic_video_detection(filepath)

    findings = []
    metadata, meta_flags = extract_video_metadata(filepath)
    findings.extend(meta_flags)
    try:
        video_dataset = validation_dataset([filepath], sequence_length=sequence_length, transform=train_transforms)
        fmap, logits = video_model(video_dataset[0].to(DEVICE))
        logits = sm(logits)
        _, prediction = torch.max(logits, 1)
        confidence = logits[:, int(prediction.item())].item() * 100

        if prediction.item() == 1:
            result_string = "REAL — Authentic Video"
            is_real = True
        else:
            result_string = "FAKE — Deepfake Manipulation Detected"
            is_real = False

        explainability_url = _save_video_heuristic_viz(
            _sample_frames_for_viz(filepath), 
            float(metadata.get('FPS', '30').replace(',', '')),
            confidence, is_real,
            [], []
        )
        findings.append({'text': f'ResNeXt50+LSTM model confidence: {confidence:.1f}%', 'type': 'info'})
        if not is_real:
            findings.append({'text': 'Possible sources: FaceSwap, DeepFaceLab, First Order Motion', 'type': 'info'})

        return {
            'result': result_string,
            'confidence': round(confidence, 1),
            'is_real': is_real,
            'explainability_image': explainability_url,
            'explainability_caption': 'Frame analysis with face detection. Model uses temporal consistency via LSTM.',
            'findings': findings,
            'metadata': metadata,
        }
    except Exception as e:
        print(f"[!] Model video detection failed, falling back to heuristic: {e}")
        return heuristic_video_detection(filepath)


def _sample_frames_for_viz(filepath):
    """Sample frames from video for visualization."""
    cap = cv2.VideoCapture(filepath)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total_frames - 1, min(6, total_frames), dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append((idx, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    return frames


# ═══════════════════════════════════════════════════════════════
#  FLASK ROUTES — WEB UI
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
    file = request.files.get('file')
    if not file or not allowed_file(file.filename):
        return render_template('result.html', result='Invalid file', confidence=0,
                               is_real=None, option='image', explainability_image=None,
                               explainability_caption='', findings=[{'text': 'Upload a valid image', 'type': 'danger'}],
                               metadata={})
    filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filename)
    analysis = detect_deepfake_image(filename)
    return render_template('result.html', option='image', **analysis)


@app.route('/upload-audio', methods=['POST'])
def upload_audio_file():
    file = request.files.get('file')
    if not file or not allowed_file(file.filename):
        return render_template('result.html', result='Invalid file', confidence=0,
                               is_real=None, option='audio', explainability_image=None,
                               explainability_caption='', findings=[{'text': 'Upload a valid audio file', 'type': 'danger'}],
                               metadata={})
    filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filename)
    analysis = detect_deepfake_audio(filename)
    return render_template('result.html', option='audio', **analysis)


@app.route('/upload-video', methods=['POST'])
def upload_video_file():
    file = request.files.get('file')
    if not file or not allowed_file(file.filename):
        return render_template('result.html', result='Invalid file', confidence=0,
                               is_real=None, option='video', explainability_image=None,
                               explainability_caption='', findings=[{'text': 'Upload a valid video file', 'type': 'danger'}],
                               metadata={})
    filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filename)
    analysis = detect_deepfake_video(filename)
    return render_template('result.html', option='video', **analysis)


# ═══════════════════════════════════════════════════════════════
#  JSON API ROUTES
# ═══════════════════════════════════════════════════════════════

@app.route('/api/detect/image', methods=['POST'])
def api_detect_image():
    """
    API: Detect deepfake in image.
    Input:  multipart/form-data with 'file' field
    Output: JSON with result, confidence, is_real, findings, metadata
    """
    file = request.files.get('file')
    if not file or not allowed_file(file.filename):
        return jsonify({'error': 'No valid file provided',
                        'accepted_formats': ['jpg', 'jpeg', 'png', 'bmp', 'webp']}), 400

    filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filename)
    analysis = detect_deepfake_image(filename)

    # Remove url_for paths for API (not useful outside web context)
    if analysis.get('explainability_image'):
        analysis['explainability_image'] = request.host_url.rstrip('/') + analysis['explainability_image']

    return jsonify({
        'status': 'success',
        'media_type': 'image',
        'result': analysis['result'],
        'confidence': analysis['confidence'],
        'is_real': analysis['is_real'],
        'findings': analysis['findings'],
        'metadata': analysis['metadata'],
        'explainability_image_url': analysis.get('explainability_image'),
        'timestamp': datetime.now().isoformat(),
    })


@app.route('/api/detect/audio', methods=['POST'])
def api_detect_audio():
    """
    API: Detect deepfake in audio.
    Input:  multipart/form-data with 'file' field
    Output: JSON with result, confidence, is_real, findings, metadata
    """
    file = request.files.get('file')
    if not file or not allowed_file(file.filename):
        return jsonify({'error': 'No valid file provided',
                        'accepted_formats': ['wav', 'mp3', 'flac', 'ogg', 'm4a']}), 400

    filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filename)
    analysis = detect_deepfake_audio(filename)

    if analysis.get('explainability_image'):
        analysis['explainability_image'] = request.host_url.rstrip('/') + analysis['explainability_image']

    return jsonify({
        'status': 'success',
        'media_type': 'audio',
        'result': analysis['result'],
        'confidence': analysis['confidence'],
        'is_real': analysis['is_real'],
        'findings': analysis['findings'],
        'metadata': analysis['metadata'],
        'explainability_image_url': analysis.get('explainability_image'),
        'timestamp': datetime.now().isoformat(),
    })


@app.route('/api/detect/video', methods=['POST'])
def api_detect_video():
    """
    API: Detect deepfake in video.
    Input:  multipart/form-data with 'file' field
    Output: JSON with result, confidence, is_real, findings, metadata
    """
    file = request.files.get('file')
    if not file or not allowed_file(file.filename):
        return jsonify({'error': 'No valid file provided',
                        'accepted_formats': ['mp4', 'avi', 'mov', 'mkv']}), 400

    filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filename)
    analysis = detect_deepfake_video(filename)

    if analysis.get('explainability_image'):
        analysis['explainability_image'] = request.host_url.rstrip('/') + analysis['explainability_image']

    return jsonify({
        'status': 'success',
        'media_type': 'video',
        'result': analysis['result'],
        'confidence': analysis['confidence'],
        'is_real': analysis['is_real'],
        'findings': analysis['findings'],
        'metadata': analysis['metadata'],
        'explainability_image_url': analysis.get('explainability_image'),
        'timestamp': datetime.now().isoformat(),
    })


@app.route('/api/health', methods=['GET'])
def api_health():
    """API: Health check and model status."""
    return jsonify({
        'status': 'running',
        'app': 'Sach-AI Deepfake Detection',
        'version': '1.0.0',
        'models': {
            'image': 'loaded' if image_model else 'heuristic_fallback',
            'audio': 'loaded' if audio_model else 'heuristic_fallback',
            'video': 'loaded' if video_model else 'heuristic_fallback',
        },
        'supported_formats': {
            'image': ['jpg', 'jpeg', 'png', 'bmp', 'webp'],
            'audio': ['wav', 'mp3', 'flac', 'ogg', 'm4a'],
            'video': ['mp4', 'avi', 'mov', 'mkv'],
        },
        'api_endpoints': {
            'POST /api/detect/image': 'Analyze image for deepfake',
            'POST /api/detect/audio': 'Analyze audio for deepfake',
            'POST /api/detect/video': 'Analyze video for deepfake',
            'GET  /api/health': 'This endpoint',
        },
        'timestamp': datetime.now().isoformat(),
    })


# ═══════════════════════════════════════════════════════════════
#  START
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("   Sach-AI — Deepfake Detection System")
    print("   Beyond the mask, lies the real.")
    print("=" * 60)
    print(f"   Image Model: {'✓ Loaded' if image_model else '⚡ Heuristic mode'}")
    print(f"   Audio Model: {'✓ Loaded' if audio_model else '⚡ Heuristic mode'}")
    print(f"   Video Model: {'✓ Loaded' if video_model else '⚡ Heuristic mode'}")
    print(f"   Device: {DEVICE if DEVICE else 'CPU'}")
    print("=" * 60)
    print("   Web UI: http://localhost:5000")
    print("   API:    http://localhost:5000/api/health")
    print("=" * 60 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)

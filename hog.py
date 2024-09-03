import numpy as np
from scipy.stats import skew, kurtosis
from skimage.feature import hog
import cv2
import xgboost as xgb


def train_xgboost_anomaly_detector(normal_features, anomalous_features):
    """Train XGBoost model for anomaly detection."""
    X = np.vstack([normal_features, anomalous_features])
    y = np.hstack([np.zeros(len(normal_features)), np.ones(len(anomalous_features))])
    
    dtrain = xgb.DMatrix(X, label=y)
    
    params = {
        'max_depth': 3,
        'eta': 0.1,
        'objective': 'binary:logistic',
        'eval_metric': 'auc'
    }
    
    num_round = 100
    bst = xgb.train(params, dtrain, num_round)
    
    return bst

def compute_optical_flow_stats(prev_frame, curr_frame):
    """Compute optical flow statistics between two frames."""
    prev_np = prev_frame.cpu().numpy().transpose(1, 2, 0)
    curr_np = curr_frame.cpu().numpy().transpose(1, 2, 0)
    
    prev_gray = cv2.cvtColor(prev_np, cv2.COLOR_RGB2GRAY)
    curr_gray = cv2.cvtColor(curr_np, cv2.COLOR_RGB2GRAY)
    
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    
    stats = {
        'mean_magnitude': np.mean(magnitude),
        'std_magnitude': np.std(magnitude),
        'mean_angle': np.mean(angle),
        'std_angle': np.std(angle)
    }
    return np.array(list(stats.values()))


def compute_hog_features(frame, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
    """Compute HOG features for a frame."""
    frame_np = frame.cpu().numpy().transpose(1, 2, 0)  # Change to HWC format
    features = hog(frame_np, orientations=orientations, pixels_per_cell=pixels_per_cell,
                   cells_per_block=cells_per_block, channel_axis=-1)
    return features

def compute_pixel_statistics(frame):
    """Compute statistical measures of pixel values."""
    frame_np = frame.cpu().numpy()
    mean = np.mean(frame_np, axis=(1, 2))
    variance = np.var(frame_np, axis=(1, 2))
    skewness = skew(frame_np, axis=(1, 2))
    kurt = kurtosis(frame_np, axis=(1, 2))
    return np.concatenate([mean, variance, skewness, kurt])

def compute_color_histogram(frame, bins=32):
    """Compute color histogram for each channel."""
    histograms = []
    for channel in range(frame.shape[0]):
        hist = torch.histc(frame[channel], bins=bins, min=0, max=1)
        histograms.append(hist)
    return torch.cat(histograms)

def get_latent_representation(frame, imf_model):
    """Extract latent representation from IMF model's encoder."""
    with torch.no_grad():
        latent = imf_model.latent_token_encoder(frame.unsqueeze(0))
    return latent.squeeze(0)


def compute_reconstruction_error(frame, reference_frame, imf_model):
    """Compute reconstruction error using the IMF model."""
    with torch.no_grad():
        reconstructed = imf_model(frame.unsqueeze(0), reference_frame.unsqueeze(0))[0]
    
    mse = F.mse_loss(reconstructed, frame.unsqueeze(0))
    return mse.item()

def extract_frame_features(curr_frame, prev_frame, reference_frame, imf_model):
    """Extract all features for a given frame."""
    pixel_stats = compute_pixel_statistics(curr_frame)
    hog_features = compute_hog_features(curr_frame)
    color_hist = compute_color_histogram(curr_frame)
    flow_stats = compute_optical_flow_stats(prev_frame, curr_frame)
    latent_rep = get_latent_representation(curr_frame, imf_model)
    recon_error = compute_reconstruction_error(curr_frame, reference_frame, imf_model)
    
    return np.concatenate([
        pixel_stats,
        hog_features,
        color_hist.cpu().numpy(),
        flow_stats,
        latent_rep.cpu().numpy(),
        [recon_error]
    ])

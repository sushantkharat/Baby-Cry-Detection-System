# Baby Cry Detection System

This project implements an audio-based baby cry detection system that can identify crying sounds in various audio environments, including those with background noise or music.

## Features

- Audio feature extraction using librosa
- Rule-based classification system
- Support for different audio environments (with/without noise, with/without music)
- Visualization of audio analysis (waveform, mel spectrogram, MFCCs)
- Performance evaluation and reporting
- Configurable detection parameters

## Project Structure

```
Cry_Detection/
├── main.py              # Main implementation file
├── config.txt           # Configuration parameters
├── requirements.txt     # Python dependencies
├── Dataset/            # Audio dataset directory
│   ├── Cry-Noise-NoMusic/
│   ├── Cry-NoNoise-Music/
│   ├── Cry-NoNoise-NoMusic/
│   ├── NoCry-Noise-NoMusic/
│   └── NoCry-NoNoise-Music/
└── results/            # Output directory for analysis
```

## Installation

1. Clone the repository
2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Run the main script to evaluate the dataset and test random samples:
```bash
python main.py
```

### Using the CryDetector Class

```python
from main import CryDetector

# Create detector instance
detector = CryDetector()

# Evaluate on dataset
detector.evaluate_on_dataset("Dataset")

# Test random samples
detector.test_three_random_samples("Dataset")
```

## Configuration

The `config.txt` file contains all the threshold values used for cry detection. You can modify these values to adjust the detection sensitivity:

```ini
# Pre-filter thresholds
CENTROID_RATIO_MIN = 0.15    # Minimum spectral centroid ratio
FLATNESS_MAX = 0.40         # Maximum spectral flatness
HARMONIC_RATIO_MIN = 0.95   # Minimum harmonic ratio

# Scoring thresholds
RMS_MIN = 0.012            # Minimum RMS energy
CENTROID_MEAN_MIN = 800    # Minimum spectral centroid
CENTROID_MEAN_MAX = 4000   # Maximum spectral centroid
MFCC_DIFF_MIN = 150        # Minimum MFCC difference
BANDWIDTH_MIN = 1800       # Minimum bandwidth
MFCC_STD_MIN = 50          # Minimum MFCC standard deviation
ONSET_MEAN_MIN = 0.5       # Minimum onset strength
ROLLOFF_MIN = 3000         # Minimum spectral rolloff

# Default score cutoff
DEFAULT_SCORE_CUTOFF = 2   # Minimum score for cry detection
```

## Features Used for Detection

1. **Basic Features**
   - Zero Crossing Rate (ZCR)
   - Root Mean Square (RMS) energy

2. **Spectral Features**
   - Spectral Centroid (mean and ratio)
   - Spectral Bandwidth
   - Spectral Flatness
   - Spectral Rolloff

3. **MFCC Features**
   - MFCC coefficients (1-4 and 7-13)
   - MFCC standard deviation

4. **Harmonic Features**
   - Harmonic/Percussive ratio

5. **Temporal Features**
   - Onset strength
   - Onset standard deviation

## Output

The system generates several outputs in the `results` directory:

1. **Analysis Images**
   - Waveform visualization
   - Mel spectrogram
   - MFCC visualization
   - Confusion matrix

2. **CSV Reports**
   - Classification results
   - Feature statistics

## Performance Metrics

The system evaluates performance using:
- Precision
- Recall
- F1 Score
- Accuracy
- Confusion Matrix

## Dataset Structure

The dataset should be organized as follows:
- `Cry-Noise-NoMusic/`: Crying sounds with background noise
- `Cry-NoNoise-Music/`: Crying sounds with background music
- `Cry-NoNoise-NoMusic/`: Clean crying sounds
- `NoCry-Noise-NoMusic/`: Non-crying sounds with noise
- `NoCry-NoNoise-Music/`: Non-crying sounds with music

## Dependencies

- numpy
- librosa
- matplotlib
- pandas
- scikit-learn
- soundfile

import os
import random
import librosa
import numpy as np
import pandas as pd
import librosa.display
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report

class CryDetector:
    def __init__(self, config_path='config.txt'):
        self.load_config(config_path)
        self.RESULTS_DIR = Path("results")
        self.RESULTS_DIR.mkdir(exist_ok=True)
        self.CRY_FOLDERS = {'Cry-Noise-NoMusic', 'Cry-NoNoise-Music', 'Cry-NoNoise-NoMusic'}

    def load_config(self, config_path):
        self.config = {}
        with open(config_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    key, value = line.split('=')
                    self.config[key.strip()] = float(value.strip())

    def extract_features(self, y, sr):
        # Trim silence
        y, _ = librosa.effects.trim(y, top_db=20)

        # Basic features
        zcr = np.mean(librosa.feature.zero_crossing_rate(y)[0])
        rms = np.mean(librosa.feature.rms(y=y)[0])
        
        # Spectral features
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        centroid_mean = np.mean(centroid)
        centroid_std = np.std(centroid)
        centroid_ratio = centroid_std / centroid_mean if centroid_mean > 0 else 0
        
        # Enhanced MFCC analysis
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean_1_4 = np.mean(mfccs[0:4])
        mfcc_mean_7_13 = np.mean(mfccs[6:13])
        mfcc_std = np.std(mfccs, axis=1)
        mfcc_std_mean = np.mean(mfcc_std)
        
        # Additional spectral features
        bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)[0])
        flatness = np.mean(librosa.feature.spectral_flatness(y=y)[0])
        rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)[0])
        
        # Harmonic and percussive features
        harm, perc = librosa.effects.hpss(y)
        h_ratio = np.mean(np.abs(harm)) / (np.mean(np.abs(perc)) + 1e-6)
        
        # Onset strength
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onset_mean = np.mean(onset_env)
        onset_std = np.std(onset_env)

        return {
            'zcr': zcr,
            'rms': rms,
            'centroid_mean': centroid_mean,
            'centroid_ratio': centroid_ratio,
            'mfcc_mean_1_4': mfcc_mean_1_4,
            'mfcc_mean_7_13': mfcc_mean_7_13,
            'mfcc_std_mean': mfcc_std_mean,
            'bandwidth': bandwidth,
            'flatness': flatness,
            'rolloff': rolloff,
            'h_ratio': h_ratio,
            'onset_mean': onset_mean,
            'onset_std': onset_std
        }

    def classify(self, features, score_cutoff=None):
        if score_cutoff is None:
            score_cutoff = self.config['DEFAULT_SCORE_CUTOFF']
        
        score = 0
        
        # Pre-filter: signal too steady? likely background/noise
        if features['centroid_ratio'] <= self.config['CENTROID_RATIO_MIN']:
            return False
        
        # Allow more cry variation, be softer on rejections
        if features['flatness'] > self.config['FLATNESS_MAX']:
            return False
        if features['h_ratio'] < self.config['HARMONIC_RATIO_MIN']:
            return False
        
        # Enhanced scoring system
        if features['rms'] > self.config['RMS_MIN']:
            score += 1
        if self.config['CENTROID_MEAN_MIN'] < features['centroid_mean'] < self.config['CENTROID_MEAN_MAX']:
            score += 1
        if (features['mfcc_mean_1_4'] - features['mfcc_mean_7_13']) > self.config['MFCC_DIFF_MIN']:
            score += 1
        if features['bandwidth'] > self.config['BANDWIDTH_MIN']:
            score += 1
        if features['mfcc_std_mean'] > self.config['MFCC_STD_MIN']:
            score += 1
        if features['onset_mean'] > self.config['ONSET_MEAN_MIN']:
            score += 1
        if features['rolloff'] > self.config['ROLLOFF_MIN']:
            score += 1
        
        return score >= score_cutoff

    def analyze_and_save(self, file_path, folder_name, prediction, ground_truth, features):
        y, sr = librosa.load(file_path, sr=22050)
        fig, ax = plt.subplots(3, 1, figsize=(12, 10))

        librosa.display.waveshow(y, sr=sr, ax=ax[0])
        ax[0].set_title("Waveform")

        S = librosa.feature.melspectrogram(y=y, sr=sr)
        S_dB = librosa.power_to_db(S, ref=np.max)
        img = librosa.display.specshow(S_dB, sr=sr, ax=ax[1], x_axis='time', y_axis='mel')
        fig.colorbar(img, ax=ax[1], format="%+2.0f dB")
        ax[1].set_title("Mel Spectrogram")

        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        img2 = librosa.display.specshow(mfccs, x_axis='time', ax=ax[2])
        fig.colorbar(img2, ax=ax[2])
        ax[2].set_title("MFCCs")

        fig.suptitle(
            f"{Path(file_path).name} | True: {'Cry' if ground_truth else 'No Cry'} | Predicted: {'Cry' if prediction else 'No Cry'}",
            fontsize=14
        )

        folder_result_path = self.RESULTS_DIR / folder_name
        folder_result_path.mkdir(parents=True, exist_ok=True)

        result_img_path = folder_result_path / f"{Path(file_path).stem}_analysis.png"
        plt.tight_layout()
        plt.savefig(result_img_path)
        plt.close()

    def evaluate_on_dataset(self, dataset_path="Dataset"):
        dataset = Path(dataset_path)
        all_feats = {'cry': [], 'nocry': []}
        records_by_cutoff = {i: [] for i in range(1, 6)}

        for folder in dataset.iterdir():
            if folder.is_dir():
                ground_truth = folder.name in self.CRY_FOLDERS
                for file_path in folder.glob("*.ogg"):
                    y, sr = librosa.load(file_path, sr=22050)
                    features = self.extract_features(y, sr)

                    for cutoff in range(1, 6):
                        prediction = self.classify(features, score_cutoff=cutoff)
                        records_by_cutoff[cutoff].append({
                            'file': file_path.name,
                            'folder': folder.name,
                            'actual': int(ground_truth),
                            'predicted': int(prediction),
                            **features
                        })

                    label = 'cry' if ground_truth else 'nocry'
                    all_feats[label].append(features)

        # Evaluate thresholds
        print("\nThreshold Tuning Results:")
        best_f1 = 0
        best_cutoff = None
        best_df = None

        for cutoff in range(1, 6):
            df = pd.DataFrame(records_by_cutoff[cutoff])
            acc = accuracy_score(df['actual'], df['predicted'])
            report = classification_report(df['actual'], df['predicted'], output_dict=True, zero_division=0)
            precision = report['1']['precision']
            recall = report['1']['recall']
            f1 = report['1']['f1-score']

            print(f"Cutoff {cutoff}: Precision={precision:.2f}, Recall={recall:.2f}, F1={f1:.2f}, Accuracy={acc:.2f}")

            if f1 > best_f1:
                best_f1 = f1
                best_cutoff = cutoff
                best_df = df

        print(f"\nBest Score Cutoff: {best_cutoff} (F1 Score = {best_f1:.2f})")

        # Save best results
        best_df.to_csv(self.RESULTS_DIR / f"classification_results_best_cutoff_{best_cutoff}.csv", index=False)
        print(f"Best results saved to classification_results_best_cutoff_{best_cutoff}.csv")

        # Save confusion matrix
        cm = confusion_matrix(best_df['actual'], best_df['predicted'])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Cry", "Cry"])
        disp.plot(cmap='Blues')
        plt.title(f"Confusion Matrix (Best Cutoff = {best_cutoff})")
        plt.savefig(self.RESULTS_DIR / "confusion_matrix.png")
        plt.close()
        print("Confusion matrix saved.")

        print("\nClassification Report (Best Cutoff):")
        print(classification_report(best_df['actual'], best_df['predicted'], target_names=["No Cry", "Cry"]))

        print("\nFeature Statistics Summary:")
        for label in ['cry', 'nocry']:
            feats_df = pd.DataFrame(all_feats[label])
            print(f"\nClass: {label.upper()}")
            print(feats_df.mean())

    def test_three_random_samples(self, dataset_path="Dataset"):
        dataset = Path(dataset_path)
        all_files = list(dataset.glob("*/" + "*.ogg"))
        sampled_files = random.sample(all_files, 3)

        print("\nRunning random sample tests:\n")
        for file_path in sampled_files:
            folder_name = file_path.parent.name
            ground_truth = folder_name in self.CRY_FOLDERS
            y, sr = librosa.load(file_path, sr=22050)
            features = self.extract_features(y, sr)
            prediction = self.classify(features)

            print(f"File: {file_path.name} | Folder: {folder_name}")
            print(f"    Ground Truth: {'Cry' if ground_truth else 'No Cry'}")
            print(f"    ZCR mean: {features['zcr']:.4f}")
            print(f"    RMS mean: {features['rms']:.4f}")
            print(f"    Centroid mean: {features['centroid_mean']:.2f} | Std/Mean: {features['centroid_ratio']:.2f}")
            print(f"    MFCC mean (1-4): {features['mfcc_mean_1_4']:.2f} | MFCC mean (7-13): {features['mfcc_mean_7_13']:.2f}")
            print(f"    Bandwidth mean: {features['bandwidth']:.2f}")
            print(f"    Spectral Flatness: {features['flatness']:.4f} | Harmonic Ratio: {features['h_ratio']:.2f}")
            print(f"    Prediction: {'Cry' if prediction else 'No Cry'}\n")

            self.analyze_and_save(file_path, folder_name, prediction, ground_truth, features)

def main():
    detector = CryDetector()
    detector.evaluate_on_dataset("Dataset")
    detector.test_three_random_samples("Dataset")

if __name__ == "__main__":
    main()

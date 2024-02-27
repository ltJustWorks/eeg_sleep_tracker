import mne
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict, cross_val_score, GridSearchCV, KFold
import xgboost as xgb
import seaborn as sns

from mne.datasets import sleep_physionet
import pandas as pd

def get_sleep_files(subject_id):
    return sleep_physionet.age.fetch_data(subjects=[subject_id])

annotation_desc_2_event_id = {
    "Sleep stage W": 1,
    "Sleep stage 1": 2,
    "Sleep stage 2": 3,
    "Sleep stage 3": 4,
    "Sleep stage 4": 4,
    "Sleep stage R": 5,
}

def get_sleep_data(subject_id):
    sleep_files = get_sleep_files(subject_id)
    raw_list = []
    events_list = []
    for file in sleep_files:
        raw = mne.io.read_raw_edf(file[0], stim_channel="Event marker", infer_types=True, preload=True)
        annot = mne.read_annotations(file[1])

        # keep last 30-min wake events before sleep and first 30-min wake events after
        # sleep and redefine annotations on raw data
        annot.crop(annot[1]["onset"] - 30 * 60, annot[-2]["onset"] + 30 * 60)
        raw.set_annotations(annot, emit_warning=False)

        events, _ = mne.events_from_annotations(
            raw, event_id=annotation_desc_2_event_id, chunk_duration=30.0
        )
        raw_list.append(raw)
        events_list.append(events)

    return raw_list, events_list

# create a new event_id that unifies stages 3 and 4
EVENT_ID = {
    "Sleep stage W": 1,
    "Sleep stage 1": 2,
    "Sleep stage 2": 3,
    "Sleep stage 3/4": 4,
    "Sleep stage R": 5,
}

def parse_epochs(raw, events, event_id):
    epochs = mne.Epochs(
        raw=raw,
        events=events,
        event_id=event_id,
        tmin=0.0,
        tmax= 30.0 - 1.0 / raw.info["sfreq"],
        baseline=None,
        preload=True,
    )

    #print("Epochs:", epochs)
    return epochs

def plot_sleep_stages(raw_train, events_train, event_id):
    fig = mne.viz.plot_events(
        events_train,
        event_id=event_id,
        sfreq=raw_train.info["sfreq"],
        first_samp=events_train[0, 0],
    )

    stage_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

def spectral_plot(FREQ_BANDS, psds, freqs):
    fig, ax = plt.subplots(figsize=(10, 6))

    for band_name, (fmin, fmax) in FREQ_BANDS.items():
        # Find indices corresponding to the frequency range
        freq_indices = np.logical_and(freqs >= fmin, freqs <= fmax)
        # Extract PSD values for the frequency range
        psds_band = psds[:, :, freq_indices].mean(axis=(0, 1))
        # Plot PSD for the frequency band
        ax.plot(freqs[freq_indices], psds_band, label=band_name)

    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power Spectral Density')
    ax.legend()
    plt.show()

def transform_epochs(epochs: mne.Epochs):
    FREQ_BANDS = {
        "delta": [0.5, 4.5],
        "theta": [4.5, 8.5],
        "alpha": [8.5, 11.5],
        "sigma": [11.5, 15.5],
        "beta": [15.5, 30],
    }

    spectrum = epochs.compute_psd(picks="eeg", fmin=0.5, fmax=30.0)
    psds, freqs = spectrum.get_data(return_freqs=True)

    psds /= np.sum(psds, axis=-1, keepdims=True)
    #spectral_plot(FREQ_BANDS, psds, freqs)

    X = []
    for fmin, fmax in FREQ_BANDS.values():
        psds_band = psds[:, :, (freqs >= fmin) & (freqs < fmax)].mean(axis=-1)
        X.append(psds_band.reshape(len(psds), -1))

    return np.concatenate(X, axis=1)

def transform_epochs_list(epochs_list):
    return [transform_epochs(epochs) for epochs in epochs_list]

def parse_patients(patients_list):
    total_epochs_list = []
    total_stages_list = []
    for patient in patients_list:
        raw_list, events_list = patient
        epochs_list = []
        stages_list = []

        for raw, events in zip(raw_list, events_list):
            recording_epochs = parse_epochs(raw, events, EVENT_ID)
            #print(pd.DataFrame.from_records(recording_epochs).head())
            epochs_list.append(transform_epochs(recording_epochs))
            recording_stages = recording_epochs.events[:, 2]
            stages_list.append(recording_stages)

        total_epochs_list.append(np.concatenate(epochs_list, axis=0))
        total_stages_list.append(np.concatenate(stages_list, axis=0))
    return np.concatenate(total_epochs_list, axis=0), np.concatenate(total_stages_list, axis=0)

def train_model(patients_list):
    epochs_data, stages_data = parse_patients(patients_list)
    stages_data -= 1

    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 6, 9],
        'learning_rate': [0.1, 0.01, 0.001],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
    }
    model = xgb.XGBClassifier(seed=1)
    kf = KFold(n_splits=5, shuffle=True, random_state=1)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=kf, scoring='accuracy', n_jobs=-1)
    grid_search.fit(epochs_data, stages_data)

    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    print("Best parameters:", best_params)
    print("Best score:", best_score)

    cross_val_scores = cross_val_score(grid_search.best_estimator_, epochs_data, stages_data, cv=kf, scoring='accuracy')

    print("Cross-validated scores:", cross_val_scores)
    print("Accuracy:", cross_val_scores.mean())

    # cm = confusion_matrix(stages_data, y_pred)
    # print("Confusion Matrix:")
    # print(cm)

    # unique_classes = np.unique(stages_data)
    # plt.figure(figsize=(8, 6))
    # sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=unique_classes, yticklabels=unique_classes, norm=LogNorm())
    # plt.xlabel('Predicted')
    # plt.ylabel('Real')
    # plt.title('Confusion Matrix')
    # plt.show()

    return model

if __name__ == "__main__":
    patients_list = [get_sleep_data(i) for i in range(1, 10+1)]
    train_model(patients_list)
import mne
import matplotlib.pyplot as plt

def parse_epochs(raw_train, events_train, event_id):
    epochs_train = mne.Epochs(
        raw=raw_train,
        events=events_train,
        event_id=event_id,
        tmin=0.0,
        tmax= 30.0 - 1.0 / raw_train.info["sfreq"],
        baseline=None,
    )

    print("Epochs:", epochs_train)

def show_eeg_plot():
    annotations = mne.read_annotations('data/testing/SC4591GY-Hypnogram.edf')

    raw_eeg = mne.io.read_raw_edf('data/testing/SC4001E0-PSG.edf', preload=True)

    raw_eeg.set_annotations(annotations)
    raw_eeg.plot(block=True)

def show_sleep_stages():
    annotation_desc_2_event_id = {
        "Sleep stage W": 1,
        "Sleep stage 1": 2,
        "Sleep stage 2": 3,
        "Sleep stage 3": 4,
        "Sleep stage 4": 4,
        "Sleep stage R": 5,
    }

    raw_train = mne.io.read_raw_edf('data/testing/SC4001E0-PSG.edf', stim_channel="Event marker", infer_types=True, preload=True)
    annot_train = mne.read_annotations('data/testing/SC4001EC-Hypnogram.edf')

    # keep last 30-min wake events before sleep and first 30-min wake events after
    # sleep and redefine annotations on raw data
    annot_train.crop(annot_train[1]["onset"] - 30 * 60, annot_train[-2]["onset"] + 30 * 60)
    raw_train.set_annotations(annot_train, emit_warning=False)

    events_train, _ = mne.events_from_annotations(
        raw_train, event_id=annotation_desc_2_event_id, chunk_duration=30.0
    )

    # create a new event_id that unifies stages 3 and 4
    event_id = {
        "Sleep stage W": 1,
        "Sleep stage 1": 2,
        "Sleep stage 2": 3,
        "Sleep stage 3/4": 4,
        "Sleep stage R": 5,
    }
    
    parse_epochs(raw_train, events_train, event_id)

    fig = mne.viz.plot_events(
        events_train,
        event_id=event_id,
        sfreq=raw_train.info["sfreq"],
        first_samp=events_train[0, 0],
    )

    stage_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

show_sleep_stages()


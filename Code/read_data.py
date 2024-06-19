import mne
import numpy as np
import pandas as pd
import os


def epoch_data(meg_path, event_path, save_dir):
    raw = mne.io.read_raw_ctf(meg_path, preload=False)
    raw.pick(picks=['mag'])
    raw.load_data()
    raw.filter(0.1, 30, method="iir")
    df = pd.read_csv(event_path, delimiter='\t')
    # subset given events of interest
    df_crop = df[df['type'].str.contains('word_onset', na=False)]
    # remove silence
    df_crop = df_crop.query("value != 'sp'")
    word_samples = np.array(df_crop['onset'] * raw.info['sfreq'], dtype='int')
    n_words = len(word_samples)

    # put the event times into a matrix shape that mne python expects
    word_events = np.zeros([n_words, 3], dtype='int')
    word_events[:, 0] = word_samples
    epochs = mne.Epochs(raw, word_events, tmin=-2.0, tmax=2.0,
                        baseline=(-2.0, 2.0), preload=False, metadata=df_crop, decim=12)
    
    print(epochs.metadata)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    epochs.save(save_dir + "clean-epo.fif", overwrite=True)


if __name__ == "__main__":
    root_dir = r"C:\Users\ricky\OneDrive\Desktop\Datasci125\Data"

    # sessions = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]
    # patients = ["01", "02", "03"]
    patients = ["03"]
    sessions = ["01"]
    for pat_id in patients:
        for ses_id in sessions:
            print(f"Processing patient {pat_id}, session {ses_id}")
            meg_path1 = f'{root_dir}/sub-0{pat_id}/ses-0{ses_id}/meg/sub-0{pat_id}_ses-0{ses_id}_task-compr_meg.ds'
            event_path1 = f'{root_dir}/sub-0{pat_id}/ses-0{ses_id}/meg/sub-0{pat_id}_ses-0{ses_id}_task-compr_events.tsv'
            save_dir1 = f"C:/Users/ricky/OneDrive/Desktop/Datasci125/Data/sub_0{pat_id}/ses_0{ses_id}/"
            epoch_data(meg_path1, event_path1, save_dir1)

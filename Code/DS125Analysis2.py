import os
import random
import re
from typing import List, Tuple

import mne
import numpy as np
import pandas as pd
import spacy
import syllapy
from fastcoref import spacy_component
from mne.stats import permutation_cluster_test
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

nlp = spacy.load('en_core_web_lg')
nlp.add_pipe("fastcoref")

mne.set_log_level(verbose=False)


def read_meg(meg_path: str, event_path: str):
    """
    Read and preprocess MEG data from the given path, clean and filter the data,
    and return it along with the event DataFrame.

    Parameters:
    meg_path (str): The file path to the MEG data.
    event_path (str): The file path to the event data.

    Returns:
    tuple: A tuple containing the preprocessed raw MEG data and the cleaned event DataFrame.
    """
    raw = mne.io.read_raw_ctf(meg_path, preload=False)
    raw.pick(picks=['mag'])
    raw.resample(100)
    raw.load_data()
    raw.filter(0.1, 30, method='iir')

    df = pd.read_csv(event_path, delimiter='\t')
    df_crop = df[df['type'].str.contains('word_onset', na=False)]
    df_crop = df_crop.query("value != 'sp'")

    return raw, df_crop


def session_text(root_dir: str, session: str, pat_id: str) -> str:
    """
    Concatenate text from files matching a specific session pattern in the given directory.

    Parameters:
    root_dir (str): The root directory containing the stimuli files.
    session (str): The session identifier to match files.

    Returns:
    str: The concatenated text from all matching files.
    """
    full_text = ""
    if session == "08" and pat_id == "03":
        pattern = re.compile(f'{session}_[1-6]\.txt')
    else:
        pattern = re.compile(f'{session}_\d\.txt')
    for filename in os.listdir(root_dir + "\stimuli"):
        if pattern.match(filename):
            file_path = os.path.join(root_dir + "\stimuli", filename)
            with open(file_path, 'r') as file:
                full_text += file.read().replace("\n", " ")
    return full_text


def story_text(root_dir: str) -> str:
    """
    Read and concatenate full story from the specified directory.

    Parameters:
    root_dir (str): The root directory containing the text files.

    Returns:
    str: The concatenated text from all the text files in the directory.
    """
    full_text = ""
    stimuli_dir = os.path.join(root_dir, "stimuli")

    for filename in os.listdir(stimuli_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(stimuli_dir, filename)
            with open(file_path, 'r') as file:
                full_text += file.read().replace("\n", " ")

    return full_text


def has_accented_characters(token) -> bool:
    """
    Checks if the given token has any accented characters.

    Parameters:
    token (spacy.tokens.Token): The token to be checked.

    Returns:
    bool: True if the token has accented characters, False otherwise.
    """
    return any(ord(char) > 127 for char in token.text)


def find_accent(token) -> int:
    """
    Finds the index of the first accented character in the token.

    Parameters:
    token (spacy.tokens.Token): The token to be checked.

    Returns:
    int: The index of the first accented character, or -1 if none are found.
    """
    for idx, char in enumerate(token.text):
        if ord(char) > 127:
            return idx
    return -1


def pos_tagger(doc: spacy.tokens.Doc, ses_id, pt_id) -> Tuple[List[str], List[str]]:
    """
    Tags parts of speech in the given text. Note: This function does not work 
    on words with more than one accent.

    Parameters:
    doc (spacy.tokens.Doc): The spacy Doc object to be processed.

    Returns:
    tuple: A tuple containing two lists - parts of speech tags and corresponding tokens.
    """
    pos = []
    tokens = []

    for token in doc:
        if ses_id == "01" and pt_id == "03":
            if token.i < 10:
                continue
        if token.text.strip() and not token.is_punct:
            if has_accented_characters(token):
                text = token.text[find_accent(token) + 1:]
                if text:
                    pos.append(token.pos_)
                    tokens.append(text)
            elif token.text.lower() in ["n't", "'ll", "'ve", "'m", "'d", "'t", "'s", "'re"]:
                tokens[-1] += token.text
            else:
                pos.append(token.pos_)
                tokens.append(token.text)

    return pos, tokens


def get_reduced_tokens(spans: List[spacy.tokens.Span]) -> List[spacy.tokens.Token]:
    """
    Reduce tokens in each span to a single representative token based on part of speech tags.

    Parameters:
    spans (List[spacy.tokens.Span]): A list of spacy token spans.

    Returns:
    List[spacy.tokens.Token]: A list of reduced tokens.
    """
    reduced_tokens = []

    for span in spans:
        pnoun = False
        reduced_token = None
        nouns = []

        for token in span:
            if token.pos_ in ['PROPN', 'PRON', 'NOUN']:
                if token.pos_ == "PROPN":
                    pnoun = True
                    reduced_token = token
                elif not pnoun:
                    nouns.append(token)

        if pnoun:
            reduced_tokens.append(reduced_token)
        elif nouns:
            reduced_tokens.append(random.choice(nouns))

    return reduced_tokens


def get_head(reduced_spans) -> str:
    """
    Determine the head token from the given spans based on their part of speech tags.

    Parameters:
    reduced_spans (list): A list of spacy tokens.

    Returns:
    str: The head token with the highest frequency, or an empty string if no head token is found.
    """
    pnoun = False
    head_token = {}

    for token in reduced_spans:
        if token.pos_ in ['PROPN', 'PRON', 'NOUN']:
            if token.pos_ == "PROPN":
                pnoun = True
                head_token[token.text] = head_token.get(token.text, 0) + 1
            elif token.pos_ == "NOUN" and not pnoun:
                head_token[token.text] = head_token.get(token.text, 0) + 1

    if not head_token:
        return ""

    return max(head_token, key=head_token.get)


def coref_tagger(doc: spacy.tokens.Doc, ses_id, pt_id) -> List[str]:
    """
    Processes the given text to resolve coreferences. Note: This function 
    does not work on words with more than one accent.

    Parameters:
    doc (spacy.tokens.Doc): The spacy Doc object to be processed.

    Returns:
    list: A list of resolved tokens or None for unresolved tokens.
    """
    resolved = []
    cluster_dict = {}

    # Extract coreference clusters
    clusters = doc._.coref_clusters
    for cluster in clusters:
        spans = [doc.char_span(span[0], span[1]) for span in cluster]
        reduced_tokens = get_reduced_tokens(spans)
        cluster_head = get_head(reduced_tokens)
        for token in reduced_tokens:
            cluster_dict[token] = cluster_head

    for token in doc:
        if ses_id == "01" and pt_id == "03":
            if token.i < 10:
                continue
        if token.text.strip() and not token.is_punct and \
                token.text.lower() not in ["n't", "'ll", "'ve", "'m", "'d", "'t", "'s", "'re"]:
            if has_accented_characters(token):
                if ord(token.text[-1]) > 127:
                    continue
            if token in cluster_dict and cluster_dict[token] != "":
                resolved.append(cluster_dict[token])
            else:
                resolved.append(None)

    return resolved


def create_epochs(df: pd.DataFrame, raw: mne.io.Raw) -> mne.Epochs:
    """
    Create epochs from MEG data using event information from a DataFrame.

    Parameters:
    df (pd.DataFrame): DataFrame containing event information with 'onset' column.
    raw (mne.io.Raw): The raw MEG data.

    Returns:
    mne.Epochs: The created epochs.
    """
    word_samples = np.array(df['onset'] * raw.info['sfreq'], dtype='int')
    n_words = len(word_samples)

    word_events = np.zeros((n_words, 3), dtype='int')
    word_events[:, 0] = word_samples

    epochs = mne.Epochs(raw, word_events, tmin=-2.0, tmax=2.0,
                        baseline=(-2.0, 2.0), preload=False, metadata=df)
    return epochs


def run_LR_model(epochs: np.array, labels: pd.DataFrame, pipeline) -> pd.DataFrame:
    """
    Run a Logistic Regression model on the MEG epochs data with cross-validation.

    Parameters:
    epochs (mne.Epochs): The epochs containing MEG data.
    labels (pd.DataFrame): The DataFrame containing labels for classification.
    pipeline (sklearn.pipeline.Pipeline): The scikit-learn pipeline for the model.

    Returns:
    pd.DataFrame: DataFrame containing AUC scores for each label.
    """
    df_scores = pd.DataFrame()

    for column in labels.columns:
        column = column[0]
        y = labels[column].to_numpy().ravel()
        auc_score = []

        for i in range(epochs.shape[2]):
            X = epochs[:, :, i]
            skf = StratifiedKFold(n_splits=5, shuffle=True)
            scores = []

            for train_index, test_index in skf.split(X, y):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                pipeline.fit(X_train, y_train)
                score = roc_auc_score(
                    y_test, pipeline.predict_proba(X_test)[:, 1])
                scores.append(score)

            auc_score.append(np.mean(scores))

        df_scores[column] = auc_score

    return df_scores


def analysis_2(root_dir: str, save_dir: str, pat_id: str):
    """
    Perform Analysis 2 on the given patient data.

    Args:
        root_dir (str): The root directory path where the patient data is stored.
        save_dir (str): The directory path where the analysis results will be saved.
        pat_id (str): The patient ID.

    Returns:
        None
    """
    print(f"patient {pat_id}")
    all_data = []
    all_labels = []
    ses_ids = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]
    characters = ['Holmes', 'Watson', 'Doctor', 'McCarthy', 'Clair', 'Doran', 
                  'Turner', 'Sutherland', 'Simon', 'Ryder', 'Frank', 'Hatherley', 
                  'Stoner', 'Adler', 'Wilson', 'Ferguson', 'Angel', 'Lestrade']
    pronouns = ['I', 'YOU', 'HE', 'MY', 'HIS', 'ME', 'YOUR', 'SHE', 'HIM', 'HER']

    for ses_id in ses_ids:
        text = session_text(root_dir, ses_id, pat_id)

        print(f"Processing session {ses_id} for patient {pat_id}")

        epochs = mne.read_epochs(f"{root_dir}/sub_0{pat_id}/ses_0{ses_id}/clean-epo.fif", preload=False)
        df = epochs.metadata

        doc = nlp(text)
        df['POS'], _ = pos_tagger(doc, ses_id, pat_id)
        df['coref'] = coref_tagger(doc, ses_id, pat_id)
        df["syllables"] = df["value"].apply(syllapy.count)

        mask = df["POS"].isin(["PROPN", "PRON", "NOUN"])
        noun_epochs = epochs[mask]
        valid_epochs = noun_epochs[pd.notnull(noun_epochs.metadata['coref'])]
        data = valid_epochs.get_data()
        all_data.append(data)
        labels = valid_epochs.metadata
        all_labels.append(labels)

    full_raw_data = np.concatenate(all_data, axis=0)
    metadata = pd.concat(all_labels, axis=0, ignore_index=True)

    for pronoun in pronouns:
        subset_df = metadata[metadata["value"] == pronoun]
        coref_counts = subset_df["coref"].value_counts()
        sig_coref = coref_counts[coref_counts > 30].index.tolist()
        sig_chars = [char for char in characters if char in sig_coref]

        char_df = subset_df[subset_df["coref"].isin(sig_chars)]
        y = char_df[["coref"]]
        X = full_raw_data[char_df.index]

        enc = OneHotEncoder()
        enc_y = enc.fit_transform(y)
        label_df = pd.DataFrame(enc_y.toarray(), columns=enc.categories_)

        pipeline = make_pipeline(StandardScaler(), LogisticRegression(
            random_state=125, max_iter=10000, solver="lbfgs", C=10e-3))

        df_scores = run_LR_model(X, label_df, pipeline)
        df_scores.to_csv(
            f'{save_dir}/df_scores_pt_{pat_id}_trial_{pronoun}.csv', index=False)


def main():
    """
    Main function to run Analysis 2.
    """
    root_dir = r"C:\Users\ricky\OneDrive\Desktop\Datasci125\Data"
    save_dir = r"C:\Users\ricky\OneDrive\Desktop\Datasci125\Code\Results\Analysis_2"
    # patients = ["01", "02", "03"]
    patients = ["02", "03"]
    for patient in patients:
        analysis_2(root_dir, save_dir, patient)


if __name__ == "__main__":
    main()

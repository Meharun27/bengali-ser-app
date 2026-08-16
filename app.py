import os
import shutil
import uuid
import zipfile
import gdown
import streamlit as st
import pandas as pd
from natsort import natsorted

st.set_page_config(page_title="Bengali SER Annotation Portal", layout="wide", initial_sidebar_state="expanded")

AUDIO_DIR = "dynamic_sentences"
VOTE_DIR = "vote_results"
CSV_PATH = os.path.join(VOTE_DIR, "votes.csv")
ZIP_PATH = "dataset.zip"

# Automatically download and extract the ZIP file from Google Drive
if not os.path.exists(AUDIO_DIR) or not os.listdir(AUDIO_DIR) or not any(f.endswith(".wav") for f in os.listdir(AUDIO_DIR)):
    with st.spinner("Downloading audio dataset from Google Drive... Please wait."):
        os.makedirs(AUDIO_DIR, exist_ok=True)
        file_id = "17jygKZPIgsSTuwR7YKg2Zis0cdM7r-QE"
        
        try:
            gdown.download(id=file_id, output=ZIP_PATH, quiet=False)
            
            with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
                zip_ref.extractall(AUDIO_DIR)
            if os.path.exists(ZIP_PATH):
                os.remove(ZIP_PATH)
        except Exception as e:
            st.error(f"Error downloading or extracting zip file: {e}")

# Flatten any nested folder structures so all .wav files sit directly in AUDIO_DIR
if os.path.exists(AUDIO_DIR):
    for root, dirs, files in os.walk(AUDIO_DIR, topdown=False):
        for file in files:
            if file.lower().endswith(".wav") and not file.startswith("._"):
                src_file = os.path.join(root, file)
                dst_file = os.path.join(AUDIO_DIR, file)
                if src_file != dst_file:
                    if not os.path.exists(dst_file):
                        shutil.move(src_file, dst_file)
        for d in dirs:
            dir_path = os.path.join(root, d)
            try:
                if not os.listdir(dir_path):
                    os.rmdir(dir_path)
            except Exception:
                pass

st.title("🎙️ Bengali Speech Emotion Recognition (SER) Annotation Portal")
st.markdown("Evaluate audio clips with batching, multi-annotator tracking, attribute tagging, and majority-consensus routing.")

BATCH_SIZE = 10     
REQUIRED_VOTES = 3  

emotions = ["Happy", "Sad", "Angry", "Neutral", "Mixed"]

for emo in emotions:
    os.makedirs(os.path.join(VOTE_DIR, emo), exist_ok=True)
os.makedirs(VOTE_DIR, exist_ok=True)

try:
    audio_files = natsorted([f for f in os.listdir(AUDIO_DIR) if f.lower().endswith(".wav") and not f.startswith("._")])
except FileNotFoundError:
    audio_files = []

if not audio_files:
    st.warning(f"No audio files found in directory: `{AUDIO_DIR}`. Please verify that your Google Drive zip file is set to **'Anyone with the link can view'**.")
    st.stop()

CSV_COLUMNS = ["file", "user", "voter_id", "emotion", "intensity", "gender", "noise", "age", "code_switching"]

if os.path.exists(CSV_PATH):
    try:
        votes_df = pd.read_csv(CSV_PATH)
        for col in CSV_COLUMNS:
            if col not in votes_df.columns:
                votes_df = pd.DataFrame(columns=CSV_COLUMNS)
                break
    except Exception:
        votes_df = pd.DataFrame(columns=CSV_COLUMNS)
else:
    votes_df = pd.DataFrame(columns=CSV_COLUMNS)

st.sidebar.header("👤 Annotator Session")
username = st.sidebar.text_input("Annotator Name", value="", placeholder="Enter your full name")

if 'current_annotator' not in st.session_state or st.session_state.current_annotator != username:
    st.session_state.current_annotator = username
    if username.strip():
        st.session_state.voter_id = f"VID-{str(uuid.uuid4())[:6].upper()}"
    else:
        st.session_state.voter_id = ""

voter_id = st.sidebar.text_input("Auto Voter ID", value=st.session_state.voter_id, disabled=True)

if os.path.exists(CSV_PATH) and os.path.getsize(CSV_PATH) > 0:
    st.sidebar.markdown("---")
    st.sidebar.subheader("📥 Download Results")
    with open(CSV_PATH, "rb") as f:
        st.sidebar.download_button("Download votes.csv", f, file_name="votes.csv", mime="text/csv")

if not username.strip():
    st.warning("⚠️ Please enter your name in the sidebar to start voting.")
    st.stop()

def get_user_batch(user):
    already_voted = set(votes_df.loc[votes_df["user"] == user, "file"]) if not votes_df.empty and "user" in votes_df.columns else set()
    needs_votes = []
    for f in audio_files:
        file_vote_count = votes_df[votes_df["file"] == f].shape[0] if not votes_df.empty and "file" in votes_df.columns else 0
        if file_vote_count < REQUIRED_VOTES:
            needs_votes.append(f)
    fresh = [f for f in needs_votes if f not in already_voted]
    return fresh[:BATCH_SIZE]

if 'batch_files' not in st.session_state or st.session_state.get('current_user') != username:
    st.session_state.current_user = username
    st.session_state.batch_files = get_user_batch(username)
    st.session_state.batch_index = 0

batch_files = st.session_state.batch_files
batch_index = st.session_state.batch_index

st.sidebar.markdown("---")
st.sidebar.header("📊 Batch Progress")
st.sidebar.write(f"Active Batch Clips: **{len(batch_files)}**")
if batch_files:
    st.sidebar.progress(min(batch_index / len(batch_files), 1.0))
    st.sidebar.write(f"Progress: Clip {min(batch_index + 1, len(batch_files))} of {len(batch_files)}")

if st.sidebar.button("🔄 Refresh / Load New Batch"):
    st.session_state.batch_files = get_user_batch(username)
    st.session_state.batch_index = 0
    st.rerun()

if batch_index >= len(batch_files):
    st.success("🎉 You have completed your current batch of files! Click 'Refresh / Load New Batch' in the sidebar to fetch more clips.")
    st.stop()

selected_file = batch_files[batch_index]
file_path = os.path.join(AUDIO_DIR, selected_file)

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader(f"🎵 Clip [{batch_index + 1}/{len(batch_files)}]")
    st.code(selected_file)
    try:
        st.audio(file_path)
    except Exception as e:
        st.error(f"Could not load audio: {e}")

with col2:
    st.subheader("🏷️ Attribute Tagging Panel")
    with st.form(key=f"form_{selected_file}"):
        emotion = st.radio("Primary Emotion", emotions, horizontal=True)
        
        col_a, col_b = st.columns(2)
        with col_a:
            intensity = st.selectbox("Intensity", ["Low", "Medium", "High"], index=1)
            gender = st.selectbox("Speaker Gender", ["Female", "Male", "Child"], index=0)
        with col_b:
            noise = st.selectbox("Noise Level", ["Clean", "Low", "Moderate", "High"], index=1)
            age = st.selectbox("Speaker Age", ["Child", "Youth", "Adult", "Senior"], index=2)
            
        code_switching = st.checkbox("Code-Switching Present?")

        submitted = st.form_submit_button("💾 Submit Vote & Next ➡️", type="primary")

        if submitted:
            duplicate_check = False
            if not votes_df.empty and "file" in votes_df.columns and "user" in votes_df.columns:
                if not votes_df[(votes_df["file"] == selected_file) & (votes_df["user"] == username)].empty:
                    duplicate_check = True

            if duplicate_check:
                st.warning(f"⚠️ {username}, you have already voted on `{selected_file}`. Skipping duplicate entry.")
            else:
                new_row = pd.DataFrame([[
                    selected_file, 
                    username, 
                    voter_id,
                    emotion, 
                    intensity, 
                    gender, 
                    noise, 
                    age, 
                    "Yes" if code_switching else "No"
                ]], columns=CSV_COLUMNS)

                if votes_df.empty:
                    votes_df = new_row
                else:
                    votes_df = pd.concat([votes_df, new_row], ignore_index=True)

                write_header = not os.path.exists(CSV_PATH)
                new_row.to_csv(CSV_PATH, mode="a", index=False, header=write_header)

                file_votes = votes_df[votes_df["file"] == selected_file]
                if len(file_votes) >= REQUIRED_VOTES:
                    majority = file_votes["emotion"].value_counts().idxmax()
                    src = os.path.join(AUDIO_DIR, selected_file)
                    dst = os.path.join(VOTE_DIR, majority, selected_file)
                    if os.path.exists(src):
                        shutil.move(src, dst)

                st.success(f"Vote saved successfully for `{selected_file}`!")
                st.session_state.batch_index += 1
                st.rerun()

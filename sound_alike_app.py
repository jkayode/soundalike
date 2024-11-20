import streamlit as st
import pandas as pd
import phonetics
from fuzzywuzzy import fuzz
from sentence_transformers import SentenceTransformer, util
import dropbox
import requests
import time
from io import BytesIO
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


# Dropbox Configuration
client_id = st.secrets["APP_KEY"]
client_secret = st.secrets["APP_SECRET"]
refresh_token = st.secrets["REFRESH_TOKEN"]

def refresh_access_token(client_id, client_secret, refresh_token):
    url = "https://api.dropbox.com/oauth2/token"
    data = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": client_id,
        "client_secret": client_secret
    }
    response = requests.post(url, data=data)
    if response.status_code == 200:
        return response.json()["access_token"]
    else:
        raise Exception(f"Failed to refresh token: {response.json()}")

new_access_token = refresh_access_token(client_id, client_secret, refresh_token)
DROPBOX_ACCESS_TOKEN = new_access_token
DROPBOX_FILE_PATH = "/product_database.csv"
dbx = dropbox.Dropbox(DROPBOX_ACCESS_TOKEN)

# Pretrained Model for Semantic Similarity
model = SentenceTransformer('all-MiniLM-L6-v2')  # Alternative: 'all-mpnet-base-v2'

# Utility Functions
def fetch_product_names_from_dropbox():
    try:
        metadata, response = dbx.files_download(DROPBOX_FILE_PATH)
        df = pd.read_csv(response.raw)
        return df
    except dropbox.exceptions.ApiError as e:
        st.error(f"Error fetching the database file: {e}")
        return pd.DataFrame()

def clean_product_name(name):
    dosage_forms = ["tablet", "capsule", "syrup", "injection", "cream", "solution", "gel", "drop", "caplet", "suspension", "infusion",
    "tablets", "capsules", "syrups", "injections", "creams", "solutions", "gels", "drops", "caplets", "suspensions", "infusions"]
    units = ["mg", "g", "ml", "l", "kg", "mcg", "iu", "mgmg", "mgmgmg"]
    stopwords = ["for", "of", "and", "in", "on", "to", "with"]
    name = name.lower()
    name = re.sub(r"[^\w\s]", "", name)
    name = re.sub(r"\d+", "", name)
    for term in dosage_forms + units + stopwords:
        name = re.sub(rf"\b{term}\b", "", name)
    return re.sub(r"\s+", " ", name).strip()

def upload_file_with_progress(dataframe, dropbox_path):
    """
    Uploads a preprocessed DataFrame to Dropbox with a progress bar.
    
    Parameters:
        dataframe (pd.DataFrame): The DataFrame to upload.
        dropbox_path (str): The Dropbox file path where the data will be uploaded.
    """
    try:
        # Create a buffer to hold the CSV data
        buffer = BytesIO()
        dataframe.to_csv(buffer, index=False)
        buffer.seek(0)

        # Simulate progress for upload
        progress_bar = st.progress(0)
        for i in range(1, 101):
            time.sleep(0.01)  # Simulate processing delay
            progress_bar.progress(i)

        # Upload to Dropbox
        dbx.files_upload(buffer.read(), dropbox_path, mode=dropbox.files.WriteMode.overwrite)
        progress_bar.empty()  # Clear progress bar after completion

        st.success("File uploaded successfully to Dropbox!")

    except dropbox.exceptions.ApiError as e:
        st.error(f"Error uploading the file to Dropbox: {e}")

def detect_similar_names(new_product, database, method, threshold):
    """
    Detect similar product names using various methods.
    - method: 'Fuzzy Matching', 'Cosine Similarity', or 'Deep Learning'
    - threshold: Similarity threshold (float for Cosine/Deep Learning, int for Fuzzy Matching)
    """
    cleaned_new_product = clean_product_name(new_product)

    if method == "Fuzzy Matching":
        # Fuzzy Matching with token set ratio
        results = [
            (new_product, row["ProductName"], fuzz.token_set_ratio(cleaned_new_product, row["CleanedName"]))
            for _, row in database.iterrows()
            if fuzz.token_set_ratio(cleaned_new_product, row["CleanedName"]) >= threshold
        ]
    elif method == "Cosine Similarity":
        # Enhanced Cosine Similarity with n-grams
        names = [cleaned_new_product] + database["CleanedName"].tolist()

        # Use CountVectorizer with word-level n-grams for richer feature extraction
        vectorizer = CountVectorizer(analyzer="word", ngram_range=(1, 2))  # Extract word bigrams
        vectors = vectorizer.fit_transform(names)

        # Compute cosine similarity between the input name and database names
        similarities = cosine_similarity(vectors[0:1], vectors[1:]).flatten()

        # Combine with Fuzzy Matching for improved results
        results = []
        for i, sim in enumerate(similarities):
            fuzzy_score = fuzz.token_set_ratio(cleaned_new_product, database.iloc[i]["CleanedName"])
            combined_score = max(sim, fuzzy_score / 100)  # Normalize fuzzy score to 0-1 range

            if combined_score >= threshold:
                results.append((new_product, database.iloc[i]["ProductName"], round(combined_score, 2)))
    elif method == "Deep Learning":
        # Generate embeddings for input and database names
        embeddings = model.encode([cleaned_new_product] + database["CleanedName"].tolist(), convert_to_tensor=True)

        # Compute cosine similarity
        cosine_scores = util.cos_sim(embeddings[0], embeddings[1:]).squeeze().tolist()

        # Combine results with fuzzy matching
        results = []
        for i, score in enumerate(cosine_scores):
            product_name = database.iloc[i]["ProductName"]
            fuzzy_score = fuzz.token_set_ratio(cleaned_new_product, database.iloc[i]["CleanedName"])
            combined_score = max(score, fuzzy_score / 100)  # Scale fuzzy score to 0-1

            if combined_score >= threshold:
                results.append((new_product, product_name, round(combined_score, 2)))
    else:
        results = []

    # Create a DataFrame from the results and sort by similarity in descending order
    results_df = pd.DataFrame(results, columns=["New Product", "Database Product", "Similarity"])
    results_df = results_df.sort_values(by="Similarity", ascending=False).reset_index(drop=True)

    return results_df

def preprocess_database(uploaded_file):
    try:
        # Read the uploaded CSV
        df = pd.read_csv(uploaded_file)
        if "ProductName" not in df.columns:
            raise ValueError("The uploaded file must have a 'ProductName' column.")
        
        # Clean product names
        df["ProductName"] = df["ProductName"].str.strip().fillna("")  # Remove leading/trailing spaces and handle NaN
        df["CleanedName"] = df["ProductName"].apply(clean_product_name)  # Apply cleaning function

        # Check for empty cleaned names and drop them
        df = df[df["CleanedName"].str.len() > 0]

        # Add progress bar
        progress_bar = st.progress(0)
        total_rows = len(df)

        soundex_codes = []
        for i, name in enumerate(df["CleanedName"]):
            try:
                # Compute Soundex for each word in the cleaned product name
                words = name.split()
                soundex_list = [phonetics.soundex(word) for word in words if word.isalpha()]
                soundex_codes.append(" ".join(soundex_list))  # Combine Soundex codes for multi-word names
            except Exception as e:
                # Handle unexpected errors and skip invalid names
                soundex_codes.append("")
                st.warning(f"Skipped invalid name at row {i + 1}: {name}")
            
            # Update progress
            progress_bar.progress(int((i + 1) / total_rows * 100))

        # Add Soundex column
        df["Soundex"] = soundex_codes
        progress_bar.empty()  # Clear progress bar after completion

        # Return the processed DataFrame
        return df[["ProductName", "CleanedName", "Soundex"]]

    except ValueError as ve:
        st.error(f"Error: {ve}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Unexpected error processing the uploaded file: {e}")
        return pd.DataFrame()

# Streamlit App
st.title("Sound-Alike Detector")
st.subheader("Product Name Similarity Checker")

tabs = st.tabs(["Check Similarity", "Manage Database", "About"])

# Check Similarity
with tabs[0]:
    st.header("Check Similar Product Names")
    new_product = st.text_input("Enter New Product Name:")
    method = st.selectbox("Choose Method", ["Fuzzy Matching", "Cosine Similarity", "Deep Learning"])
    threshold = st.slider("Similarity Threshold", 0.0, 1.0 if method != "Fuzzy Matching" else 100.0, 0.8 if method != "Fuzzy Matching" else 80.0)
    
    if st.button("Check Similarity"):
        database = fetch_product_names_from_dropbox()
        if not database.empty:
            results = detect_similar_names(new_product, database, method, threshold)
            if not results.empty:
                st.write("Similar Product Names Found:")
                st.table(results)
            else:
                st.write("No similar names found.")
        else:
            st.error("Database is empty.")

# Ensure database status is tracked in session state
if "database_status" not in st.session_state:
    st.session_state["database_status"] = fetch_product_names_from_dropbox()

if "needs_refresh" not in st.session_state:
    st.session_state["needs_refresh"] = False

with tabs[1]:
    st.header("Upload and Manage Product Database")

    # Refresh database if needed
    if st.session_state["needs_refresh"]:
        st.session_state["database_status"] = fetch_product_names_from_dropbox()
        st.session_state["needs_refresh"] = False

    # Show real-time status of the database
    st.subheader("Current Database Status")
    database = st.session_state["database_status"]

    if database.empty:
        st.warning("The database is currently empty.")
    else:
        st.write(f"Total Records in Database: {len(database)}")
        st.write("Sample Records:")
        st.dataframe(database.head())

    # Button to clear the database
    if st.button("Clear Database"):
        try:
            # Create an empty DataFrame with the expected structure
            empty_data = pd.DataFrame(columns=["ProductName", "CleanedName", "Soundex"])
            buffer = BytesIO()
            empty_data.to_csv(buffer, index=False)
            buffer.seek(0)

            # Upload the empty file to clear the database
            dbx.files_upload(buffer.read(), DROPBOX_FILE_PATH, mode=dropbox.files.WriteMode.overwrite)
            st.success("Database cleared successfully!")

            # Trigger refresh
            st.session_state["needs_refresh"] = True
        except dropbox.exceptions.ApiError as e:
            st.error(f"Error clearing the database: {e}")

    # File uploader for new product database
    st.subheader("Upload New Database")
    uploaded_file = st.file_uploader("Upload a new product database (CSV format, column 'ProductName')", type=["csv"])

    # Button to preprocess and upload
    if st.button("Upload and Preprocess"):
        if uploaded_file:
            preprocessed_data = preprocess_database(uploaded_file)
            if not preprocessed_data.empty:
                # Fetch the current database
                current_database = fetch_product_names_from_dropbox()

                if not current_database.empty:
                    # Remove duplicates by comparing against current database
                    unique_new_data = preprocessed_data[
                        ~preprocessed_data["CleanedName"].isin(current_database["CleanedName"])
                    ]
                    combined_data = pd.concat([current_database, unique_new_data], ignore_index=True)
                    records_added = len(unique_new_data)
                else:
                    # Current database is empty; all new records are unique
                    combined_data = preprocessed_data
                    records_added = len(preprocessed_data)

                # Upload the updated database to Dropbox
                buffer = BytesIO()
                combined_data.to_csv(buffer, index=False)
                buffer.seek(0)
                try:
                    dbx.files_upload(buffer.read(), DROPBOX_FILE_PATH, mode=dropbox.files.WriteMode.overwrite)
                    st.success(f"Database updated successfully! {records_added} new records added.")
                    
                    # Trigger refresh
                    st.session_state["needs_refresh"] = True
                except dropbox.exceptions.ApiError as e:
                    st.error(f"Error updating the database: {e}")
            else:
                st.error("The uploaded file is empty or invalid.")
        else:
            st.error("No file uploaded. Please upload a CSV file.")

# About Tab
with tabs[2]:
    st.write("""
    ## Overview
    The **Product Name Similarity Checker** is a versatile web application designed to manage and analyze product names 
    in a centralized database. It helps ensure that product names are unique and free of conflicts by detecting 
    lookalike and soundalike names using advanced algorithms. The app integrates with Dropbox for seamless database 
    management and provides users with real-time insights into their product database.
    
    ## Key Features
    1. **Product Name Similarity Detection**:
       - Compare new product names against an existing database.
       - Identify potential conflicts with lookalike or soundalike names.
    2. **Multiple Similarity Detection Methods**:
       - **Fuzzy Matching**: Token-based string similarity for quick comparisons.
       - **Cosine Similarity**: Captures textual patterns using vectorization techniques.
       - **Deep Learning**: Employs pre-trained Sentence Transformers for semantic similarity.
    3. **Database Management**:
       - View, upload, and preprocess product databases.
       - Append only unique product names to avoid duplication.
       - Clear the database as needed.
    4. **Interactive and User-Friendly Interface**:
       - Built with Streamlit for responsiveness and ease of use.
       - Dynamic updates and real-time feedback for users.

    ## How It Works
    1. **Check Similar Product Names**:
       - Enter a new product name in the **Check Similarity** tab.
       - Choose from three similarity detection methods:
         - Fuzzy Matching, Cosine Similarity, or Deep Learning.
       - Adjust the similarity threshold to refine results.
       - View a sorted list of product names from the database that are similar to the input.
    2. **Manage Product Database**:
       - Upload a CSV file containing product names.
       - Automatically preprocess and clean the data:
         - Remove dosage forms, units, special characters, and stopwords.
         - Normalize names to lowercase.
       - Append only unique product names to the database, preserving existing records.
       - Clear the database if needed.
    3. **Database Cleaning and Validation**:
       - Computes Soundex codes for phonetic similarity detection.
       - Identifies and skips invalid or redundant entries dynamically.

    ## Technologies Used
    1. **Similarity Detection Algorithms**:
       - **FuzzyWuzzy**: Token-based similarity scoring.
       - **Scikit-learn**: Vectorization and cosine similarity computation.
       - **SentenceTransformer**: Pre-trained models for semantic similarity.
    2. **Text Preprocessing**:
       - Uses regular expressions to clean product names and remove unwanted terms.
       - Generates Soundex codes for phonetic matching.
    3. **Cloud Storage**:
       - Integrates with Dropbox for persistent storage and retrieval of product databases.
    4. **Streamlit**:
       - Provides an interactive and dynamic user interface.

    ## Use Cases
    1. **Regulatory Compliance**:
       - Avoid registering new product names that conflict with existing ones to ensure legal and regulatory adherence.
    2. **Pharmaceutical Market**:
       - Identify lookalike or soundalike names in drug registration databases.
    3. **Product Management**:
       - Manage large product catalogs by detecting and preventing duplicate entries.

    ## Advantages
    1. **Flexible Detection Options**:
       - Offers three similarity detection methods to suit different needs.
    2. **Real-Time Updates**:
       - Automatically refreshes the database view after updates or uploads.
    3. **Robust Preprocessing**:
       - Handles noisy and inconsistent data effectively.
    4. **Secure Cloud Storage**:
       - Uses Dropbox to ensure data is accessible and safe.

    ## Future Enhancements
    1. Add support for additional cloud storage options like AWS S3 or Google Drive.
    2. Enable batch comparisons for multiple product names simultaneously.
    3. Develop domain-specific similarity algorithms for improved accuracy.
    """)

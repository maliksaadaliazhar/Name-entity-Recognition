import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import pickle
import re
import pandas as pd

# Set page config for a wider, cleaner layout
st.set_page_config(page_title="NER Deep Learning App", layout="centered")

# --- CACHING THE MODEL & DICTIONARY ---
# This ensures the 83MB model only loads once and stays in memory!
@st.cache_resource
def load_assets():
    # Load the Keras model
    model = load_model('NER_BiLSTM_model.keras')
    
    # Load the dictionary
    with open('words_to_index.pkl', 'rb') as f:
        word_to_index = pickle.load(f)
        
    return model, word_to_index

# --- PREDICTION LOGIC ---
def predict_entities(text, model, word_to_index, max_len=113):
    # Upgraded tokenization: cleanly separates words and punctuation!
    tokens = re.findall(r"[\w']+|[.,!?;]", text)
    
    # Convert to lowercase and find indices
    indices = []
    for w in tokens:
        w_lower = w.lower()
        if w_lower in word_to_index:
            indices.append(word_to_index[w_lower])
        else:
            indices.append(1) # <UNK> token
            
    # Pad the sequence
    padded_indices = np.zeros((1, max_len), dtype='int32')
    padded_indices[0, :len(indices)] = indices[:max_len]
    
    # Run the model prediction
    predictions = model.predict(padded_indices)
    pred_tags_indices = np.argmax(predictions[0], axis=-1)
    
    # Tag dictionary
    tag_map = {
        0: 'O', 1: 'B-PER', 2: 'I-PER', 
        3: 'B-ORG', 4: 'I-ORG', 5: 'B-LOC', 
        6: 'I-LOC', 7: 'B-MISC', 8: 'I-MISC'
    }
    
    # Pair tokens with their predicted tags
    results = []
    for i in range(min(len(tokens), max_len)):
        results.append({
            "Token": tokens[i],
            "Tag": tag_map.get(pred_tags_indices[i], 'O')
        })
        
    return results

# --- STREAMLIT UI ---
st.title("Named Entity Recognition (NER)")
st.write("This application uses a custom Many-to-Many Bidirectional LSTM trained on the CoNLL-2003 dataset to extract Organizations, People, and Locations from raw text.")

# Load our massive assets behind the scenes
with st.spinner("Loading AI Model (This takes a few seconds on startup)..."):
    model, word_to_index = load_assets()

# User Input
st.subheader("Enter your text below:")
user_input = st.text_area(
    "Type a sentence...", 
    value="Apple CEO Tim Cook is traveling to New York City tomorrow.",
    height=100
)

# When the user clicks the button
if st.button("Analyze Text", type="primary"):
    if user_input.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing..."):
            # Get the predictions
            results = predict_entities(user_input, model, word_to_index)
            
            # Convert to a Pandas DataFrame for a beautiful table UI
            df = pd.DataFrame(results)
            
            st.success("Analysis Complete!")
            
            # Create two columns for the output layout
            col1, col2 = st.columns([1.5, 1])
            
            with col1:
                st.subheader("Data Table")
                # Display the dataframe interactively
                st.dataframe(df, use_container_width=True, hide_index=True)
                
            with col2:
                st.subheader("Entity Summary")
                # Count and display how many entities were found (ignoring 'O')
                entities_found = df[df['Tag'] != 'O']
                
                if entities_found.empty:
                    st.info("No Named Entities found in this text.")
                else:
                    st.write(f"**Total Entities Found:** {len(entities_found)}")
                    for index, row in entities_found.iterrows():
                        # Add some visual color cues using Streamlit Markdown
                        tag = row['Tag']
                        if 'PER' in tag:
                            st.markdown(f"**{row['Token']}**  `{tag}`")
                        elif 'ORG' in tag:
                            st.markdown(f"**{row['Token']}**  `{tag}`")
                        elif 'LOC' in tag:
                            st.markdown(f"**{row['Token']}**  `{tag}`")
                        else:
                            st.markdown(f"**{row['Token']}**  `{tag}`")
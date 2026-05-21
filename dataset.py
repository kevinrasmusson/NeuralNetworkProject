import html
import re

import pandas as pd
from bs4 import BeautifulSoup


def remove_html(html_text):
    soup = BeautifulSoup(str(html_text), "html.parser")
    return soup.get_text(separator=" ")


def clean_text(text):
    text = html.unescape(str(text))
    text = text.lower()

    text = text.replace("–", " ")
    text = text.replace("—", " ")
    text = text.replace('"', " ")
    text = text.replace("”", " ")
    text = text.replace("“", " ")

    # Replace all numbers with a shared number token
    text = re.sub(r"\d+", " numtoken ", text)

    # Keep Swedish characters and basic punctuation
    text = re.sub(r"[^a-zA-ZåäöÅÄÖéÉ0-9.,!?]+", " ", text)

    text = re.sub(r"([.,!?])", r" \1 ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def load_data(filepath):
    df = pd.read_csv(filepath)

    df.columns = df.columns.str.strip()

    #print("Columns in dataset:")
    #print(df.columns)

    headline_col = "headline"
    body_col = "regexp_replace"

    required_columns = [headline_col, body_col]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}. Found columns: {list(df.columns)}")

    # Remove missing values
    df = df.dropna(subset=[headline_col, body_col])

    # Make sure columns are strings
    df[headline_col] = df[headline_col].astype(str)
    df[body_col] = df[body_col].astype(str)

    # Remove empty rows
    df = df[df[headline_col].str.strip() != ""]
    df = df[df[body_col].str.strip() != ""]

    # Remove HTML tags from body
    df["body_clean"] = df[body_col].apply(remove_html)

    # Combine headline and body
    df["text"] = df[headline_col] + " " + df["body_clean"]

    # Clean combined text
    df["text_clean"] = df["text"].apply(clean_text)
    df["num_tokens"] = df["text_clean"].apply(lambda x: len(x.split()))
    df = df[df["num_tokens"] >= 6]

    # Remove rows where cleaned text is empty
    df = df[df["text_clean"].str.strip() != ""]

    #print(df[["text", "text_clean"]].head())
    #print("Dataset shape after loading:", df.shape)

    return df
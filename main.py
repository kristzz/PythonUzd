import nltk
import string
import re
import spacy
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag, ne_chunk
from nltk import FreqDist
from langdetect import detect, LangDetectException
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from collections import Counter
import requests
from dotenv import load_dotenv
from googletrans import Translator
import os

load_dotenv()

API_KEY = os.getenv("HUGGINGFACE_API_KEY")
headers = {"Authorization": f"Bearer {API_KEY}"}

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('maxent_ne_chunker_tab')
nltk.download('words')
nltk.download('vader_lexicon')

nlp = spacy.load("en_core_web_sm")  

def translate_text(text):
    translator = Translator()
    try:
        # Translate the text from Latvian ('lv') to English ('en')
        translation = translator.translate(text, src='lv', dest='en')
        return translation.text
    except Exception as e:
        return f"Error occurred during translation: {str(e)}"

def generate_story(prompt, model="gpt2", max_length=150):
    url = f"https://api-inference.huggingface.co/models/{model}"
    
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_length": max_length,
            "num_return_sequences": 1
        }
    }

    response = requests.post(url, headers=headers, json=payload)
    
    if response.status_code == 200:
        output = response.json()
        generated_text = output[0]['generated_text']
        return generated_text
    else:
        return f"Error: {response.status_code} - {response.text}"

def count_word_frequency():
    text = input("Enter your text: ")
    text = text.lower()
    words = word_tokenize(text)
    words = [word for word in words if word not in string.punctuation]
    word_counts = Counter(words)
    print("\nWord Frequency Analysis:")
    for word, count in word_counts.items():
        print(f"{word}: {count}")

def detect_language():
    text = input("Enter your text: ")
    try:
        language = detect(text)
        print(f"\nDetected language: {language}")
    except LangDetectException as e:
        print(f"Could not detect language. Error: {str(e)}")

def word_match_percentage(text1, text2):
    words1 = set(word_tokenize(text1.lower()))
    words2 = set(word_tokenize(text2.lower()))
    
    words1 = {word for word in words1 if word not in string.punctuation and word.strip()}
    words2 = {word for word in words2 if word not in string.punctuation and word.strip()}
    
    common_words = words1.intersection(words2)
    
    total_unique_words = len(words1.union(words2))
    match_percentage = (len(common_words) / total_unique_words) * 100 if total_unique_words > 0 else 0

    print(f"\nCommon words: {common_words}")
    print(f"Match Percentage: {match_percentage:.2f}%")

def sentiment_analysis():
    sia = SentimentIntensityAnalyzer()

    text = input("Enter your sentence(s): ")

    sentiment_score = sia.polarity_scores(text)
    compound_score = sentiment_score['compound']

    if compound_score >= 0.05:
        sentiment = "Positive"
    elif compound_score <= -0.05:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    
    print(f"\nSentiment Analysis Result: {sentiment}")
    print(f"Sentiment Score: {sentiment_score}")

def clean_and_normalize_text():
    raw_text = input("Enter the raw text: ")
    
    text = re.sub(r'http\S+', '', raw_text)
    text = re.sub(r'[@#]', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    
    text = text.lower()
    text = ' '.join(text.split())
    
    print("\nCleaned and Normalized Text:")
    print(text)

def summarize_text():
    article = input("Enter the article: ")
    
    sentences = sent_tokenize(article)
    
    stop_words = set(stopwords.words("english"))
    words = word_tokenize(article.lower())
    words = [word for word in words if word not in stop_words and word not in string.punctuation]
    
    word_freq = Counter(words)
    max_freq = max(word_freq.values())
    for word in word_freq.keys():
        word_freq[word] /= max_freq

    sentence_scores = {}
    for sentence in sentences:
        sentence_words = word_tokenize(sentence.lower())
        sentence_score = sum(word_freq.get(word, 0) for word in sentence_words)
        sentence_scores[sentence] = sentence_score
    
    summarized_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:3]
    
    print("\nSummary:")
    for sentence in summarized_sentences:
        print(sentence)

def word_embeddings():
    words = input("Enter words separated by commas (e.g., māja, dzīvoklis, jūra): ").split(',')
    words = [word.strip() for word in words]
    
    print("\nWord Embeddings (Vector Previews):")
    for word in words:
        token = nlp(word)
        print(f"Vector for '{word}': {token.vector[:5]}...")
    print("\nSemantic Similarity Between Words:")
    for i in range(len(words)):
        for j in range(i + 1, len(words)):
            word1, word2 = words[i], words[j]
            similarity = nlp(word1).similarity(nlp(word2))
            print(f"Similarity between '{word1}' and '{word2}': {similarity:.4f}")

def named_entity_recognition():
    text = input("Enter the text: ")
    
    words = word_tokenize(text)
    pos_tags = pos_tag(words)
    
    chunked = ne_chunk(pos_tags)
    
    print("\nNamed Entities (Person and Organization):")
    for subtree in chunked:
        if isinstance(subtree, nltk.Tree):
            label = subtree.label()
            entity = " ".join([word for word, tag in subtree])
            if label == "PERSON":
                print(f"Person: {entity}")
            elif label == "GPE" or label == "ORG":
                print(f"Organization: {entity}")

def main_menu():
    print("\nChoose an action:")
    print("1. Perform Word Frequency Analysis")
    print("2. Detect Language of Text")
    print("3. Check Word Match Between Two Texts")
    print("4. Perform Sentiment Analysis (only works in English)")
    print("5. Clean and Normalize Text")
    print("6. Summarize Text")
    print("7. Compute Word Embeddings and Similarity (only works in English)")
    print("8. Perform Named Entity Recognition (Person and Organization)")
    print("9. Generate Text (Story or continuation of a prompt)")
    print("10. Translate Text from Latvian to English")
    print("11. Exit")
    
    choice = input("Enter your choice: ")
    
    match choice:
        case "1":
            count_word_frequency()
        case "2":
            detect_language()
        case "3":
            text1 = input("Enter Text 1: ")
            text2 = input("Enter Text 2: ")
            word_match_percentage(text1, text2)
        case "4":
            sentiment_analysis()
        case "5":
            clean_and_normalize_text()
        case "6":
            summarize_text()
        case "7":
            word_embeddings()
        case "8":
            named_entity_recognition()
        case "9":
            prompt = input("Enter a starting phrase for the story: ")
            generated_story = generate_story(prompt, model="gpt2", max_length=150)
            print("\nGenerated Story:")
            print(generated_story)
        case "10":
            text_to_translate = input("Enter the Latvian text to translate: ")
            translated_text = translate_text(text_to_translate)
            print(f"Translated Text: {translated_text}")
        case "11":
            print("Exiting the program.")
            exit()
        case _:
            print("Invalid choice! Please try again.")

while True:
    main_menu()
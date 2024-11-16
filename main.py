import nltk
from nltk.tokenize import word_tokenize
from collections import Counter
import string
from langdetect import detect, LangDetectException
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('punkt')
nltk.download('vader_lexicon')

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

def main_menu():
    print("\nChoose an action:")
    print("1. Perform Word Frequency Analysis")
    print("2. Detect Language of Text")
    print("3. Check Word Match Between Two Texts")
    print("4. Perform Sentiment Analysis")
    print("5. Exit")
    
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
            print("Exiting the program.")
            exit()
        case _:
            print("Invalid choice! Please try again.")

while True:
    main_menu()

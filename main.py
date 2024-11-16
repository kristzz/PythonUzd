import nltk
from nltk.tokenize import word_tokenize
from collections import Counter
import string
from langdetect import detect, LangDetectException

nltk.download('punkt')

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

def main_menu():
    print("\nChoose an action:")
    print("1. Perform Word Frequency Analysis")
    print("2. Detect Language of Text")
    print("3. Exit")
    choice = input("Enter your choice: ")
    match choice:
        case "1":
            count_word_frequency()
        case "2":
            detect_language()
        case "3":
            print("Exiting the program.")
            exit()
        case _:
            print("Invalid choice! Please try again.")

while True:
    main_menu()
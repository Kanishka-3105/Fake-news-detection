import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

class FakeNewsDetector:
    def __init__(self):
        self.classifier = MultinomialNB()
        self.vectorizer = TfidfVectorizer(stop_words='english')

    def preprocess_text(self, text):
        """
        Preprocesses input text by lowercasing and removing special characters/numbers.
        """
        text = str(text).lower()  # Convert to lowercase
        text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove non-alphabetic characters
        return text

    def load_and_analyze_csv(self, csv_path):
        """
        Loads dataset, preprocesses text, trains a model, evaluates it, and saves predictions.
        """
        # Load dataset
        df = pd.read_csv(csv_path)
        
        # Ensure required columns exist
        if 'text' not in df.columns or 'label' not in df.columns:
            raise ValueError("Dataset must contain 'text' and 'label' columns.")
        
        # Preprocess text
        df['processed_text'] = df['text'].apply(self.preprocess_text)
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            df['processed_text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
        )
        
        # Vectorize text data
        X_train_vectorized = self.vectorizer.fit_transform(X_train)
        X_test_vectorized = self.vectorizer.transform(X_test)
        
        # Train Naive Bayes classifier
        self.classifier.fit(X_train_vectorized, y_train)
        
        # Evaluate the model
        y_pred = self.classifier.predict(X_test_vectorized)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, zero_division=1)
        
        print(f"Model Accuracy: {accuracy:.2f}")
        print("\nClassification Report:\n", report)
        
        # Analyze predictions for the entire dataset
        results = []
        for _, row in df.iterrows():
            text_vectorized = self.vectorizer.transform([row['processed_text']])
            prediction = self.classifier.predict(text_vectorized)[0]
            confidence = max(self.classifier.predict_proba(text_vectorized)[0]) * 100
            
            results.append({
                'text': row['text'],
                'actual_label': row['label'],
                'predicted_label': prediction,
                'confidence': f"{confidence:.2f}%"
            })
        
        # Save predictions to a CSV file
        results_df = pd.DataFrame(results)
        results_df.to_csv('fake_news_predictions.csv', index=False)
        
        # Print first 5 results
        print("\nSample Predictions:")
        print(results_df.head(5).to_string(index=False))
        
        return results_df

def main():
    detector = FakeNewsDetector()
    try:
        
        results = detector.load_and_analyze_csv('fake-news-dataset.csv')
        print("Analysis saved to 'fake_news_predictions.csv'.")
    except FileNotFoundError:
        print("Error: The dataset file was not found. Please check the file path.")
    except ValueError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
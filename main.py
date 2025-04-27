import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import joblib
def train_email_classifier():
    """
    Trains a model to classify emails as personal or commercial.
    """
    # Sample dataset
    emails = [
        "john.doe@gmail.com", "jane.smith@yahoo.com", "info@company.com",
        "support@business.org", "contact@startup.net", "user@hotmail.com"
    ]
    labels = ["personal", "personal", "commercial", "commercial", "commercial", "personal"]

    # Convert emails to feature vectors
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(emails)
    y = labels

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Naive Bayes classifier
    classifier = MultinomialNB()
    classifier.fit(X_train, y_train)

    # Save the trained model and vectorizer
    joblib.dump(classifier, "email_classifier.pkl")
    joblib.dump(vectorizer, "email_vectorizer.pkl")

    print("Model trained and saved successfully.")

# Uncomment the following line to train the model when running this script
# train_email_classifier()
def validate_email(email):
    """
    Validates an email address using a regular expression.
    
    Args:
        email (str): The email address to validate.
    
    Returns:
        bool: True if the email is valid, False otherwise.
    """
    email_regex = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(email_regex, email) is not None

# Example usage
if __name__ == "__main__":
    test_email = "8080.com"
    if validate_email(test_email):
        print(f"{test_email} is a valid email address.")
    else:
        print(f"{test_email} is not a valid email address.")
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import joblib 
import pandas as pd
def train_email_classifier():
    """
    Trains a model to classify emails as personal or commercial.
    """

    # Sample dataset
    # email_domain = [
    #     "gmail.com", "yahoo.com", "amazon.com",
    #     "google.com", "microsoft.com", "hotmail.com",
    #     "outlook.com","icloud.com","aol.com"
    # ]
    # labels = ["personal", "personal", "commercial", 
    #           "commercial", "commercial", "personal",
    #           "personal","personal","personal"]
    # Read email domains and labels from CSV
    data = pd.read_csv("email_domain.csv")
    email_domain = data['domain'].tolist()
    labels = data['label'].tolist()



    # Convert emails to feature vectors
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(email_domain)
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
    email_regex = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(email_regex, email) is not None

# Example usage
if __name__ == "__main__":
    test_email = "anish@80.com"
    if validate_email(test_email):
        print(f"{test_email} is a valid email address.")
    else:
        print(f"{test_email} is not a valid email address.")


    train_email_classifier()
    # Load the trained model and vectorizer
    classifier = joblib.load("email_classifier.pkl")
    vectorizer = joblib.load("email_vectorizer.pkl")
    # Example email to classify

    # Read email addresses from CSV
    email_data = pd.read_csv("email_address_list.csv")
    email_to_classify = email_data['Email_Address'].tolist()
    for email in email_to_classify:
        email_to_classify = [email.split('@')[-1]]    
        email_features = vectorizer.transform(email_to_classify)
        prediction = classifier.predict(email_features)
        print(f"The email '{email_to_classify[0]}' is classified as: {prediction[0]}")
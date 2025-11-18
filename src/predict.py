import sys
import joblib

LABELS = {
    0: "HAM",
    1: "SPAM"
}

def load_model():
    model = joblib.load("models/best_model.pkl")
    vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
    return model, vectorizer

def predict_message(message: str):
    model, vectorizer = load_model()
    X = vectorizer.transform([message])
    pred = model.predict(X)[0]
    return LABELS[int(pred)]

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py \"your message here\"")
        sys.exit(1)

    message = " ".join(sys.argv[1:])
    result = predict_message(message)
    print(result)

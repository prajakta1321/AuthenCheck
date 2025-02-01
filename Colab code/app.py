from flask import Flask, request, jsonify
from transformers import BertTokenizer, BertForSequenceClassification
import torch

app = Flask(__name__)

# Load your fine-tuned BERT model and tokenizer
model_path = "./fine_tuned_bert"  # Path to your saved model
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data['text']

    # Preprocess the text
    inputs = tokenizer(text, return_tensors="pt")

    # Make prediction
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_class = torch.argmax(outputs.logits).item()

    # Get label for predicted class (if using LabelEncoder)
    # label = label_encoder.inverse_transform([predicted_class])[0]

    # Return the prediction
    return jsonify({'predicted_class': predicted_class}) #, 'label': label})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

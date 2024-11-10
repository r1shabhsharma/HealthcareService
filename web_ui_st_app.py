
import streamlit as st
import docx2txt
import textwrap
import speech_recognition as sr
from datetime import datetime
from huggingface_hub import InferenceClient
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer
from datasets import load_dataset, Dataset
import torch
from docx import Document # pip install python-docx
import time

import os
 
folder_path = "fine_tuned_healthcare_faq_model"


# Initialize the InferenceClient with your Hugging Face token
api_token = "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"  # replace with your actual token
PhiClient = InferenceClient(model="microsoft/Phi-3.5-mini-instruct", token=api_token)
SentimentClient = InferenceClient(model="bhadresh-savani/distilbert-base-uncased-emotion", token=api_token)

# Load the pre-trained model and tokenizer for question answering
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# Function to load the dataset and flatten it
def load_and_flatten_dataset():
    dataset = load_dataset('json', data_files='healthcare_faq.json')
    flattened_samples = []
    for paragraph in dataset['train'][0]['paragraphs']:
        for qa in paragraph['qas']:
            flattened_samples.append({
                "context": paragraph["context"],
                "question": qa["question"],
                "answer": qa["answer"]
            })
    return Dataset.from_dict({
        "context": [item["context"] for item in flattened_samples],
        "question": [item["question"] for item in flattened_samples],
        "answer": [item["answer"] for item in flattened_samples]
    })

# Preprocessing function
def preprocess_function(examples):
    inputs = tokenizer(examples['question'], examples['context'], truncation=True, padding='max_length', max_length=512)
    
    start_positions = []
    end_positions = []
    
    for i in range(len(examples['answer'])):
        answer = examples['answer'][i]
        context = examples['context'][i]
        start_pos = context.find(answer)
        end_pos = start_pos + len(answer) - 1
        start_positions.append(start_pos)
        end_positions.append(end_pos)
    
    inputs['start_positions'] = start_positions
    inputs['end_positions'] = end_positions
    
    return inputs

# Train the model
def fine_tune_model():
    flattened_dataset = load_and_flatten_dataset()
    tokenized_dataset = flattened_dataset.map(preprocess_function, batched=True)

    final_dataset = tokenized_dataset.train_test_split(test_size=0.2)

    training_args = TrainingArguments(
        output_dir='./results',
        evaluation_strategy="steps",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        save_steps=10,
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=final_dataset['train'],
        eval_dataset=final_dataset['test'],
        tokenizer=tokenizer
    )

    trainer.train()

    trainer.save_model("fine_tuned_healthcare_faq_model")
    tokenizer.save_pretrained("fine_tuned_healthcare_faq_model")

# Define Q&A bot function
def qna_bot(context, reference_answers=None):
    # Load the fine-tuned model and tokenizer
    model = AutoModelForQuestionAnswering.from_pretrained("fine_tuned_healthcare_faq_model")
    tokenizer = AutoTokenizer.from_pretrained("fine_tuned_healthcare_faq_model")

    question = st.text_input("Enter your question:")
    predictions = []
    if question:
        start_time = time.time() 
        answer = get_answer(context, question, model, tokenizer)
        end_time = time.time() 
        latency = end_time - start_time
        st.write(f"Latency: {latency:.4f} seconds") 
        st.write(f"Answer: {answer}")
        predictions.append(answer)
        if reference_answers:
            evaluate_metrics(predictions, reference_answers)


def get_answer(context, question, model, tokenizer):
    inputs = tokenizer(question, context, return_tensors='pt')
    start_scores, end_scores = model(**inputs)

    start_index = torch.argmax(start_scores)
    end_index = torch.argmax(end_scores)

    answer = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][start_index:end_index+1])
    )
    return print_wrapped_text(answer)


# Sentiment Analysis
def detect_sentiment(statement):
    output = SentimentClient.text_classification(statement)
    sentiment_results = "\nPatient sentiment analysis:\n"
    for sentiment in output:
        sentiment_results += f"{sentiment.label}: {sentiment.score * 100:.2f}%\n"
    return sentiment_results

# Summarizer
def summarizer(context):
    prompt = f"Summarize the following patient notes:\n\n{context}\n\nSummary:"
    output = PhiClient.text_generation(prompt, max_new_tokens=300, temperature=0.3)
    return f"Summary of the patient details:\n{print_wrapped_text(output)}"

# Helper function to wrap text
def print_wrapped_text(text, width=125):
    wrapped_text = textwrap.fill(text, width=width)
    return wrapped_text

# Speech-to-text function
def record_audio():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Listening for your speech...")
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            st.write(f"Recognized text: {text}")
            return text
        except sr.UnknownValueError:
            st.write("Sorry, I couldn't understand the audio.")
            return None
        except sr.RequestError:
            st.write("Could not request results from Google Speech Recognition service.")
            return None

# Update patient notes in docx
def update_patient_notes(doc_path, new_notes):
    doc = Document(doc_path)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    doc.add_paragraph(f"Updated on {timestamp}:\n{new_notes}\n")
    doc.save(doc_path)

# Streamlit Interface
st.title("Healthcare Assistant Tool")

option = st.sidebar.selectbox("Choose an option:", 
                              ["Select an option", "Get Question Answering chatbot", 
                               "Detect sentiments from patient feedback", 
                               "Summarize patient details", 
                               "Fine-tune model", 
                               "Update Patient Notes"])

# Handle the different options
if option == "Get Question Answering chatbot":
    st.write("### Upload a Word document containing the context for Q&A")
    uploaded_file = st.file_uploader("Upload Document", type=["docx"])
    if uploaded_file:
        context = docx2txt.process(uploaded_file)
        qna_bot(context)

elif option == "Detect sentiments from patient feedback":
    feedback = st.text_area("Enter patient feedback:")
    if feedback:
        sentiment_analysis = detect_sentiment(feedback)
        st.write(sentiment_analysis)

elif option == "Summarize patient details":
    st.write("### Upload a Word document containing patient history and details")
    uploaded_file = st.file_uploader("Upload Document", type=["docx"])
    if uploaded_file:
        context = docx2txt.process(uploaded_file)
        summary = summarizer(context)
        st.write(summary)

elif option == "Fine-tune model":
    st.write("### Fine-tuning model on healthcare FAQ dataset")
    fine_tune_model()
    st.write("Fine-tuning complete! The model has been saved.")

elif option == "Update Patient Notes":
    st.write("### Select the patient's document and add speech notes")
    uploaded_patient_doc = st.file_uploader("Upload Patient Document", type=["docx"])
    if uploaded_patient_doc:
        new_notes = record_audio()
        if new_notes:
            update_patient_notes(uploaded_patient_doc, new_notes)
            st.write("Patient notes updated successfully.")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Load the dataset
df = pd.read_csv('tweets.csv')
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['tweet_text'], df['cyberbullying_type'], random_state=0)
# Preprocess the training set
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess(text):
    # Tokenize the text
    words = word_tokenize(text.lower())

    # Remove stop words
    words = [word for word in words if word not in stop_words]

    # Stem the words
    words = [stemmer.stem(word) for word in words]

    # Join the words back into a string
    return ' '.join(words)

X_train = X_train.apply(preprocess)
X_test = X_test.apply(preprocess)

# Extract features from the data
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Train the machine learning algorithm
clf = MultinomialNB()
clf.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = clf.predict(X_test)

# Evaluate the performance of the system
#print('Accuracy:', accuracy_score(y_test, y_pred))

def is_cyberbullying(message):
    # Preprocess and extract features from the message
    message = preprocess(message)
    message_features = vectorizer.transform([message])

    # Predict the label of the message
    message_pred = clf.predict(message_features)

    # Return True if it's classified as cyberbullying, False otherwise
    message_pred = "not_cyberbullying" if message_pred[0] == "not_cyberbullying" else "cyberbullying"
    return message_pred



#voice input method

import speech_recognition as sr

# Function to perform speech recognition
def get_voice_input():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Say something...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
        print("Recognizing...")
    try:
        message = recognizer.recognize_google(audio)
        return message
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio.")
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))
    return None



from flask import Flask, render_template, request, session, redirect, url_for
from flask_socketio import join_room, leave_room, send, SocketIO
import random
from string import ascii_uppercase
import base64
import os
import uuid

app = Flask(__name__)
app.config["SECRET_KEY"] = "hjhjsdahhds"
socketio = SocketIO(app)

rooms = {}

def generate_unique_code(length):
    while True:
        code = ""
        for _ in range(length):
            code += random.choice(ascii_uppercase)

        if code not in rooms:
            break

    return code

@app.route("/", methods=["POST", "GET"])
def home():
    session.clear()
    if request.method == "POST":
        name = request.form.get("name")
        code = request.form.get("code")
        join = request.form.get("join", False)
        create = request.form.get("create", False)

        if not name:
            return render_template("home.html", error="Please enter a name.", code=code, name=name)

        if join != False and not code:
            return render_template("home.html", error="Please enter a room code.", code=code, name=name)

        room = code
        if create != False:
            room = generate_unique_code(4)
            rooms[room] = {"members": 0, "messages": []}
        elif code not in rooms:
            return render_template("home.html", error="Room does not exist.", code=code, name=name)

        session["room"] = room
        session["name"] = name
        return redirect(url_for("room"))

    return render_template("home.html")

@app.route("/room")
def room():
    room = session.get("room")
    if room is None or session.get("name") is None or room not in rooms:
        return redirect(url_for("home"))

    return render_template("room.html", code=room, messages=rooms[room]["messages"])

#text
@socketio.on("message")
def message(data):
    room = session.get("room")
    if room not in rooms:
        return

    message_text = data["data"]
    res = is_cyberbullying(message_text)
    
    if res == "not_cyberbullying":
        content = {
            "name": session.get("name"),
            "message": message_text
        }
        rooms[room]["messages"].append(content)  # Add the message to the room's messages
        send(content, to=room)  # Send the message to the other party
        print(f"{session.get('name')} said: {message_text}")
    else:
        message_text = "Your message was classified as cyberbullying and will not be sent!"
        content = {
            "name": session.get("name"),
            "message": message_text
        }
        rooms[room]["messages"].append(content)  # Add the message to the room's messages
        send(content)  # Send the message to the other party
        print(f"{session.get('name')} said: {message_text}")


#image
@socketio.on("image1")
def message(data):
    room = session.get("room")
    if room not in rooms:
        return

    content = {
        "name": session.get("name"),
        "message": data["data"]
    }
    send(content, to=room)
    rooms[room]["messages"].append(content)
    print(f"{session.get('name')} said: {data['data']}")


#image
@socketio.on("image2")
def message(data):
    room = session.get("room")
    if room not in rooms:
        return

    content = {
        "name": session.get("name"),
        "message": data["data"]
    }
    send(content)
    rooms[room]["messages"].append(content)
    print(f"{session.get('name')} said: {data['data']}")




@socketio.on("connect")
def connect(auth):
    room = session.get("room")
    name = session.get("name")
    if not room or not name:
        return
    if room not in rooms:
        leave_room(room)
        return

    join_room(room)
    send({"name": name, "message": "has entered the room"}, to=room)
    rooms[room]["members"] += 1
    print(f"{name} joined room {room}")

@socketio.on("disconnect")
def disconnect():
    room = session.get("room")
    name = session.get("name")
    leave_room(room)

    if room in rooms:
        rooms[room]["members"] -= 1
        if rooms[room]["members"] <= 0:
            del rooms[room]

    send({"name": name, "message": "has left the room"}, to=room)
    print(f"{name} has left the room {room}")

if __name__ == "__main__":
    socketio.run(app, debug=True)

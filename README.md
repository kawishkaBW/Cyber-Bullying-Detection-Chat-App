# Live Chat App with Cyberbullying Detection
This is a simple web-based chat application that incorporates cyberbullying detection using machine learning techniques. The application supports text, voice, and image inputs for communication between users.

## Features
- **Text Chat:** Users can communicate with each other via text messages in real-time.
- **Voice Input:** Users can input messages using their voice, which are then transcribed and processed.
- **Image Sharing:** Users can share images within the chat room.
- **Cyberbullying Detection:** Messages are analyzed for potential cyberbullying content using a machine learning model.
- **Multiple Rooms:** Users can create or join different chat rooms.

## Technologies Used
- **Python:** Backend server and machine learning model implementation.
- **Flask:** Web framework used for routing and handling HTTP requests.
- **Flask-SocketIO:** Integration for WebSocket communication in Flask.
- **HTML/CSS:** Frontend design and layout.
- **JavaScript:** Client-side scripting for real-time interactions.
- **scikit-learn:** Machine learning library for building and training the cyberbullying detection model.
- **NLTK:** Natural Language Toolkit for text preprocessing.

## Setup
1. Clone the repository to your local machine: `git clone https://github.com/kawishkaBW/Cyber-Bullying-Detection-Chat-App.git`
2. Install the required Python packages: `pip install -r requirements.txt`
3. Run the Flask server: `python main.py`
4. Open your web browser and navigate to `http://localhost:5000` to access the chat application.

## Usage
- Upon accessing the application, users can either join an existing chat room by entering the room code or create a new room.
- Users can send messages by typing in the text box and pressing enter, or by speaking into their microphone (voice input).
- Images can be shared by clicking the image upload button and selecting an image file.
- The application automatically detects potential cyberbullying content in messages and notifies users accordingly.

## Contributors
- Kawishka Buddhi
- Hirantha Ranathunga
- Uditha Jayalath
- Niruni Karunanayaka
- Rahal Mahawaththa

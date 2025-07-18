# app.py - Flask Backend
from flask import Flask, render_template, request, jsonify, session
from flask_socketio import SocketIO, emit, join_room, leave_room
import cv2
import numpy as np
import pytesseract
import threading
import time
from PIL import Image, ImageGrab
import re
from googletrans import Translator
import queue
import json
import os
from datetime import datetime
import base64
import io
import uuid
import logging
from werkzeug.utils import secure_filename

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-change-this'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global variables
active_sessions = {}
translator = Translator()

class ChatAnalyzer:
    def __init__(self, session_id):
        self.session_id = session_id
        self.chat_queue = queue.Queue()
        self.previous_chat = []
        self.is_running = False
        self.capture_thread = None
        self.settings = {
            'chat_area': {'x': 100, 'y': 400, 'width': 600, 'height': 200},
            'capture_interval': 2.0,
            'target_language': 'en',
            'confidence_threshold': 60,
            'filter_duplicates': True
        }
        
    def extract_text_from_image(self, image):
        """Extract text from image using OCR"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Apply image processing to improve OCR accuracy
            gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=0)
            
            # Apply threshold
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Use pytesseract to extract text
            custom_config = r'--oem 3 --psm 6'
            text = pytesseract.image_to_string(thresh, config=custom_config)
            
            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting text: {e}")
            return ""
            
    def process_chat_text(self, text):
        """Process and filter chat text"""
        if not text:
            return []
            
        lines = text.split('\n')
        chat_messages = []
        
        for line in lines:
            line = line.strip()
            if len(line) < 3:
                continue
                
            # Chat message patterns
            chat_patterns = [
                r'^(.+?):\s*(.+)$',  # PlayerName: message
                r'^\[(.+?)\]\s*(.+)$',  # [PlayerName] message
                r'^<(.+?)>\s*(.+)$',  # <PlayerName> message
            ]
            
            for pattern in chat_patterns:
                match = re.match(pattern, line)
                if match:
                    player = match.group(1).strip()
                    message = match.group(2).strip()
                    chat_messages.append({
                        'player': player,
                        'message': message,
                        'timestamp': datetime.now().strftime('%H:%M:%S')
                    })
                    break
                    
        return chat_messages
        
    def translate_text(self, text, target_lang):
        """Translate text to target language"""
        try:
            if not text or len(text.strip()) < 2:
                return text
                
            detected = translator.detect(text)
            if detected.lang == target_lang:
                return text
                
            result = translator.translate(text, dest=target_lang)
            return result.text
        except Exception as e:
            logger.error(f"Translation error: {e}")
            return text
            
    def process_uploaded_image(self, image_data):
        """Process uploaded image for chat analysis"""
        try:
            # Decode base64 image
            image_bytes = base64.b64decode(image_data.split(',')[1])
            image = Image.open(io.BytesIO(image_bytes))
            image_array = np.array(image)
            
            # Extract text
            text = self.extract_text_from_image(image_array)
            
            if text:
                chat_messages = self.process_chat_text(text)
                
                # Process new messages
                new_messages = []
                for msg in chat_messages:
                    msg_key = f"{msg['player']}:{msg['message']}"
                    if msg_key not in self.previous_chat:
                        # Translate message
                        translated_msg = self.translate_text(msg['message'], self.settings['target_language'])
                        
                        msg_data = {
                            'player': msg['player'],
                            'original': msg['message'],
                            'translated': translated_msg,
                            'timestamp': msg['timestamp'],
                            'needs_translation': translated_msg != msg['message']
                        }
                        
                        new_messages.append(msg_data)
                        self.previous_chat.append(msg_key)
                
                # Keep only recent messages
                if len(self.previous_chat) > 50:
                    self.previous_chat = self.previous_chat[-25:]
                
                return new_messages
                
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            
        return []

# Routes
@app.route('/')
def index():
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    return render_template('index.html')

@app.route('/api/settings', methods=['GET', 'POST'])
def settings():
    session_id = session.get('session_id')
    if not session_id:
        return jsonify({'error': 'No session'}), 400
        
    if session_id not in active_sessions:
        active_sessions[session_id] = ChatAnalyzer(session_id)
    
    analyzer = active_sessions[session_id]
    
    if request.method == 'POST':
        data = request.json
        analyzer.settings.update(data)
        return jsonify({'status': 'success'})
    
    return jsonify(analyzer.settings)

@app.route('/api/process_image', methods=['POST'])
def process_image():
    session_id = session.get('session_id')
    if not session_id:
        return jsonify({'error': 'No session'}), 400
        
    if session_id not in active_sessions:
        active_sessions[session_id] = ChatAnalyzer(session_id)
    
    analyzer = active_sessions[session_id]
    
    data = request.json
    image_data = data.get('image')
    
    if not image_data:
        return jsonify({'error': 'No image data'}), 400
    
    messages = analyzer.process_uploaded_image(image_data)
    
    # Emit messages to connected clients
    for msg in messages:
        socketio.emit('new_message', msg, room=session_id)
    
    return jsonify({'messages': messages})

# Socket.IO events
@socketio.on('connect')
def on_connect():
    session_id = session.get('session_id')
    if session_id:
        join_room(session_id)
        emit('connected', {'session_id': session_id})

@socketio.on('disconnect')
def on_disconnect():
    session_id = session.get('session_id')
    if session_id:
        leave_room(session_id)

@socketio.on('join_session')
def on_join_session(data):
    session_id = data['session_id']
    join_room(session_id)
    emit('joined_session', {'session_id': session_id})

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
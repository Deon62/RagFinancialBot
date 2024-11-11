Deon - AI Financial Assistant
A powerful and intelligent financial assistant that provides personalized financial advice, budget recommendations, and savings guidance with a friendly touch and humor.

Features
Multi-modal interaction (Text and Voice)
Comprehensive financial tracking
Bank accounts
Mobile money
Cash accounts
Savings accounts
Debts and repayments
Personality modes
Normal Mode: Friendly and professional
Roast Mode: Brutally honest with humor
Hype Mode: Ultra-enthusiastic motivation
Voice gender selection (Male/Female)
Chat history tracking
Real-time financial summaries
Goal tracking
Transaction monitoring
Setup
Clone the repository
Install dependencies:
pip install -r requirements.txt

Copy

Execute

Configure environment variables:
GEMINI_API_KEY
SUPABASE_URL
SUPABASE_KEY
Usage
Start the application:

python financial_bot_v2.py

Copy

Execute

Select your preferred:

Interaction mode (Text/Voice)
Voice gender (Male/Female)
Bot personality (Normal/Roast/Hype)
API Endpoints
/chat: Text-based interaction
/voice-chat: Voice-based interaction
/chat-history: Retrieve chat history
/set-mode: Change bot personality
/update-embeddings: Update transaction embeddings
/interact: Multi-modal interaction endpoint
Database Structure
profiles
financial_goals
bank_accounts
mobile_money_accounts
cash_accounts
savings_accounts
debts
debt_repayments
transactions
notifications
chat_history
Technologies
Flask
Supabase
Google Gemini AI
SentenceTransformers
Speech Recognition
Text-to-Speech
License
MIT License


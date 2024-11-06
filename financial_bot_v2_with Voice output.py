
from supabase import create_client, Client
from sentence_transformers import SentenceTransformer
from flask import Flask, request, jsonify
import google.generativeai as genai
from config import GEMINI_API_KEY, SUPABASE_URL, SUPABASE_KEY
import numpy as np
import speech_recognition as sr
from gtts import gTTS
import os
import pygame
import playsound
from flask import send_file


supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

class FinancialBotV2:
    def __init__(self, api_key, model_name="gemini-1.5-flash"):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            model_name=model_name,
            generation_config={
                "temperature": 0.7,
                "top_p": 0.95,
                "top_k": 64,
                "max_output_tokens": 80192,
            }
        )
        self.system_instructions = "You're a financial assistant  known as Deon. your work is to give personalized advice also act as a budget and savings reccomender  to users  with a friendly touch and also add humor."
        self.conversation_window = 5  

    def get_user_profile(self, user_id):
        try:
            profile = supabase.table("user_profiles").select("*").eq('user_id', user_id).execute().data
            return profile[0] if profile else None
        except Exception as e:
            print(f"Error fetching user profile: {str(e)}")
            return None

    def get_financial_goals(self, user_id):
        try:
            return supabase.table("financial_goals").select("*").eq('user_id', user_id).execute().data
        except Exception as e:
            print(f"Error fetching financial goals: {str(e)}")
            return []
    def update_embeddings_for_existing_transactions(self):
        max_retries = 3
        retry_count = 0
    
        while retry_count < max_retries:
            try:
                transactions = supabase.table("transactions").select("*").execute().data
            
                for transaction in transactions:
                    text_to_embed = f"{transaction['category']} {transaction['subcategory']} {transaction['type']}"
                    embedding = embedding_model.encode(text_to_embed).tolist()
                
                    supabase.table("transactions")\
                        .update({"embedding": embedding})\
                        .eq('transaction_id', transaction['transaction_id'])\
                        .execute()
            
                return "Embeddings updated successfully!"
            
            except Exception as e:
                retry_count += 1
                if retry_count == max_retries:
                    print(f"Final attempt failed: {str(e)}")
                    return "Process completed with errors"
                print(f"Retrying... Attempt {retry_count} of {max_retries}")
    def retrieve_relevant_data(self, query, user_id):
        try:
            query_embedding = embedding_model.encode(query).tolist()
            
            # Get user profile and financial goals
            user_profile = self.get_user_profile(user_id)
            financial_goals = self.get_financial_goals(user_id)
            
            # First try semantic search for transactions
            matched_transactions = supabase.rpc(
                'match_transactions',
                {
                    'query_embedding': query_embedding,
                    'match_threshold': 0.7,
                    'match_count': 5
                }
            ).execute().data

            if matched_transactions:
                transaction_ids = [t['transaction_id'] for t in matched_transactions]
                transactions = supabase.table("transactions")\
                    .select("*")\
                    .in_('transaction_id', transaction_ids)\
                    .execute().data
            else:
                transactions = supabase.table("transactions").select("*").execute().data

            return {
                'transactions': transactions,
                'user_profile': user_profile,
                'financial_goals': financial_goals
            }

        except Exception as e:
            print(f"Query error: {str(e)}")
            return {'transactions': [], 'user_profile': None, 'financial_goals': []}
        

        
    def listen_for_command(self):
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            print("Listening...")
            audio = recognizer.listen(source)
            try:
                query = recognizer.recognize_google(audio)
                print(f"You said: {query}")
                return query
            except sr.UnknownValueError:
                print("Could not understand audio")
                return None
            except sr.RequestError:
                print("Could not request results")
                return None
    def speak_response(self, text):
        try:
            from playsound import playsound  
            tts = gTTS(text=text, lang='en')
            temp_file = "temp_response.mp3"
            tts.save(temp_file)
            playsound(temp_file)  
            os.remove(temp_file)
        except Exception as e:
            print(f"Audio playback error: {str(e)}")
            print("Text response:", text)
    def process_input(self, input_type="text"):
        if input_type == "voice":
                return self.listen_for_command()
        else:
            return input("\nEnter your financial question (or 'quit' to exit): ")

    def deliver_response(self, response, output_type="text"):
        print("\nFinancial Advice:", response)
        if output_type == "voice":
            self.speak_response(response)

    def store_chat_history(self, user_id, query, response):
        try:
            supabase.table("chat_history").insert({
                "user_id": user_id,
                "query": query,
                "response": response
            }).execute()
        except Exception as e:
            print(f"Error storing chat history: {str(e)}")
    
    def get_chat_history(self, user_id):
        try:
            history = supabase.table("chat_history")\
                .select("*")\
                .eq('user_id', user_id)\
                .order('timestamp', desc=True)\
                .limit(self.conversation_window)\
                .execute().data
            return history[::-1]  # Reverse to get chronological order
        except Exception as e:
            print(f"Error fetching chat history: {str(e)}")
            return []

    def get_response(self, query, user_id):
        relevant_data = self.retrieve_relevant_data(query, user_id)
        chat_history = self.get_chat_history(user_id)
        
        # chat history context
        history_context = "\nPrevious Conversation:\n" + "\n".join([
            f"User: {exchange['query']}\nAssistant: {exchange['response']}"
            for exchange in chat_history
        ]) if chat_history else ""
        
        # transaction context
        transaction_context = "\n".join([
            f"- Transaction on {t['transaction_date']}: ${t['amount']} - {t['category']}/{t['subcategory']} ({t['type']}) paid via {t['mode_of_payment']}"
            for t in relevant_data['transactions']
        ])

        
        profile = relevant_data['user_profile']
        profile_context = f"\nUser Profile:\nAge: {profile['age']}\nIncome: ${profile['income']}\nDebts: ${profile['debts']}\nAccount Balance: ${profile['account_balance']}" if profile else ""

        
        goals_context = "\nFinancial Goals:\n" + "\n".join([
            f"- {g['goal_name']}: Target ${g['target_amount']}, Current Progress: ${g['current_amount']}, Due: {g['target_date']}"
            for g in relevant_data['financial_goals']
        ]) if relevant_data['financial_goals'] else ""

        prompt = f"""
        {self.system_instructions}
        
        User Financial Information:
        {profile_context}
        {goals_context}
        
        Recent Transactions:
        {transaction_context}
        
        {history_context}
        
        Question: {query}
        """

        response = self.model.generate_content(prompt)
        
        # Store the new exchange in history
        self.store_chat_history(user_id, query, response.text)
        
        return response.text
# Flask app 
app = Flask(__name__)
financial_bot = FinancialBotV2(api_key=GEMINI_API_KEY)

@app.route('/update-embeddings', methods=['POST'])
def update_embeddings():
    result = financial_bot.update_embeddings_for_existing_transactions()
    return jsonify({"status": "success", "message": result})
@app.route('/chat-history', methods=['GET'])
def get_history():
    user_id = request.args.get('user_id', '1')
    history = financial_bot.get_chat_history(user_id)
    return jsonify({"history": history})

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    query = data.get('query')
    user_id = str(data.get('user_id', 1))
    response = financial_bot.get_response(query, user_id)
    return jsonify({
        "response": response,
        "history": financial_bot.get_chat_history(user_id)
    })
if __name__ == '__main__':
    print("Welcome to Spendrick V2 - Text and Voice enabled!")
    financial_bot = FinancialBotV2(api_key=GEMINI_API_KEY)
    financial_bot.update_embeddings_for_existing_transactions()

    print("\nChoose your interaction mode:")
    print("1. Text to Text")
    print("2. Voice to Voice")
    print("3. Text to Voice")
    print("4. Voice to Text")
    
    mode = input("Select mode (1-4): ")
    
    while True:
        input_type = "voice" if mode in ["2", "4"] else "text"
        output_type = "voice" if mode in ["2", "3"] else "text"
        
        query = financial_bot.process_input(input_type)
        if not query or query.lower() == 'quit':
            break
            
        response = financial_bot.get_response(query, user_id="1")
        financial_bot.deliver_response(response, output_type)
        print("\n" + "="*50)


@app.route('/voice-chat', methods=['POST'])
def voice_chat():
    audio_file = request.files.get('audio')
    output_type = request.form.get('output_type', 'text')
    user_id = request.form.get('user_id', '1')
    
    if not audio_file:
        return jsonify({"error": "No audio file provided"}), 400
    
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio = recognizer.record(source)
        try:
            query = recognizer.recognize_google(audio)
            response = financial_bot.get_response(query, user_id)
            
            if output_type == 'voice':
                tts = gTTS(text=response, lang='en')
                temp_file = "response.mp3"
                tts.save(temp_file)
                return send_file(
                    temp_file,
                    mimetype="audio/mpeg",
                    as_attachment=True,
                    download_name="response.mp3"
                )
            else:
                return jsonify({
                    "response": response,
                    "history": financial_bot.get_chat_history(user_id)
                })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

@app.route('/interact', methods=['POST'])
def interact():
    data = request.json
    input_type = data.get('input_type', 'text')
    output_type = data.get('output_type', 'text')
    user_id = str(data.get('user_id', 1))
    query = data.get('query')
    
    response = financial_bot.get_response(query, user_id)
    
    if output_type == 'voice':
        tts = gTTS(text=response, lang='en')
        temp_file = "response.mp3"
        tts.save(temp_file)
        return send_file(
            temp_file,
            mimetype="audio/mpeg",
            as_attachment=True,
            download_name="response.mp3"
        )
    else:
        return jsonify({
            "response": response,
            "history": financial_bot.get_chat_history(user_id)
        })

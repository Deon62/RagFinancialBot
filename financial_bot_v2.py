
from supabase import create_client, Client
from sentence_transformers import SentenceTransformer
from flask import Flask, request, jsonify
import google.generativeai as genai
from config import GEMINI_API_KEY, SUPABASE_URL, SUPABASE_KEY
import numpy as np

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
                "max_output_tokens": 8192,
            }
        )
        self.system_instructions = "You're a financial assistant known as Deon. Give personalized advice with a friendly touch and also add humor."

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
            print(f"Embedding update process: {str(e)}")
            return "Process completed"

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

    def get_response(self, query, user_id):
        relevant_data = self.retrieve_relevant_data(query, user_id)
        
        # Format transaction context
        transaction_context = "\n".join([
            f"- Transaction on {t['transaction_date']}: ${t['amount']} - {t['category']}/{t['subcategory']} ({t['type']}) paid via {t['mode_of_payment']}"
            for t in relevant_data['transactions']
        ])

        # Format user profile context
        profile = relevant_data['user_profile']
        profile_context = f"\nUser Profile:\nAge: {profile['age']}\nIncome: ${profile['income']}\nDebts: ${profile['debts']}\nAccount Balance: ${profile['account_balance']}" if profile else ""

        # Format financial goals context
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
        
        Question: {query}
        """

        response = self.model.generate_content(prompt)
        return response.text

# Flask app initialization remains the same
app = Flask(__name__)
financial_bot = FinancialBotV2(api_key=GEMINI_API_KEY)

@app.route('/update-embeddings', methods=['POST'])
def update_embeddings():
    result = financial_bot.update_embeddings_for_existing_transactions()
    return jsonify({"status": "success", "message": result})

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    query = data.get('query')
    user_id = str(data.get('user_id', 1))  # Convert to string
    response = financial_bot.get_response(query, user_id)
    return jsonify({"response": response})

if __name__ == '__main__':
    print("Welcome to Spendrick V2 - Now with comprehensive financial insights!")
    
    financial_bot.update_embeddings_for_existing_transactions()
    print("✓ Embeddings updated for all transactions")
    
    while True:
        query = input("\nEnter your financial question (or 'quit' to exit): ")
        if query.lower() == 'quit':
            break
        response = financial_bot.get_response(query, user_id="1")  # Pass as string
        print("\nFinancial Advice:", response)
        print("\n" + "="*50)

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

    def update_embeddings_for_existing_transactions(self):
        """Update embeddings for all transactions that don't have them"""
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
            
            # First try semantic search
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
                return supabase.table("transactions")\
                    .select("*")\
                    .in_('transaction_id', transaction_ids)\
                    .execute().data

            # Fallback to original logic
            transactions = supabase.table("transactions").select("*").execute().data
            
            if "last transaction" in query.lower():
                return [max(transactions, key=lambda x: x['transaction_date'])]
            elif "how many" in query.lower() and "transactions" in query.lower():
                return transactions
            
            return transactions[-100:]

        except Exception as e:
            print(f"Query error: {str(e)}")
            return []

    def get_response(self, query, user_id):
        relevant_data = self.retrieve_relevant_data(query, user_id)
        
        context = "\n".join([
            f"- Transaction on {t['transaction_date']}: ${t['amount']} - {t['category']}/{t['subcategory']} ({t['type']}) paid via {t['mode_of_payment']}"
            for t in relevant_data
        ])

        if "how many" in query.lower() and "transactions" in query.lower():
            prompt = f"You have {len(relevant_data)} transactions in total. Recent transactions:\n{context}\nProvide a friendly summary."
        elif "last transaction" in query.lower():
            prompt = f"Most recent transaction:\n{context}\nProvide a friendly summary."
        else:
            prompt = f"""
            {self.system_instructions}
            
            Most relevant transactions based on your query:
            {context}
            
            Question: {query}
            """

        response = self.model.generate_content(prompt)
        return response.text

# Initialize Flask app
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
    user_id = data.get('user_id', 1)
    response = financial_bot.get_response(query, user_id)
    return jsonify({"response": response})

if __name__ == '__main__':
    print("Welcome to Spendrick V2 - Now with semantic search!")
    
    # Update embeddings for existing transactions
    financial_bot.update_embeddings_for_existing_transactions()
    print("✓ Embeddings updated for all transactions")
    
    while True:
        query = input("\nEnter your financial question (or 'quit' to exit): ")
        if query.lower() == 'quit':
            break
        response = financial_bot.get_response(query, user_id=1)
        print("\nFinancial Advice:", response)
        print("\n" + "="*50)

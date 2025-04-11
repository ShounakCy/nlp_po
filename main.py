import os
import re
import spacy
import json
import glob
import pandas as pd
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path


# Load spaCy model
try:
    nlp = spacy.load("en_core_web_trf")
except OSError:
    raise RuntimeError(
        "The spaCy model 'en_core_web_trf' is not installed.\n"
        "Install it using: python -m spacy download en_core_web_trf"
    )


class PurchaseOrderProcessor:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english')
    
    def extract_po_number(self, filename):
        """Extract PO number directly from filename using regex"""
        # Extract PO number from filename pattern like "Customer_26_PO_26.txt"
        filename_match = re.search(r'PO_(\d+)', filename)
        if filename_match:
            return f"{filename_match.group(1)}"
        
        return None
    
    def extract_delivery_date(self, text):
        """Extract delivery date using pattern recognition with improved pattern matching"""
        # First, try using spaCy NER
        doc = nlp(text)
        
        # Look for date entities
        for ent in doc.ents:
            if ent.label_ == "DATE" and any(date_word in text[max(0, ent.start_char-20):ent.start_char].lower() 
                                        for date_word in ["delivery", "required", "expected"]):
                # Try to parse the date
                try:
                    # Get the actual date part from the entity
                    date_text = re.search(r'([A-Za-z]+ \d{1,2},? \d{4})', ent.text)
                    if date_text:
                        date_str = date_text.group(1)
                        # Handle different formats
                        if ',' in date_str:
                            return datetime.strptime(date_str, '%B %d, %Y').strftime('%Y-%m-%d')
                        else:
                            return datetime.strptime(date_str, '%B %d %Y').strftime('%Y-%m-%d')
                except ValueError:
                    pass
        
        return None
    
    def extract_customer_name(self, file_name, content):
        """Extract customer name using NLP if present in content, otherwise from filename"""
        # First, try to extract from content using NLP
        doc = nlp(content)
        
        # Look for person names at the end of the document (signatures)
        for ent in reversed(list(doc.ents)):
            if ent.label_ == "PERSON" and content.find(ent.text) > len(content) // 2:
                return ent.text
        
        # Look for indicators like "Best," or "Regards," followed by a name
        closing_matches = re.finditer(r'(?:Best|Regards|Sincerely),?\s*\n?\s*([A-Za-z ]+)', content)
        for match in closing_matches:
            return match.group(1).strip()
        
        return "Unknown"
    
    def extract_items(self, text):
        """Extract item details using improved pattern matching for multiple items"""
        items = []
        
        # Look for patterns like "X units of [Item Name] (Item Code: [CODE])"
        # This regex captures each line item independently
        item_pattern = r'(\d+)\s+units\s+of\s+([^(]+)\s*\(Item\s+Code:\s+([A-Za-z0-9-]+)\)'
        
        # Find all matches in the text
        matches = re.finditer(item_pattern, text, re.IGNORECASE)
        
        for match in matches:
            quantity = int(match.group(1))
            item_name = match.group(2).strip()
            item_code = match.group(3).strip()
            
            items.append({
                'quantity': quantity,
                'item_name': item_name,
                'item_code': item_code
            })
        
        # If no items found using regex, fall back to NLP-based extraction
        if not items:
            doc = nlp(text)
            for sent in doc.sents:
                # Sentence must contain a digit and the word "units"
                if any(token.like_num for token in sent) and "units" in sent.text.lower():
                    quantity = None
                    item_code = None
                    item_name = None

                    # Look for quantity (e.g. "5 units")
                    for i, token in enumerate(sent):
                        if token.like_num and i + 1 < len(sent) and sent[i+1].text.lower() == "units":
                            quantity = int(token.text)
                            break

                    # Look for item code (e.g. "(Item Code: XYZ123)")
                    item_code_match = re.search(r'\(Item Code: ([A-Za-z0-9-]+)\)', sent.text)
                    item_code = item_code_match.group(1).strip() if item_code_match else None

                    # Try to extract item name using the structure around quantity and item code
                    if quantity and item_code:
                        name_match = re.search(
                            fr'{quantity} units of (.*?)\(Item Code: {re.escape(item_code)}', 
                            sent.text
                        )
                        item_name = name_match.group(1).strip() if name_match else None

                    if quantity and item_name and item_code:
                        items.append({
                            'quantity': quantity,
                            'item_name': item_name,
                            'item_code': item_code
                        })
        
        return items
  
    def extract_po_data(self, text, filename):
        """Extract purchase order data into a structured JSON format"""
        # Initialize the PO data structure
        po_data = {
            "po_number": self.extract_po_number(filename),
            "customer": self.extract_customer_name(filename, text),
            "delivery_date": self.extract_delivery_date(text),
            "source": filename,
            "line_items": []
        }
        
        # Extract all line items
        items = self.extract_items(text)
        for item in items:
            po_data["line_items"].append({
                "quantity": item["quantity"],
                "item_name": item["item_name"],
                "item_code": item["item_code"]
            })
        
        return po_data
    
    def process_purchase_orders(self, documents):
        """Process all purchase orders and return structured JSON data"""
        processed_orders = []
        processed_items = []  # For backward compatibility with existing code
        
        for doc in documents:
            source = doc.get('source', '')
            content = doc.get('document_content', '')
            
            # Extract all PO data into a JSON structure
            po_data = self.extract_po_data(content, source)
            po_data["raw_text"] = content
            
            processed_orders.append(po_data)
            
            # Create individual item records for compatibility with existing analysis code
            # Each line item from the same PO now gets its own record
            for item in po_data["line_items"]:
                processed_items.append({
                    'po_source': source,
                    'po_number': po_data["po_number"],
                    'customer': po_data["customer"],
                    'delivery_date': po_data["delivery_date"],
                    'item_name': item['item_name'],
                    'item_code': item['item_code'],
                    'quantity': item['quantity'],
                    'raw_text': content
                })
        
        # Create DataFrame from the processed items - allowing multiple items per source
        return processed_orders, pd.DataFrame(processed_items)
    
    def detect_potential_duplicates(self, df):
        """
        Detect potential duplicate purchase orders using text similarity.
        """
        if df.empty or len(df) < 2:
            return []
        
        duplicates = []
        processed_pairs = set()  # To avoid comparing the same pair twice
        
        # Group DataFrame by source to get unique POs (one entry per source)
        unique_pos = df.drop_duplicates('po_source')[['po_source', 'po_number', 'raw_text', 'customer']]
        
        # Reset index to ensure contiguous indices for the TFIDF matrix
        unique_pos = unique_pos.reset_index(drop=True)
        
        # Create corpus for TF-IDF vectorization
        if 'raw_text' in unique_pos.columns:
            corpus = unique_pos['raw_text'].fillna('').tolist()
            tfidf_matrix = self.vectorizer.fit_transform(corpus)
        else:
            # If raw_text column is missing, we can't perform text similarity
            return []
        
        # Loop through each PO document using the row index for tfidf_matrix access
        for i in range(len(unique_pos)):
            row_i = unique_pos.iloc[i]
            source_i = row_i['po_source']
            raw_text_i = row_i['raw_text']
            
            for j in range(i + 1, len(unique_pos)):  # Start from i+1 to avoid duplicate comparisons
                row_j = unique_pos.iloc[j]
                source_j = row_j['po_source']
                raw_text_j = row_j['raw_text']
                
                # Skip if we've already processed this pair (should not happen with this loop structure)
                if (source_i, source_j) in processed_pairs or (source_j, source_i) in processed_pairs:
                    continue
                
                # Track that we've processed this pair
                processed_pairs.add((source_i, source_j))
                
                # Calculate text similarity using TF-IDF with proper indices
                text_similarity = cosine_similarity(
                    tfidf_matrix[i].reshape(1, -1), 
                    tfidf_matrix[j].reshape(1, -1)
                )[0][0]

                text_similarity = min(1.0, max(0.0, text_similarity))
                
                # Determine similarity type based on text similarity score
                if text_similarity > 0.8:
                    similarity_type = "Very high text similarity"
                elif text_similarity > 0.7:
                    similarity_type = "High text similarity"
                elif text_similarity > 0.6:
                    similarity_type = "Moderate text similarity"
                elif text_similarity > 0.5:
                    similarity_type = "Low text similarity"
                else:
                    similarity_type = "Very low similarity"
                
                # Report only significant similarities
                if text_similarity >= 0.8:
                    duplicates.append({
                        'po1': {
                            'source': source_i,
                            'raw_text': raw_text_i
                        },
                        'po2': {
                            'source': source_j,
                            'raw_text': raw_text_j
                        },
                        'text_similarity': float(text_similarity),
                        'similarity_type': similarity_type
                    })
        
        # Sort duplicates by similarity score (highest first)
        return sorted(duplicates, key=lambda x: x['text_similarity'], reverse=True)
                    
    def analyze_orders_by_item_code(self, df):
        """Analyze orders by item code to identify potential issues"""
        item_summary = df.groupby('item_code').agg({
            'quantity': ['sum', 'count'],
            'po_source': 'nunique'
        })
        
        item_summary.columns = ['total_quantity', 'order_count', 'unique_po_sources']
        
        # Convert to list of dictionaries for JSON serialization
        result = []
        for item_code, row in item_summary.reset_index().iterrows():
            result.append({
                'item_code': row['item_code'],
                'total_quantity': int(row['total_quantity']),
                'order_count': int(row['order_count']),
                'unique_po_sources': int(row['unique_po_sources'])
            })
        
        return result
    

def load_documents_from_data(doc_data):
    """Load documents from the provided data structure"""
    documents = []
    
    for doc_idx in range(len(doc_data)):
        doc = {
            'source': doc_data[doc_idx].get('source', f'document_{doc_idx}.txt'),
            'document_content': doc_data[doc_idx].get('document_content', '')
        }
        documents.append(doc)
    
    return documents

def load_documents_from_folder(folder_path):
        """Load documents from text files in the specified folder"""
        documents = []
        folder = Path(folder_path)
        for file_path in folder.glob("*.txt"):
            with file_path.open('r', encoding='utf-8') as file:
                content = file.read()
                documents.append({
                    'source': os.path.basename(file_path),
                    'document_content': content
                })
        return documents

def main(documents_data):
    """Main function to process purchase orders"""
    processor = PurchaseOrderProcessor()
    
    print("Loading documents...")
    documents = load_documents_from_data(documents_data)
    
    print("Processing purchase orders with NLP...")
    json_data, df = processor.process_purchase_orders(documents)
    
    print(f"Processed {len(json_data)} purchase orders with {len(df)} line items")
    
    results_folder = Path('results')
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    
    # Save processed data to JSON
    with open(results_folder / 'processed_purchase_orders.json', 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=4)
    print("Saved processed data to results/processed_purchase_orders.json")
    
    # Detect potential duplicates using NLP
    print("Detecting potential duplicate orders with NLP techniques...")
    duplicates = processor.detect_potential_duplicates(df)
    
    # Save duplicates report as JSON
    with open(results_folder / 'duplicate_purchase_orders.json', 'w', encoding='utf-8') as f:
        json.dump(duplicates, f, indent=4)
    print(f"Found {len(duplicates)} potential duplicate orders")
    print("Saved duplicate report to results/duplicate_purchase_orders.json")
    
    # Analyze orders by item code
    item_analysis = processor.analyze_orders_by_item_code(df)
    with open(results_folder / 'item_code_analysis.json', 'w', encoding='utf-8') as f:
        json.dump(item_analysis, f, indent=4)
    print("Saved item code analysis to results/item_code_analysis.json")
    
    print("Processing completed successfully!")
    
    # Return processed data for further analysis or reporting
    return {
        'processed_data': json_data,
        'duplicates': duplicates,
        'item_analysis': item_analysis,
    }

if __name__ == "__main__":
    
    folder_path = 'updated_purchase_orders'
    example_docs = load_documents_from_folder(folder_path)
    main(example_docs)
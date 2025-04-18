# Purchase Order Processing Automation - Summary

## Approach

I implemented a solution using Python that leverages natural language processing (NLP) and text analysis techniques. The approach consisted of:

1. **Data Extraction Pipeline**: Created a system using spaCy for NLP processing coupled with regex pattern matching to extract key information from various PO formats.

2. **Structured Data Organization**: Implemented a JSON-based data structure to standardize extracted information including PO numbers, customer details, delivery dates, and line items.

3. **Duplicate Detection System**: Developed a text similarity analysis engine using TF-IDF vectorization and cosine similarity metrics to identify potential duplicate orders.

4. **Multi-stage Fallback Processing**: Built extraction methods with progressive fallbacks to handle different document formats and incomplete information.

## Key Challenges and Solutions

### Challenge 1: Varied Document Formats
**Solution**: Implemented multiple extraction strategies including NLP entity recognition, contextual pattern matching, and filename parsing to handle format variations. The system attempts different strategies and falls back gracefully when primary methods fail.

### Challenge 2: Entity Extraction Accuracy
**Solution**: Used spaCy's transformer-based model (`en_core_web_trf`) for improved entity recognition accuracy, particularly for dates and person names. Supplemented this with context-aware pattern matching to increase precision.

### Challenge 3: Effective Duplicate Detection
**Solution**: Implemented a vectorization approach using TF-IDF to convert text documents into numerical representations, then calculated cosine similarity between document pairs. This method successfully identified duplicates even with minor formatting differences.

### Challenge 4: Missing Information
**Solution**: Created intelligent fallbacks to extract information from filenames when document content was insufficient, and implemented a robust data structure that could accommodate partial information while maintaining integrity.



## Extraction Evaluation Summary

### Overall Metrics
- **Accuracy**: 1.0000  
- **Precision**: 1.0000  
- **Recall**: 1.0000  
- **F1 Score**: 1.0000  
- **Documents Processed**: 3  

### Field-specific Metrics

#### PO Number
- **Accuracy**: 1.0000  
- **Precision**: 1.0000  
- **Recall**: 1.0000  
- **F1 Score**: 1.0000  

#### Customer
- **Accuracy**: 1.0000  
- **Precision**: 1.0000  
- **Recall**: 1.0000  
- **F1 Score**: 1.0000  

#### Delivery Date
- **Accuracy**: 1.0000  
- **Precision**: 1.0000  
- **Recall**: 1.0000  
- **F1 Score**: 1.0000  

#### Line Item Count
- **Accuracy**: 1.0000  
- **Precision**: 1.0000  
- **Recall**: 1.0000  
- **F1 Score**: 1.0000  

#### Line Items
- **Accuracy**: 1.0000  
- **Precision**: 1.0000  
- **Recall**: 1.0000  
- **F1 Score**: 1.0000  

##### Line Item Details
- **Item Count**: 3/3  
- **Item Codes**: 10/10  
- **Quantities**: 10/10  
- **Item Names**: 10/10  
- **Fully Correct Items**: 10/10  

### Document-level Evaluation

#### PO_11.txt
- **PO Number**: Correct  
- **Line Items**: 3/3 fully correct  

#### Customer_26_PO_26.txt
- **PO Number**: Correct  
- **Customer**: Correct  
- **Delivery Date**: Correct  
- **Line Items**: 3/3 fully correct  

#### Tommy_Davis_PO_27.txt
- **PO Number**: Correct  
- **Customer**: Correct  
- **Line Items**: 4/4 fully correct  

### Additional Information
- Full results saved to: `evaluation_results/po_extraction_results.json`  
- **Evaluation Completed**: Overall F1 Score: 1.0000  
- **Performance Assessment**: Excellent  


## Suggestions for Further Improvements

1. **Machine Learning Enhancement**: Train a custom NER (Named Entity Recognition) model specific to purchase order documents to improve extraction accuracy.

2. **Fuzzy Matching for Products**: Incorporate fuzzy matching algorithms to identify similar products that might be listed with slight naming variations.

3. **Database Integration**: Connect the system to a company database for real-time validation of customer information, product codes, and pricing.

4. **Multi-factor Duplicate Detection**: Extend duplicate detection beyond text similarity to include semantic analysis and business rule validation.

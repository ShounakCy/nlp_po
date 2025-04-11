import os
import json
import time
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Import the processor module (main.py instead of paste.py)
from main import PurchaseOrderProcessor, load_documents_from_folder

# Accurate ground truth based on the provided text files
GROUND_TRUTH = {
    "Customer_26_PO_26.txt": {
        "po_number": "26",
        "customer": "Angelica Delgado",
        "delivery_date": "2025-02-15",
        "line_items": [
            {"quantity": 200, "item_name": "Control Valve", "item_code": "MA-2200"},
            {"quantity": 430, "item_name": "Compressor Unit", "item_code": "CU-5643"},
            {"quantity": 183, "item_name": "Hydraulic Pump", "item_code": "HE-2345"}
        ]
    },
    "PO_11.txt": {
        "po_number": "11",
        "customer": "Unknown",  # Not clearly specified in the text
        "delivery_date": None,  # Not specified in the text
        "line_items": [
            {"quantity": 422, "item_name": "Pneumatic Cylinder", "item_code": "HE-2345"},
            {"quantity": 162, "item_name": "Hydraulic Pump", "item_code": "BS-7890"},
            {"quantity": 445, "item_name": "Bearing Set", "item_code": "CB-3300"}
        ]
    },
    "Tommy_Davis_PO_27.txt": {
        "po_number": "27",
        "customer": "Tommy Davis",
        "delivery_date": None,  # Not specified in the text
        "line_items": [
            {"quantity": 434, "item_name": "Conveyor Belt", "item_code": "CV-6677"},
            {"quantity": 113, "item_name": "Compressor Unit", "item_code": "CU-5643"},
            {"quantity": 436, "item_name": "Control Valve", "item_code": "CB-3300"},
            {"quantity": 385, "item_name": "Motor Assembly", "item_code": "BS-7890"}
        ]
    }
}


class POEvaluator:
    def __init__(self, test_data_path):
        """
        Initialize evaluator with path to test data and ground truth
        
        Args:
            test_data_path: Path to folder containing purchase order text files
        """
        self.test_data_path = test_data_path
        self.processor = PurchaseOrderProcessor()
        self.ground_truth = GROUND_TRUTH
    
    def evaluate_extraction(self):
        """
        Evaluate extraction against ground truth
        """
        # Load and process documents
        documents = load_documents_from_folder(self.test_data_path)
        print(f"Processing {len(documents)} documents...")
        
        # Filter documents to only those with ground truth
        eval_documents = [doc for doc in documents if doc["source"] in self.ground_truth]
        print(f"Found {len(eval_documents)} documents with ground truth for evaluation")
    
        
        # Process documents and measure performance
        processed_data, df = self.processor.process_purchase_orders(eval_documents)
        
        # Create lookup for processed data
        processed_lookup = {item["source"]: item for item in processed_data}
        
        # Track correct and incorrect extractions
        results = {
            "po_number": {"correct": 0, "incorrect": 0, "missing": 0},
            "customer": {"correct": 0, "incorrect": 0, "missing": 0},
            "delivery_date": {"correct": 0, "incorrect": 0, "missing": 0},
            "line_items": {
                "item_count": {"correct": 0, "total": 0},
                "item_codes": {"correct": 0, "total": 0},
                "quantities": {"correct": 0, "total": 0},
                "names": {"correct": 0, "total": 0},
                "fully_correct_items": {"correct": 0, "total": 0}
            }
        }
        
        # Prepare lists for sklearn metrics
        y_true = {"po_number": [], "customer": [], "delivery_date": [], "line_item_count": []}
        y_pred = {"po_number": [], "customer": [], "delivery_date": [], "line_item_count": []}
        
        # For line items, we'll track individual elements
        line_items_true = []
        line_items_pred = []
        
        # Store detailed results for each document
        detailed_results = []
        
        # Compare each file with ground truth
        for doc in eval_documents:
            source = doc["source"]
            truth = self.ground_truth[source]
            
            doc_result = {
                "source": source,
                "results": {},
                "errors": [],
                "line_item_details": []
            }
            
            if source in processed_lookup:
                processed = processed_lookup[source]
                
                # Compare PO number
                po_correct = processed["po_number"] == truth["po_number"]
                if po_correct:
                    results["po_number"]["correct"] += 1
                    doc_result["results"]["po_number"] = "correct"
                else:
                    results["po_number"]["incorrect"] += 1
                    doc_result["results"]["po_number"] = "incorrect"
                    doc_result["errors"].append(f"PO Number: expected '{truth['po_number']}', got '{processed['po_number']}'")
                
                y_true["po_number"].append(1)  # Ground truth is always correct
                y_pred["po_number"].append(1 if po_correct else 0)
                
                # Compare customer (only if ground truth has a customer)
                if truth["customer"] != "Unknown":
                    customer_correct = processed["customer"] == truth["customer"]
                    if customer_correct:
                        results["customer"]["correct"] += 1
                        doc_result["results"]["customer"] = "correct"
                    else:
                        results["customer"]["incorrect"] += 1
                        doc_result["results"]["customer"] = "incorrect"
                        doc_result["errors"].append(f"Customer: expected '{truth['customer']}', got '{processed['customer']}'")
                    
                    y_true["customer"].append(1)
                    y_pred["customer"].append(1 if customer_correct else 0)
                
                # Compare delivery date (only if ground truth has a date)
                if truth["delivery_date"]:
                    date_correct = processed["delivery_date"] == truth["delivery_date"]
                    if date_correct:
                        results["delivery_date"]["correct"] += 1
                        doc_result["results"]["delivery_date"] = "correct"
                    else:
                        results["delivery_date"]["incorrect"] += 1
                        doc_result["results"]["delivery_date"] = "incorrect"
                        doc_result["errors"].append(f"Delivery Date: expected '{truth['delivery_date']}', got '{processed['delivery_date']}'")
                    
                    y_true["delivery_date"].append(1)
                    y_pred["delivery_date"].append(1 if date_correct else 0)
                
                # Evaluate line item count
                truth_items = truth["line_items"]
                processed_items = processed["line_items"] if processed["line_items"] else []
                
                # Check if the count of line items is correct
                item_count_correct = len(processed_items) == len(truth_items)
                results["line_items"]["item_count"]["total"] += 1
                if item_count_correct:
                    results["line_items"]["item_count"]["correct"] += 1
                else:
                    doc_result["errors"].append(f"Line item count: expected {len(truth_items)}, got {len(processed_items)}")
                
                y_true["line_item_count"].append(1)
                y_pred["line_item_count"].append(1 if item_count_correct else 0)
                
                # Store the total number of items for this document
                results["line_items"]["fully_correct_items"]["total"] += len(truth_items)
                
                # Create lookup by item code for easier matching
                processed_by_code = {item["item_code"]: item for item in processed_items}
                
                # Evaluate each line item individually
                for truth_item in truth_items:
                    item_code = truth_item["item_code"]
                    item_result = {
                        "item_code": item_code,
                        "expected": truth_item,
                        "found": False,
                        "extracted": None,
                        "correct_code": False,
                        "correct_quantity": False,
                        "correct_name": False,
                        "fully_correct": False
                    }
                    
                    # Track metrics for this item
                    results["line_items"]["item_codes"]["total"] += 1
                    results["line_items"]["quantities"]["total"] += 1
                    results["line_items"]["names"]["total"] += 1
                    
                    # Check if this item code was found
                    if item_code in processed_by_code:
                        proc_item = processed_by_code[item_code]
                        item_result["found"] = True
                        item_result["extracted"] = proc_item
                        item_result["correct_code"] = True
                        results["line_items"]["item_codes"]["correct"] += 1
                        
                        # Check quantity
                        quantity_correct = proc_item["quantity"] == truth_item["quantity"]
                        item_result["correct_quantity"] = quantity_correct
                        if quantity_correct:
                            results["line_items"]["quantities"]["correct"] += 1
                        else:
                            doc_result["errors"].append(
                                f"Quantity mismatch for {item_code}: "
                                f"expected {truth_item['quantity']}, got {proc_item['quantity']}"
                            )
                        
                        # Check name (with some flexibility)
                        name_match = (truth_item["item_name"].lower() in proc_item["item_name"].lower() or 
                                     proc_item["item_name"].lower() in truth_item["item_name"].lower())
                        item_result["correct_name"] = name_match
                        if name_match:
                            results["line_items"]["names"]["correct"] += 1
                        else:
                            doc_result["errors"].append(
                                f"Item name mismatch for {item_code}: "
                                f"expected '{truth_item['item_name']}', got '{proc_item['item_name']}'"
                            )
                        
                        # Check if the item is fully correct
                        fully_correct = quantity_correct and name_match
                        item_result["fully_correct"] = fully_correct
                        if fully_correct:
                            results["line_items"]["fully_correct_items"]["correct"] += 1
                        
                        # Add to line items true/pred vectors for overall metrics
                        line_items_true.append(1)  # Should be found and correct
                        line_items_pred.append(1 if fully_correct else 0)
                    else:
                        # Item code not found
                        doc_result["errors"].append(f"Missing item: {item_code}")
                        line_items_true.append(1)  # Should be found
                        line_items_pred.append(0)  # Not found
                    
                    doc_result["line_item_details"].append(item_result)
                
                # Also check for extra items that weren't in the ground truth
                truth_codes = {item["item_code"] for item in truth_items}
                for proc_item in processed_items:
                    if proc_item["item_code"] not in truth_codes:
                        doc_result["errors"].append(f"Extra item: {proc_item['item_code']}")
                        # This is a false positive for line items
                        line_items_true.append(0)  # Should not be found
                        line_items_pred.append(1)  # Was found
            else:
                # Document wasn't processed
                results["po_number"]["missing"] += 1
                results["customer"]["missing"] += 1
                results["delivery_date"]["missing"] += 1
                doc_result["results"] = "document missing from processed results"
            
            detailed_results.append(doc_result)
        
        # Calculate metrics with sklearn for document-level fields
        metrics = {}
        for field in ["po_number", "customer", "delivery_date", "line_item_count"]:
            if y_true[field]:  # Only calculate if we have data for this field
                metrics[field] = {
                    "accuracy": accuracy_score(y_true[field], y_pred[field]),
                    "precision": precision_score(y_true[field], y_pred[field], zero_division=0),
                    "recall": recall_score(y_true[field], y_pred[field], zero_division=0),
                    "f1": f1_score(y_true[field], y_pred[field], zero_division=0)
                }
            else:
                metrics[field] = {
                    "accuracy": 0,
                    "precision": 0,
                    "recall": 0,
                    "f1": 0
                }
        
        # Calculate line items metrics
        if line_items_true:
            metrics["line_items"] = {
                "accuracy": accuracy_score(line_items_true, line_items_pred),
                "precision": precision_score(line_items_true, line_items_pred, zero_division=0),
                "recall": recall_score(line_items_true, line_items_pred, zero_division=0),
                "f1": f1_score(line_items_true, line_items_pred, zero_division=0)
            }
        else:
            metrics["line_items"] = {
                "accuracy": 0,
                "precision": 0,
                "recall": 0,
                "f1": 0
            }
        
        # Add raw counts to metrics
        metrics["po_number"]["counts"] = results["po_number"]
        metrics["customer"]["counts"] = results["customer"]
        metrics["delivery_date"]["counts"] = results["delivery_date"]
        metrics["line_items"]["counts"] = results["line_items"]
        
        # Calculate overall accuracy for all fields combined
        all_true = []
        all_pred = []
        for field in ["po_number", "customer", "delivery_date"]:
            all_true.extend(y_true[field])
            all_pred.extend(y_pred[field])
        
        # Add line items to overall metrics
        all_true.extend(line_items_true)
        all_pred.extend(line_items_pred)
        
        overall_accuracy = accuracy_score(all_true, all_pred) if all_true else 0
        overall_precision = precision_score(all_true, all_pred, zero_division=0) if all_true else 0
        overall_recall = recall_score(all_true, all_pred, zero_division=0) if all_true else 0
        overall_f1 = f1_score(all_true, all_pred, zero_division=0) if all_true else 0
        
        return {
            "field_metrics": metrics,
            "overall_accuracy": overall_accuracy,
            "overall_precision": overall_precision, 
            "overall_recall": overall_recall,
            "overall_f1": overall_f1,
            "documents_processed": len(eval_documents),
            "detailed_results": detailed_results
        }
    
    def run_evaluation(self):
        """
        Run evaluation of the purchase order processor
        """
        
        # Run extraction evaluation
        extraction_results = self.evaluate_extraction()
        
        if not extraction_results:
            print("Evaluation failed - no matching documents found.")
            return None
        
        results = {"extraction": extraction_results}
        
        # Save results to JSON
        output_dir = Path("evaluation_results")
        if not output_dir.exists():
            output_dir.mkdir(parents=True)
            
        with open(output_dir / "po_extraction_results.json", 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4)
        
        # Print summary
        self._print_evaluation_summary(results)
        
        return results
    
    
    def _print_evaluation_summary(self, results):
        """Print a summary of the evaluation results"""
        print("\nExtraction Evaluation Summary:")
        print("-" * 80)
        
        # Overall metrics
        print(f"Overall Accuracy: {results['extraction']['overall_accuracy']:.4f}")
        print(f"Overall Precision: {results['extraction']['overall_precision']:.4f}")
        print(f"Overall Recall: {results['extraction']['overall_recall']:.4f}")
        print(f"Overall F1 Score: {results['extraction']['overall_f1']:.4f}")
        print(f"Documents Processed: {results['extraction']['documents_processed']}")
        
        # Field-specific metrics
        print("\nField-specific metrics:")
        for field, metrics in results['extraction']['field_metrics'].items():
            print(f"\n{field.replace('_', ' ').title()}:")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  F1 Score: {metrics['f1']:.4f}")
            
            # Print counts for fields with count data
            if field == "line_items" and "counts" in metrics:
                li_counts = metrics["counts"]
                print("\n  Line Item Details:")
                print(f"    Item Count: {li_counts['item_count']['correct']}/{li_counts['item_count']['total']}")
                print(f"    Item Codes: {li_counts['item_codes']['correct']}/{li_counts['item_codes']['total']}")
                print(f"    Quantities: {li_counts['quantities']['correct']}/{li_counts['quantities']['total']}")
                print(f"    Item Names: {li_counts['names']['correct']}/{li_counts['names']['total']}")
                print(f"    Fully Correct Items: {li_counts['fully_correct_items']['correct']}/{li_counts['fully_correct_items']['total']}")
        
        # Document-level errors (abbreviated)
        print("\nDocument-level evaluation (abbreviated):")
        for doc_result in results['extraction']['detailed_results']:
            print(f"\n- {doc_result['source']}:")
            if isinstance(doc_result['results'], dict):
                for field, field_result in doc_result['results'].items():
                    print(f"  {field}: {field_result}")
                
                if "line_item_details" in doc_result:
                    correct_items = sum(1 for item in doc_result['line_item_details'] if item['fully_correct'])
                    total_items = len(doc_result['line_item_details'])
                    print(f"  Line items: {correct_items}/{total_items} fully correct")
                
                if len(doc_result['errors']) > 0:
                    print(f"  {len(doc_result['errors'])} errors (see full report for details)")
            else:
                print(f"  {doc_result['results']}")
        
        print("-" * 80)
        print(f"Full results saved to: evaluation_results/po_extraction_results.json")


def main():
    """Main function to run the evaluation"""
    # Use the folder path from command line if provided, otherwise use default
    import sys
    folder_path = "test_po_dataset"
    
    print(f"Starting evaluation with ground truth using files from {folder_path}")
    evaluator = POEvaluator(folder_path)
    evaluation_results = evaluator.run_evaluation()
    
    if evaluation_results:
        # Get overall F1 score for quick assessment
        f1 = evaluation_results['extraction']['overall_f1']
        print(f"\nEvaluation completed with overall F1 score: {f1:.4f}")
        
        # Provide interpretation
        if f1 > 0.9:
            print("Performance assessment: Excellent")
        elif f1 > 0.8:
            print("Performance assessment: Good")
        elif f1 > 0.6:
            print("Performance assessment: Moderate")
        else:
            print("Performance assessment: Needs improvement")


if __name__ == "__main__":
    main()
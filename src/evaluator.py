# src/evaluator.py
import os
import json
import time
from src.config import EVAL_FILE
from src.agent import AeroAgent

def run_full_ablation_matrix():
    print("--- INITIATING SYSTEM EVALUATION FRAMEWORK RUNNER ---")
    
    if not os.path.exists(EVAL_FILE):
        print(f"Error: Missing evaluation input file path at {EVAL_FILE}!")
        return
        
    with open(EVAL_FILE, 'r', encoding='utf-8') as f:
        questions = [json.loads(line) for line in f]
        
    agent = AeroAgent()
    
    # The 6 exact configurations requested in your grading description
    configurations = ["baseline", "no_planner", "no_reflector", "no_hybrid", "no_citation_verifier", "full_agent"]
    
    output_directory = os.path.join(os.path.dirname(os.path.dirname(__file__)), "predictions")
    os.makedirs(output_directory, exist_ok=True)
    
    for config in configurations:
        print(f"\nEvaluating Configuration Variant Pipeline: [{config}]...")
        output_records = []
        total_latency = 0
        total_tool_calls = 0
        
        for idx, q_record in enumerate(questions):
            print(f" Processing Q{idx+1}/{len(questions)} [ID: {q_record['id']}]...")
            
            result = agent.execute_pipeline(
                question=q_record['question'],
                question_type=q_record['type'],
                config_mode=config
            )
            
            output_payload = {
                "id": q_record['id'],
                "answer": result['answer'],
                "cited_papers": result['cited_papers']
            }
            output_records.append(output_payload)
            
            total_latency += result['metrics']['latency'] if 'metrics' in result else result.get('latency', 0)
            total_tool_calls += result['metrics']['tool_calls'] if 'metrics' in result else result.get('tool_calls', 0)
            
        output_file_path = os.path.join(output_directory, f"{config}.jsonl")
        with open(output_file_path, 'w', encoding='utf-8') as out_f:
            for record in output_records:
                out_f.write(json.dumps(record) + "\n")
                
        print(f" SUCCESS: Results exported cleanly to predictions/{config}.jsonl")
        print(f" Average Latency: {total_latency / len(questions):.4f} seconds")
        print(f" Average Tool Calls: {total_tool_calls / len(questions):.2f} rounds")

if __name__ == "__main__":
    run_full_ablation_matrix()
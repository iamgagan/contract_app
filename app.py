import gradio as gr
import requests
import pandas as pd
import json
import base64
import os

BACKEND_URL = os.getenv("BACKEND_URL", "https://contractsense-backend.onrender.com")

def upload_contract(file):
    if file is None:
        return "Please upload a contract file.", None, None
    
    try:
        filename = file.name
        with open(file.name, 'rb') as f:
            file_content = f.read()
        
        file_content_b64 = base64.b64encode(file_content).decode('utf-8')
        
        response = requests.post(f"{BACKEND_URL}/upload-contract", 
                                 json={"filename": filename, "content": file_content_b64})
        response.raise_for_status()
        conditions = response.json()['conditions']
        
        # Save conditions to a JSON file
        json_filename = "contract_conditions.json"
        with open(json_filename, 'w') as f:
            json.dump(conditions, f, indent=2)
        
        return json.dumps(conditions, indent=2), conditions, json_filename
    except Exception as e:
        return f"An error occurred: {str(e)}", None, None

def analyze_tasks(tasks_file, contract_conditions):
    if tasks_file is None:
        return gr.Dataframe(value=pd.DataFrame({"Error": ["Please upload a tasks file."]})), "No analysis performed.", None
    if contract_conditions is None:
        return gr.Dataframe(value=pd.DataFrame({"Error": ["Please upload and extract contract conditions first."]})), "No analysis performed.", None
    
    try:
        with open(tasks_file.name, 'rb') as f:
            file_content = f.read()
        
        file_content_b64 = base64.b64encode(file_content).decode('utf-8')
        
        data = {
            'filename': tasks_file.name,
            'content': file_content_b64,
            'contract_conditions': json.dumps(contract_conditions)
        }
        response = requests.post(f"{BACKEND_URL}/analyze-tasks", json=data)
        response.raise_for_status()
        results = response.json()['results']
        df = pd.DataFrame(results)
        
        # Save results to a CSV file
        csv_filename = "task_analysis_results.csv"
        df.to_csv(csv_filename, index=False)
        
        return gr.Dataframe(value=df), format_detailed_results(results), csv_filename
    except Exception as e:
        return gr.Dataframe(value=pd.DataFrame({"Error": [f"An error occurred: {str(e)}"]})), "Error occurred during analysis", None

def format_detailed_results(results):
    formatted = ""
    for i, result in enumerate(results, 1):
        formatted += f"Task {i}:\n"
        formatted += f"Compliant: {result['compliant']}\n"
        formatted += f"Reason: {result['reason']}\n"
        formatted += f"Adjusted Amount: ${result['adjusted_amount']:.2f}\n"
        formatted += f"Applied Multipliers: {', '.join(result['applied_multipliers'])}\n"
        formatted += f"Additional Notes: {result['additional_notes']}\n\n"
    return formatted

with gr.Blocks() as demo:
    gr.Markdown("# Contract Analysis App")
    
    contract_conditions = gr.State(None)
    
    with gr.Tab("Upload Contract"):
        contract_input = gr.File(label="Upload Contract File (DOCX)")
        contract_output = gr.Textbox(label="Extracted Conditions", lines=10)
        contract_button = gr.Button("Extract Conditions")
        contract_download = gr.File(label="Download Extracted Conditions (JSON)")
    
    with gr.Tab("Analyze Tasks"):
        tasks_input = gr.File(label="Upload Tasks File (CSV/Excel)")
        tasks_output = gr.Dataframe(label="Analysis Results")
        detailed_output = gr.Textbox(label="Detailed Analysis", lines=10)
        tasks_button = gr.Button("Analyze Tasks")
        results_download = gr.File(label="Download Analysis Results (CSV)")
    
    contract_button.click(
        upload_contract, 
        inputs=contract_input, 
        outputs=[contract_output, contract_conditions, contract_download]
    )
    tasks_button.click(
        analyze_tasks, 
        inputs=[tasks_input, contract_conditions], 
        outputs=[tasks_output, detailed_output, results_download]
    )

demo.launch(share=True)

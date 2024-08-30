import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
import json
import csv
import io
import base64
import re
import logging
from langchain_openai import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from docx import Document
import zipfile
import chardet
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

load_dotenv()  # This loads the .env file

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


# Add the root route here
@app.api_route("/", methods=["GET", "HEAD"])
async def root(request: Request):
    if request.method == "HEAD":
        return {}  # Return an empty response for HEAD requests
    return {"message": "Welcome to ContractSense API", "status": "operational"}

class Task(BaseModel):
    description: str
    amount: float

class FileUpload(BaseModel):
    filename: str
    content: str

class TasksAnalysis(BaseModel):
    filename: str
    content: str
    contract_conditions: str

@app.post("/upload-contract")
async def upload_contract(file: FileUpload):
    logger.info(f"Received file: {file.filename}")
    if not file.filename.endswith('.docx'):
        logger.warning(f"Invalid file type: {file.filename}")
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a .docx file.")
    
    try:
        # Decode base64 content
        content = base64.b64decode(file.content)
        logger.info(f"Successfully decoded base64 content. Size: {len(content)} bytes")
        logger.info(f"First 20 bytes of content: {content[:20]}")
        
        # Attempt to open the content as a DOCX file
        try:
            contract_text = extract_text_from_docx(content)
            logger.info(f"Successfully extracted text from DOCX. Text length: {len(contract_text)}")
        except zipfile.BadZipFile:
            logger.error("The uploaded file is not a valid DOCX file.")
            raise HTTPException(status_code=400, detail="The uploaded file is not a valid DOCX file. Please ensure you're uploading a proper Microsoft Word document.")
        except Exception as docx_error:
            logger.error(f"Failed to read DOCX file: {str(docx_error)}")
            raise HTTPException(status_code=400, detail=f"Failed to read DOCX file: {str(docx_error)}")
        
        conditions = extract_contract_conditions(contract_text)
        logger.info("Successfully extracted contract conditions")
        return JSONResponse(content={"conditions": conditions})
    except base64.binascii.Error as be:
        logger.error(f"Invalid base64 encoding: {str(be)}")
        raise HTTPException(status_code=400, detail="Invalid file encoding. Please try uploading the file again.")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

@app.post("/analyze-tasks")
async def analyze_tasks(data: TasksAnalysis):
    try:
        # Decode base64 content
        content = base64.b64decode(data.content)
        tasks = parse_file(content, data.filename)
        results = analyze_task_compliance(tasks, json.loads(data.contract_conditions))
        return {"results": results}
    except base64.binascii.Error as be:
        logger.error(f"Invalid base64 encoding: {str(be)}")
        raise HTTPException(status_code=400, detail="Invalid file encoding. Please try uploading the file again.")
    except json.JSONDecodeError as je:
        logger.error(f"Invalid JSON in contract conditions: {str(je)}")
        raise HTTPException(status_code=400, detail="Invalid JSON in contract conditions")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

def extract_text_from_docx(content):
    doc = Document(io.BytesIO(content))
    return "\n".join([paragraph.text for paragraph in doc.paragraphs])

def extract_contract_conditions(content):
    llm = OpenAI(temperature=0)
    prompt = PromptTemplate(
        input_variables=["content"],
        template="""
        Extract key conditions from this contract, providing a more detailed structure:
        {content}
        Provide the output as a JSON string with the following structure:
        {{
            "travel_provisions": {{
                "budget_caps": {{
                    "single_trip": float,
                    "daily_expenses": float
                }},
                "multipliers": {{
                    "night_weekend": float,
                    "seasonal_location": float,
                    "urgency": float
                }},
                "travel_class": {{
                    "domestic": string,
                    "international": {{
                        "duration_threshold": int,
                        "class": string
                    }}
                }},
                "special_circumstances": {{
                    "weather_allowance": float,
                    "health_safety_covered": boolean
                }},
                "high_cost_locations": {{
                    "increase_percentage": float,
                    "approval_required": boolean
                }},
                "seasonal_adjustments": {{
                    "increase_percentage": float
                }}
            }},
            "pre_approval_required": boolean,
            "expense_report_deadline": int
        }}
        """
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    result = chain.run(content=content)
    
    logger.info(f"Raw API response: {result}")
    
    # Try to extract JSON from the response
    json_match = re.search(r'\{.*\}', result, re.DOTALL)
    if json_match:
        json_str = json_match.group(0)
        logger.info(f"Extracted JSON string: {json_str}")
    else:
        logger.error("No JSON-like structure found in the API response")
        raise ValueError("The API response does not contain a valid JSON structure")

    try:
        parsed_result = json.loads(json_str)
        return parsed_result
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON: {e}")
        logger.error(f"Problematic JSON string: {json_str}")
        raise ValueError("Failed to parse the extracted JSON structure.")

def parse_text_response(response, original_amount):
    compliant = "is compliant" in response.lower()
    
    if compliant:
        reason = "Task is compliant with contract conditions"
    else:
        reason_match = re.search(r'(?:The task is not compliant because|The reason for (?:non-)?compliance is) (.*?)(?:\.|$)', response, re.IGNORECASE | re.DOTALL)
        reason = reason_match.group(1) if reason_match else "Unable to extract reason"
    
    adjusted_amount_match = re.search(r'adjusted amount:?\s*\$?([\d,]+(?:\.\d+)?)', response, re.IGNORECASE)
    adjusted_amount = float(adjusted_amount_match.group(1).replace(',', '')) if adjusted_amount_match else original_amount
    
    multipliers = re.findall(r'(\w+)(?: multiplier| adjustment)(?:s?):?\s*([\d.]+)', response, re.IGNORECASE)
    applied_multipliers = [f"{m[0]}: {m[1]}" for m in multipliers if m[1] != '0']
    
    additional_notes = response.split('Additional notes:', 1)[-1].strip() if 'Additional notes:' in response else response

    return {
        "compliant": compliant,
        "reason": reason,
        "adjusted_amount": adjusted_amount,
        "applied_multipliers": applied_multipliers,
        "additional_notes": additional_notes
    }

def analyze_task_compliance(tasks, conditions):
    llm = OpenAI(temperature=0)
    prompt = PromptTemplate(
        input_variables=["task", "conditions"],
        template="""
        Analyze if this task complies with the contract conditions:
        Task: {task}
        Conditions: {conditions}
        Provide a detailed analysis, considering all relevant factors such as budget caps, multipliers, travel class, special circumstances, and location-based adjustments.
        Start your response with "The task is compliant" or "The task is not compliant", followed by the reason.
        Then, state the adjusted amount (if applicable) as "Adjusted amount: $X".
        List any applied multipliers as "Applied multipliers: X: Y, Z: W".
        Finally, provide any additional notes or explanations under "Additional notes:".
        """
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    results = []
    for task in tasks:
        try:
            result = chain.run(task=json.dumps(task.dict()), conditions=json.dumps(conditions))
            logger.info(f"Raw API response for task analysis: {result}")
            
            parsed_result = parse_text_response(result, task.amount)
            results.append(parsed_result)
        except Exception as e:
            logger.error(f"Unexpected error during task analysis: {e}")
            results.append({
                "compliant": False,
                "reason": f"Error in analysis: Unexpected error occurred",
                "adjusted_amount": task.amount,
                "applied_multipliers": [],
                "additional_notes": f"Error details: {str(e)}\nRaw API response: {result}"
            })
    return results


def parse_file(content, filename):
    if filename.endswith('.csv'):
        return parse_csv(content)
    elif filename.endswith(('.xls', '.xlsx')):
        return parse_excel(content)
    else:
        raise ValueError(f"Unsupported file type: {filename}")

def parse_csv(content):
    # Detect the file encoding
    detected = chardet.detect(content)
    file_encoding = detected['encoding']
    logger.info(f"Detected file encoding: {file_encoding}")

    # Try to decode the content with the detected encoding
    try:
        decoded_content = content.decode(file_encoding)
    except UnicodeDecodeError:
        logger.warning(f"Failed to decode with {file_encoding}, falling back to 'latin-1'")
        decoded_content = content.decode('latin-1')

    # Use csv.Sniffer to detect the dialect
    dialect = csv.Sniffer().sniff(decoded_content)
    logger.info(f"Detected CSV dialect: {dialect}")

    csv_reader = csv.DictReader(io.StringIO(decoded_content), dialect=dialect)
    return parse_rows(csv_reader)

def parse_excel(content):
    df = pd.read_excel(io.BytesIO(content))
    return parse_rows(df.to_dict('records'))

def parse_rows(rows):
    tasks = []
    for row in rows:
        try:
            amount = float(str(row['Amount']).replace('$', '').replace(',', ''))
            tasks.append(Task(description=row['Task Description'], amount=amount))
        except KeyError as e:
            logger.error(f"Missing required column: {e}")
            raise ValueError(f"File is missing required column: {e}")
        except ValueError as e:
            logger.error(f"Invalid amount format: {e}")
            raise ValueError(f"Invalid amount format in file: {e}")
    return tasks

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

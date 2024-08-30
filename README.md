# ContractSense: AI-Powered Contract and Task Compliance Analyzer

ContractSense is an advanced, AI-driven system designed to streamline contract analysis and task compliance checking. By leveraging the power of natural language processing and machine learning, ContractSense extracts key terms from complex contracts and evaluates tasks for compliance, saving time and reducing human error in contract management processes.

## 🌟 Features

- **Intelligent Contract Analysis**: Automatically extract key conditions from uploaded DOCX files using state-of-the-art NLP techniques.
- **Task Compliance Checking**: Analyze tasks from CSV or Excel files against extracted contract conditions with high accuracy.
- **User-Friendly Interface**: Intuitive Gradio-based frontend for easy interaction and quick results.
- **Detailed Compliance Reports**: Get comprehensive analysis including compliance status, reasons for non-compliance, adjusted amounts, and applied multipliers.
- **Exportable Results**: Download analysis results in CSV format and extracted contract conditions in JSON for further use or record-keeping.
- **Multi-Format Support**: Process contracts in DOCX format and tasks in both CSV and Excel formats.

## 🛠️ Technology Stack

- **Backend**: FastAPI, LangChain, OpenAI GPT
- **Frontend**: Gradio
- **File Processing**: python-docx, pandas, chardet
- **AI/ML**: OpenAI API
- **Data Handling**: JSON, CSV

## 🚀 Installation

1. Clone the repository:
   ```
   git clone https://github.com/your-username/contractsense.git
   cd contractsense
   ```

2. Set up a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Set up your OpenAI API key:
   - Create a `.env` file in the project root.
   - Add your OpenAI API key: `OPENAI_API_KEY=your_api_key_here`

## 🖥️ Usage

1. Start the backend server:
   ```
   python main.py
   ```

2. In a new terminal, launch the Gradio interface:
   ```
   python app.py
   ```

3. Open the provided URL in your web browser to access the ContractSense interface.

4. Upload a contract file (DOCX format) to extract conditions.

5. Upload a tasks file (Excel format) to analyze compliance.

6. View the detailed results in the interface and download them for further analysis or reporting.

## 📁 File Format Requirements

- **Contracts**: Microsoft Word documents (.docx)
- **Tasks**: CSV or Excel files with the following columns:
  - "Task Description": A detailed description of the task
  - "Amount": The cost associated with the task (numerical value)

## 🤝 Contributing

We welcome contributions to ContractSense! If you have suggestions for improvements or encounter any issues, please feel free to:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 🙏 Acknowledgments

- OpenAI for providing the powerful GPT models
- Gradio developers for making it easy to create ML web interfaces

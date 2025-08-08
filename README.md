# Proofread XML Pipeline

This project automatically proofreads `<p>` elements inside XML files using **OpenAI GPT models** plus lightweight heuristic rules, and inserts `<error>` tags describing each detected issue.

## 📌 Features
- Processes **one file** or **all files** in a folder.
- Uses **OpenAI API** for grammar, spelling, punctuation, and style suggestions.
- Adds `<error>` elements with:
  - `type` — category of issue (grammar, spelling, punctuation, etc.)
  - `correction` — suggested fix
  - `reason` — explanation
- Includes **deterministic heuristics** for common mistakes missed by the model.
- Works in **single-file** or **batch (directory)** mode.
- Generates a JSON summary with runtime, memory usage, and modification stats.

---

## 📂 Project Structure
```

proofread_xml/
│
├── proofread_xml.py     # Main CLI entry point
├── genai_client.py      # OpenAI API integration
├── xml_utils.py         # XML parsing and <error> tag insertion
├── requirements.txt     # Python dependencies
├── .env                 # Configuration file
└── README.md            # This file

````

---

## ⚙️ Installation

### 1️⃣ Create Conda Environment
```bash
conda create -n proofread-env python=3.11 -y
conda activate proofread-env
````

### 2️⃣ Install Requirements

```bash
pip install -r requirements.txt
```

---

## 🔑 Configuration

Create a `.env` file in the project root:

```env
# OpenAI API key
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Input folder or single file
INPUT_DIR=input_file_path
OUTPUT_DIR=output_file_path
# Optional suffix to add before .xml in output files
OUTPUT_SUFFIX=_proofread

```

---

## 🚀 Usage


### **Process a specific file**

```bash
python proofread_xml.py --lang en 
```


---

## 📊 Example Output Summary

When you run the script, you’ll get JSON output like:

```json
{
  "file": "input/sample.xml",
  "time_seconds": 5.23,
  "mem_peak_mb": 6.46,
  "total_p": 2,
  "modified_p": 2,
  "error_tags": 6,
  "invariant_score": 0.0,
  "output": "output/sample.xml"
}
```

---

## 🧠 How It Works

1. **Load XML** from the input path.
2. **Extract `<p>` elements**.
3. **Send text to OpenAI API** for proofreading.
4. **Apply heuristic rules** for common issues.
5. **Insert `<error>` tags** in the correct positions without altering the original text flow.
6. **Write output XML** to the specified folder.
7. **Print JSON summary** of processing.

---

## ⚠️ Notes

* Requires an **active OpenAI API key** with GPT access.
* `invariant_score` should ideally be `1.0`; a value of `0.0` means text structure changed unintentionally.
* Large files may take longer because each `<p>` is processed individually.

---

## 📄 License

MIT License


# Auto-Docstring Generator (Qwen2.5-Coder Fine-Tuning) 🚀

This project focuses on fine-tuning the **Qwen2.5-Coder-1.5B-Instruct** model to automatically generate high-quality **Google-style docstrings** for Python code.

The project utilizes **DPO (Direct Preference Optimization)** to align the model's outputs with strict formatting rules, ensuring valid and helpful documentation. It also includes a synthetic data generation pipeline using **Groq**.

## 🌟 Key Features

* **Efficient Fine-Tuning:** Uses [Unsloth](https://github.com/unslothai/unsloth) for 2x faster training and 70% less memory usage.
* **DPO Training:** Implements Direct Preference Optimization using the `trl` library to prefer high-quality Google-style docstrings over vague ones.
* **Synthetic Data Generation:** Includes a script (`load_data.py`) to generate "chosen" vs "rejected" datasets using **Llama-3.3-70b** via the Groq API.
* **Strict Verification:** Includes an inference pipeline that uses Python's `ast` (Abstract Syntax Tree) to verify that generated docstrings match the function signatures and adhere to Google style guidelines.

## 📂 Project Structure

```text
fine_tuning_qwen2.5_docstring/
├── .venv/                 # Virtual environment
├── notebooks/             # Jupyter Notebooks for training
│   └── Qwen2_5_Coder_1_5B.ipynb
├── src/                   # Source scripts
│   ├── load_data.py       # Data generation script (Groq API)
│   └── main.py
├── .env                   # Environment variables (API Keys)
├── .gitignore             # Git ignore file
├── pyproject.toml         # Project configuration
├── uv.lock                # Dependency lock file (managed by uv)
└── README.md              # Project documentation

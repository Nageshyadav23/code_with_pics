📸 Code with Pics

Handwritten Code Recognition, Execution & AI Assistance on Mobile

📌 Overview

Code with Pics is a mobile-based intelligent coding platform that converts handwritten programming code images into clean, executable digital code. It leverages Transformer-based OCR (TrOCR), Supabase-powered error correction, continuous learning, and an AI coding assistant to provide an end-to-end solution for recognizing, editing, compiling, and optimizing code directly from handwritten inputs.

The system supports Python, C, and Java, enabling users to write logic naturally using pen and paper, capture it via a mobile camera, and execute it instantly without manual retyping.

✨ Key Features

Transformer-based OCR for handwritten code recognition

Image preprocessing & line segmentation for improved accuracy

Supabase-powered error correction with continuous learning

Built-in mobile code editor for manual editing

Multi-language compilation & execution (Python, C, Java)

AI-powered assistance for debugging, explanation, and optimization

🧠 Technologies Used

React Native – Mobile frontend

Python – Backend & ML pipeline

Flask – API layer

PyTorch – TrOCR fine-tuning

TrOCR (Transformer OCR) – Handwritten text recognition

Supabase – Keyword mapping & adaptive learning

Ngrok – Secure backend exposure

Google Colab – Cloud backend & model hosting

Meta-LLaMA (HuggingFace) – AI code assistant

📁 Project Structure

Code-with-Pics/
│
├── GoogleColab/
│   ├── main.py
│   └── train.py
│
└── VSCode/
    ├── backend/
    │   └── app.py
    │
    └── frontend/
        ├── home.py
        ├── editor.py
        ├── changeScreen.py
        ├── aiSupport.py
        └── codeeditor.py



📂 Folder & File Description
🔹 GoogleColab/

This folder contains backend logic and model training code designed to run on Google Colab.

main.py

Core backend file

Defines all Flask routes

Loads the fine-tuned TrOCR model

Generates and manages the Ngrok public URL

Handles communication between mobile frontend and cloud backend

Executes Python, C, and Java programs

train.py

Used to fine-tune the TrOCR model

Trains on a custom handwritten code dataset (CSV/JSONL)

Implements supervised training using PyTorch

Produces a domain-specific OCR model optimized for programming syntax

🔹 VSCode/backend/

This folder handles AI-powered assistance.

app.py

Integrates Meta-LLaMA using HuggingFace APIs

Provides endpoints for:

Code debugging

Code explanation

Language conversion

Code optimization

Communicates directly with the frontend AI support tab

🔹 VSCode/frontend/

This folder contains the React Native UI logic, organized into multiple functional screens.

home.py

Image capture and gallery selection

Entry point of the application

editor.py

Displays raw OCR output and cleaned code

Allows manual corrections

changeScreen.py

Handles navigation and tab switching

aiSupport.py

Chat interface for AI-based code assistance

codeeditor.py

Integrated code editor

Supports syntax highlighting

Enables code compilation and execution

⚙️ Workflow Summary

Capture handwritten code image using mobile camera

Apply preprocessing and line segmentation

Recognize text using fine-tuned TrOCR

Auto-correct errors using Supabase mappings

Edit code in built-in editor

Compile & execute code (Python / C / Java)

Optimize or debug using AI assistant

Store corrections for continuous learning

🎯 Goal

To streamline the process of converting handwritten code images into executable, optimized programs within a single, mobile-friendly, AI-powered platform, reducing dependence on laptops and manual retyping.

🚀 Future Enhancements

Support for more programming languages (C++, JavaScript)

Multi-writer handwriting dataset expansion

Offline OCR with on-device inference

Stylus-based real-time handwriting input

Cloud migration for production deployment

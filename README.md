# Smart-MCQ-Solver-Challenge
Name: Abhishek Kumar
ID: 23f2004693

Smart MCQ Solver:
A deep-learning project for solving 5-option multiple-choice questions (A–E) by ranking the most likely answers and returning the top 3 predictions.
Student: Abhishek Kumar
Roll Number: 23F2004693
Project Overview
The project treats MCQ solving as a classification/ranking problem rather than free-text generation.
For each question, the model receives:
- Question / prompt
- Five answer options: A, B, C, D, E
It produces a probability score for each option and ranks them from most to least likely.
Example:
Question + 5 options
        ↓
      Model
        ↓
[A: 0.08, B: 0.12, C: 0.64, D: 0.10, E: 0.06]
        ↓
Top-3 Prediction: C B D
Models Explored
1. BiLSTM Baseline
A Bidirectional LSTM model was trained from scratch to learn relationships between questions and answer options. It was used as a baseline before moving to pretrained transformer models.
2. DeBERTa-v3
The main approach uses DeBERTa-v3 fine-tuned for multiple-choice classification.
Each question is paired with all five options, tokenized, passed through DeBERTa, and converted into five logits. Softmax converts these logits into probabilities.
Question + Options
      ↓
Tokenizer
      ↓
DeBERTa
      ↓
5 logits
      ↓
Softmax
      ↓
Option probabilities
3. Qwen
A Qwen instruction model was also tested for scoring the answer choices. Its predictions were compared and combined with the other models during ensemble experiments.
Validation Strategy
The dataset is divided into training and validation data using a group-aware split to reduce information leakage between similar questions.
Performance is evaluated using MAP@3 (Mean Average Precision at 3).
For a correct answer:
- Rank 1 → 1.0
- Rank 2 → 0.5
- Rank 3 → 0.33
- Outside Top 3 → 0
This metric is useful because the task requires good answer ranking, not only top-1 accuracy.
Ensemble
Predictions from multiple models were converted into comparable probability scores and tested in different weighted combinations.
BiLSTM probabilities
        +
DeBERTa probabilities
        +
Qwen probabilities
        ↓
Weighted Ensemble
        ↓
Final Top-3 Answers
The best weights are selected using validation MAP@3.
Deployment
A simple Gradio interface is provided for interactive testing.
The user enters:
1. A question
2. Option A
3. Option B
4. Option C
5. Option D
6. Option E
The application returns the top three predicted answers with confidence scores.
Project Structure
├── notebooks/              # Training and experiments
├── deployment/             # Gradio application and model config
├── config.json             # Model configuration
├── tokenizer files         # Tokenizer configuration
└── README.md
Running the Project
Install the main dependencies:
pip install torch transformers pandas numpy scikit-learn gradio
Run the notebook to reproduce the training and experiments.
For deployment, place the trained model weights with the tokenizer/configuration files and run the Gradio application.
Key Techniques Used
- Natural Language Processing
- Deep Learning
- BiLSTM
- Transformer fine-tuning
- DeBERTa-v3
- Qwen
- Softmax classification
- MAP@3 ranking metric
- Model ensembling
- Gradio deployment
Limitations
- Performance depends heavily on the training data quality and domain.
- Large transformer models require significant GPU memory.
- Very high validation scores should be checked carefully for possible data leakage or dataset shortcuts.
- The final trained model weights are not included in the current repository ZIP.
Future Improvements
- Train on a larger and more diverse MCQ dataset.
- Improve confidence calibration.
- Experiment with stronger transformer models.
- Use cross-validation for more reliable evaluation.
- Optimize the model for faster inference.
Summary
The project demonstrates an end-to-end deep-learning pipeline for MCQ solving. It starts with a custom BiLSTM baseline, explores transformer-based approaches such as DeBERTa and Qwen, evaluates predictions using MAP@3, combines models through ensembling, and finally provides an interactive Gradio interface for inference.


import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForMultipleChoice

MODEL_PATH = "."

OPTIONS = ["A", "B", "C", "D", "E"]

STUDENT_NAME = "Abhishek Kumar"
ROLL_NUMBER = "23F2004693"

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForMultipleChoice.from_pretrained(
    MODEL_PATH
)

model.to(device)
model.eval()


def predict(question, option_a, option_b,
            option_c, option_d, option_e):

    choices = [
        option_a,
        option_b,
        option_c,
        option_d,
        option_e
    ]

    if not question.strip():
        return "Please enter a question."

    if any(not str(x).strip() for x in choices):
        return "Please enter all five options."

    encoded = tokenizer(
        [question] * 5,
        choices,
        truncation=True,
        padding=True,
        max_length=256,
        return_tensors="pt"
    )

    encoded = {
        key: value.unsqueeze(0).to(device)
        for key, value in encoded.items()
    }

    with torch.no_grad():
        output = model(**encoded)

    probabilities = torch.softmax(
        output.logits,
        dim=1
    )[0]

    ranking = torch.argsort(
        probabilities,
        descending=True
    ).tolist()

    result = []

    for rank, index in enumerate(ranking[:3], 1):

        result.append(
            f"{rank}. Option {OPTIONS[index]} - "
            f"{choices[index]} "
            f"({probabilities[index].item():.2%})"
        )

    return "\n".join(result)


with gr.Blocks(title="Smart MCQ Solver") as demo:

    gr.Markdown(
        f"""
# Smart MCQ Solver

**Student Name:** {STUDENT_NAME}  
**Roll Number:** {ROLL_NUMBER}

Enter a question and five possible answers.
The trained DeBERTa model will return its top-3 predictions.
"""
    )

    question = gr.Textbox(
        label="Question",
        lines=4
    )

    option_a = gr.Textbox(label="Option A")
    option_b = gr.Textbox(label="Option B")
    option_c = gr.Textbox(label="Option C")
    option_d = gr.Textbox(label="Option D")
    option_e = gr.Textbox(label="Option E")

    button = gr.Button(
        "Predict Top 3 Answers"
    )

    output = gr.Textbox(
        label="Predictions",
        lines=5
    )

    button.click(
        predict,
        inputs=[
            question,
            option_a,
            option_b,
            option_c,
            option_d,
            option_e
        ],
        outputs=output
    )

    gr.Markdown(
        f"""
---
Smart MCQ Solver  
**{STUDENT_NAME} — {ROLL_NUMBER}**
"""
    )


if __name__ == "__main__":
    demo.launch()

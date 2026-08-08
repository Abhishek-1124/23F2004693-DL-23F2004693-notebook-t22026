import gradio as gr
import torch
import spaces

from transformers import AutoTokenizer, AutoModelForMultipleChoice

MODEL_PATH = "."
OPTIONS = ["A", "B", "C", "D", "E"]

STUDENT_NAME = "Abhishek Kumar"
ROLL_NUMBER = "23F2004693"

device = torch.device("cuda")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForMultipleChoice.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()

@spaces.GPU(duration=30)
def predict(question, option_a, option_b, option_c, option_d, option_e):
    question = str(question).strip()

    choices = [
        str(option_a).strip(),
        str(option_b).strip(),
        str(option_c).strip(),
        str(option_d).strip(),
        str(option_e).strip(),
    ]

    if not question:
        return "Please enter a question."

    if any(not choice for choice in choices):
        return "Please enter all five options."

    encoded = tokenizer(
        [question] * 5,
        choices,
        truncation=True,
        padding=True,
        max_length=256,
        return_tensors="pt",
    )

    encoded = {
        key: value.unsqueeze(0).to(device)
        for key, value in encoded.items()
    }

    with torch.no_grad():
        logits = model(**encoded).logits

    probabilities = torch.softmax(logits, dim=1)[0]

    ranking = torch.argsort(
        probabilities,
        descending=True,
    ).tolist()

    result = []

    if probabilities[ranking[0]].item() < 0.23:
        result.append(
            "⚠️ Low-confidence prediction: "
            "the model is almost equally unsure about all five options."
        )
        result.append("")

    for rank, index in enumerate(
        ranking[:3],
        start=1,
    ):
        result.append(
            f"{rank}. Option {OPTIONS[index]} - "
            f"{choices[index]} "
            f"({probabilities[index].item():.2%})"
        )

    return "\n".join(result)

with gr.Blocks(
    title="Smart MCQ Solver",
    theme=gr.themes.Soft(),
) as demo:

    gr.Markdown(
        f"""
# 🧠 Smart MCQ Solver
### AI-powered multiple choice answer ranking
**Student Name:** {STUDENT_NAME}  
**Roll Number:** {ROLL_NUMBER}
Enter one question and all five options.
> This model was trained for the Smart MCQ Solver competition dataset.
> It is not guaranteed to answer unrelated general-knowledge questions correctly.
"""
    )

    with gr.Row():
        with gr.Column(scale=2):
            question = gr.Textbox(
                label="Question",
                lines=5,
            )

            option_a = gr.Textbox(label="Option A")
            option_b = gr.Textbox(label="Option B")
            option_c = gr.Textbox(label="Option C")
            option_d = gr.Textbox(label="Option D")
            option_e = gr.Textbox(label="Option E")

            with gr.Row():
                predict_button = gr.Button(
                    "🔍 Predict Top 3",
                    variant="primary",
                )
                clear_button = gr.Button("Clear")

        with gr.Column(scale=1):
            gr.Markdown("### Model Prediction")

            output = gr.Textbox(
                label="Top 3 Predictions",
                lines=12,
                interactive=False,
            )

    predict_button.click(
        fn=predict,
        inputs=[
            question,
            option_a,
            option_b,
            option_c,
            option_d,
            option_e,
        ],
        outputs=output,
    )

    clear_button.click(
        fn=lambda: ("", "", "", "", "", "", ""),
        outputs=[
            question,
            option_a,
            option_b,
            option_c,
            option_d,
            option_e,
            output,
        ],
    )

if __name__ == "__main__":
    demo.launch()

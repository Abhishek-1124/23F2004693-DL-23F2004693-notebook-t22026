with gr.Blocks(
    title="Smart MCQ Solver",
    theme=gr.themes.Soft()
) as demo:

    gr.Markdown(
        f"""
# 🧠 Smart MCQ Solver

### AI-powered multiple choice answer ranking

**Student Name:** {STUDENT_NAME}  
**Roll Number:** {ROLL_NUMBER}

Enter a question and all five options below.  
The model will rank the **top 3 answers** with confidence scores.
"""
    )

    with gr.Row():

        with gr.Column(scale=2):

            question = gr.Textbox(
                label="Question",
                lines=5,
                placeholder="Enter your question here..."
            )

            gr.Markdown("### Answer Options")

            option_a = gr.Textbox(
                label="Option A",
                placeholder="Enter option A"
            )

            option_b = gr.Textbox(
                label="Option B",
                placeholder="Enter option B"
            )

            option_c = gr.Textbox(
                label="Option C",
                placeholder="Enter option C"
            )

            option_d = gr.Textbox(
                label="Option D",
                placeholder="Enter option D"
            )

            option_e = gr.Textbox(
                label="Option E",
                placeholder="Enter option E"
            )

            with gr.Row():

                button = gr.Button(
                    "🔍 Predict Top 3",
                    variant="primary"
                )

                clear_button = gr.Button(
                    "Clear"
                )

        with gr.Column(scale=1):

            gr.Markdown(
                """
### Model Prediction

The model ranks each option based on its learned probability.
"""
            )

            output = gr.Textbox(
                label="Top 3 Predictions",
                lines=10,
                interactive=False,
                placeholder="Predictions will appear here..."
            )

    button.click(
        fn=predict,
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

    clear_button.click(
        fn=lambda: ("", "", "", "", "", "", ""),
        outputs=[
            question,
            option_a,
            option_b,
            option_c,
            option_d,
            option_e,
            output
        ]
    )

    gr.Markdown(
        f"""
---

### About this project

This application uses a fine-tuned **DeBERTa multiple-choice model**
to rank answers for MCQ questions.

**Student:** {STUDENT_NAME}  
**Roll Number:** {ROLL_NUMBER}
"""
    )

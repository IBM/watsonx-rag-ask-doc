#
# Copyright IBM Corp. 2024
# SPDX-License-Identifier: Apache-2.0
#

import gradio as gr
from utils.upload_file import UploadFile
from utils.upload_data_manually import UploadDataManually
from utils.chatbot import ChatBot
from utils.ui_settings import UISettings


with gr.Blocks() as demo:
    with gr.Tabs():
        with gr.TabItem("ASK-DOC"):
            with gr.Row() as row_one:
                with gr.Column(visible=False) as reference_bar:
                    ref_output = gr.Markdown()

                with gr.Column() as chatbot_output:
                    chatbot = gr.Chatbot(
                        [],
                        elem_id="chatbot",
                        bubble_full_width=False,
                        height=500,
                        avatar_images=(("images/user.png"), "images/bot.png"),
                        # render=False
                    )
                    # **Adding like/dislike icons
                    chatbot.like(UISettings.feedback, None, None)
            with gr.Row():
                input_txt = gr.Textbox(
                    # lines=4,
                    scale=8,
                    placeholder="Enter text and press enter, or upload PDF files",
                    container=False,
                )

            with gr.Row() as row_two:
                sidebar_state = gr.State(False)
                btn_toggle_sidebar = gr.Button(value="References")
                btn_toggle_sidebar.click(
                    UISettings.toggle_sidebar,
                    [sidebar_state],
                    [reference_bar, sidebar_state],
                )
                upload_btn = gr.UploadButton(
                    "📁 Upload files",
                    file_types=[".pdf"],
                    file_count="multiple",
                )
                upload_btn_for_preprocess = gr.UploadButton(
                    "📁 Preprocess files",
                    file_types=[".pdf"],
                    file_count="multiple",
                )
                temperature_bar = gr.Slider(
                    minimum=0,
                    maximum=1,
                    value=0,
                    step=0.1,
                    label="Temperature",
                    info="Choose between 0 and 1",
                )
                rag_with_dropdown = gr.Dropdown(
                    label="Chat with",
                    choices=[
                        "Preprocessed doc",
                        "Upload doc: Process for RAG",
                        "Upload doc: Give Full summary",
                    ],
                    value="Preprocessed doc",
                )
                clear_button = gr.ClearButton([input_txt, chatbot])

            file_msg = upload_btn.upload(
                fn=UploadFile.process_uploaded_files,
                inputs=[upload_btn, chatbot, rag_with_dropdown],
                outputs=[input_txt, chatbot],
                queue=False,
            )

            processed_file_msg = upload_btn_for_preprocess.upload(
                fn=UploadDataManually.upload_data_manually,
                inputs=[upload_btn_for_preprocess, chatbot],
                outputs=[input_txt, chatbot],
                queue=False,
            )

            txt_msg = input_txt.submit(
                fn=ChatBot.respond,
                inputs=[chatbot, input_txt, rag_with_dropdown, temperature_bar],
                outputs=[input_txt, chatbot, ref_output],
                queue=False,
            ).then(lambda: gr.Textbox(interactive=True), None, [input_txt], queue=False)


if __name__ == "__main__":
    demo.launch()

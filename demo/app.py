
import os
import gradio as gr
from PIL import ImageEnhance

print("[1/4] Importing PairLIE inference module...", flush=True)
from inference import enhance_image

print("[2/4] PairLIE model loaded successfully!", flush=True)


def process_image(image, brightness_factor):
    if image is None:
        raise gr.Error("Please upload an image first.")

    print("[3/4] Processing uploaded image...", flush=True)

    original = image.convert("RGB")

    brightness = ImageEnhance.Brightness(original).enhance(
        brightness_factor
    )

    enhanced = enhance_image(original)

    print("[4/4] Image processing completed!", flush=True)

    return original, brightness, enhanced


with gr.Blocks() as demo:

    gr.Markdown("""
    # PairLIE: Low-Light Image Enhancement

    ### Deep Learning vs. Traditional Brightness Adjustment

    Upload a low-light photograph and compare the results.
    """)

    with gr.Row():
        input_image = gr.Image(
            type="pil",
            label="Upload Low-Light Image"
        )

        with gr.Column():
            brightness_factor = gr.Slider(
                minimum=1.0,
                maximum=3.0,
                value=2.0,
                step=0.1,
                label="Brightness Adjustment Factor"
            )

            process_button = gr.Button(
                "Enhance Image",
                variant="primary"
            )

    gr.Markdown("## Results")

    with gr.Row():
        original_output = gr.Image(label="Original")
        brightness_output = gr.Image(label="Brightness Adjustment")
        pairlie_output = gr.Image(label="PairLIE Enhancement")

    process_button.click(
        fn=process_image,
        inputs=[input_image, brightness_factor],
        outputs=[
            original_output,
            brightness_output,
            pairlie_output
        ]
    )


if __name__ == "__main__":

    is_colab = os.path.exists("/content")

    print("Starting PairLIE demo...", flush=True)

    demo.launch(
        share=is_colab,
        server_name="0.0.0.0",
        show_error=True
    )

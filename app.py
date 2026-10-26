import os
import re
import tempfile
from datetime import datetime

import gradio as gr
import torch
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageDraw, ImageFont
from transformers import BlipProcessor, BlipForConditionalGeneration

# -------------------- SETTINGS --------------------
MODEL_NAME = "Salesforce/blip-image-captioning-base"
MAX_IMAGE_SIZE = 1600

print("Loading VisionAI image understanding model...")
processor = BlipProcessor.from_pretrained(MODEL_NAME)
model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME)
model.eval()
print("VisionAI model is ready!")


# -------------------- IMAGE HELPERS --------------------
def prepare_image(image):
    if image is None:
        raise gr.Error("Please upload a photo first.")
    image = image.convert("RGB")
    image.thumbnail((MAX_IMAGE_SIZE, MAX_IMAGE_SIZE))
    return image


def analyze_image(image):
    image = prepare_image(image)
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=35, num_beams=3)
    caption = processor.decode(output[0], skip_special_tokens=True).strip()
    if caption:
        caption = caption[0].upper() + caption[1:]
    return caption or "A photo ready for creative editing."


def apply_filter(image, filter_name, intensity=70):
    image = prepare_image(image)
    strength = max(0, min(100, int(intensity))) / 100.0
    original = image.copy()

    if filter_name == "Original":
        result = original
    elif filter_name == "Vintage":
        warm = ImageOps.colorize(ImageOps.grayscale(original), "#2d2118", "#f6d7a7")
        result = Image.blend(original, warm, 0.48 * strength)
        result = ImageEnhance.Contrast(result).enhance(1 + 0.18 * strength)
        result = ImageEnhance.Color(result).enhance(0.78)
    elif filter_name == "Cinematic":
        result = ImageEnhance.Color(original).enhance(0.78)
        result = ImageEnhance.Contrast(result).enhance(1 + 0.38 * strength)
        overlay = Image.new("RGB", result.size, (15, 36, 48))
        result = Image.blend(result, overlay, 0.12 * strength)
    elif filter_name == "Retro":
        result = ImageEnhance.Color(original).enhance(0.70)
        result = ImageEnhance.Contrast(result).enhance(1 + 0.25 * strength)
        overlay = Image.new("RGB", result.size, (205, 120, 65))
        result = Image.blend(result, overlay, 0.14 * strength)
    elif filter_name == "Black & White":
        result = ImageOps.grayscale(original).convert("RGB")
        result = ImageEnhance.Contrast(result).enhance(1 + 0.22 * strength)
    elif filter_name == "Dreamy Glow":
        blurred = original.filter(ImageFilter.GaussianBlur(radius=2 + 7 * strength))
        result = Image.blend(original, blurred, 0.28 * strength)
        result = ImageEnhance.Brightness(result).enhance(1 + 0.12 * strength)
        result = ImageEnhance.Color(result).enhance(1 + 0.10 * strength)
    elif filter_name == "Soft Blur":
        blurred = original.filter(ImageFilter.GaussianBlur(radius=1 + 8 * strength))
        result = Image.blend(original, blurred, 0.85 * strength)
    elif filter_name == "Warm Sunset":
        overlay = Image.new("RGB", original.size, (255, 145, 65))
        result = Image.blend(original, overlay, 0.22 * strength)
        result = ImageEnhance.Color(result).enhance(1 + 0.20 * strength)
        result = ImageEnhance.Brightness(result).enhance(1 + 0.06 * strength)
    elif filter_name == "Cool Blue":
        overlay = Image.new("RGB", original.size, (55, 120, 210))
        result = Image.blend(original, overlay, 0.18 * strength)
        result = ImageEnhance.Contrast(result).enhance(1 + 0.12 * strength)
    elif filter_name == "Bright & Clean":
        result = ImageEnhance.Brightness(original).enhance(1 + 0.24 * strength)
        result = ImageEnhance.Contrast(result).enhance(1 + 0.10 * strength)
        result = ImageEnhance.Color(result).enhance(1 + 0.18 * strength)
    elif filter_name == "Matte":
        result = ImageEnhance.Contrast(original).enhance(1 - 0.25 * strength)
        result = ImageEnhance.Color(result).enhance(1 - 0.10 * strength)
        overlay = Image.new("RGB", original.size, (220, 205, 185))
        result = Image.blend(result, overlay, 0.08 * strength)
    elif filter_name == "Sharp":
        result = original.filter(ImageFilter.UnsharpMask(radius=2, percent=int(80 + 140 * strength), threshold=3))
    elif filter_name == "Pink Aesthetic":
        overlay = Image.new("RGB", original.size, (245, 145, 190))
        result = Image.blend(original, overlay, 0.18 * strength)
        result = ImageEnhance.Brightness(result).enhance(1 + 0.07 * strength)
    else:
        result = original

    return result


def suggest_filters(description):
    text = (description or "").lower()
    suggestions = []
    if any(word in text for word in ["sunset", "sunrise", "beach", "golden", "sky", "summer"]):
        suggestions += ["Warm Sunset", "Vintage", "Cinematic"]
    if any(word in text for word in ["person", "woman", "man", "girl", "boy", "portrait", "face", "selfie"]):
        suggestions += ["Dreamy Glow", "Bright & Clean", "Vintage"]
    if any(word in text for word in ["food", "cake", "coffee", "pizza", "fruit", "meal"]):
        suggestions += ["Bright & Clean", "Warm Sunset", "Sharp"]
    if any(word in text for word in ["building", "city", "street", "car", "night"]):
        suggestions += ["Cinematic", "Cool Blue", "Retro"]
    if any(word in text for word in ["dog", "cat", "bird", "animal", "pet"]):
        suggestions += ["Bright & Clean", "Dreamy Glow", "Vintage"]
    if any(word in text for word in ["forest", "tree", "flower", "garden", "nature", "mountain"]):
        suggestions += ["Vintage", "Matte", "Cinematic"]
    suggestions += ["Cinematic", "Vintage", "Bright & Clean"]
    # preserve order and remove duplicates
    return list(dict.fromkeys(suggestions))[:4]


def generate_social_content(description, platform, mood, language):
    desc = (description or "this moment").strip()
    mood_templates = {
        "Aesthetic": [
            f"Little moments, beautiful memories ✨ — {desc}",
            f"Finding beauty in the ordinary 🌿 — {desc}",
            f"A frame full of feelings 🤍 — {desc}"
        ],
        "Funny": [
            f"Camera roll said this one deserved a post 😂",
            f"Proof that I went outside 📸",
            f"Main character energy… on a budget 😎"
        ],
        "Travel": [
            f"Collecting moments, not things 🌍",
            f"New view, same wonder ✈️",
            f"Somewhere between lost and loving it 🧭"
        ],
        "Professional": [
            f"Sharing a moment from my journey: {desc}.",
            f"Small details. Meaningful progress. ✨",
            f"Creating, learning, and moving forward."
        ],
        "Romantic": [
            f"Some moments speak straight to the heart ❤️",
            f"A little magic in an ordinary moment 🌹",
            f"Memories that feel like a warm hug 🤍"
        ],
        "Motivational": [
            f"Keep showing up. Your story is still being written. 💫",
            f"Progress over perfection. One moment at a time. 🚀",
            f"Make today a memory worth keeping. ✨"
        ],
        "Attitude": [
            f"Own your vibe. No explanation needed 😎",
            f"Different energy, same goals 🔥",
            f"Not here to fit in. Here to be me."
        ]
    }
    choices = mood_templates.get(mood, mood_templates["Aesthetic"])
    caption = choices[0]
    hashtags = {
        "Instagram": "#VisionAIStudio #InstaDaily #PhotoOfTheDay #Aesthetic #Explore",
        "Facebook": "#Memories #LifeMoments #VisionAIStudio #PhotoShare",
        "Both": "#VisionAIStudio #PhotoOfTheDay #Memories #Aesthetic #Creative"
    }.get(platform, "#VisionAIStudio #PhotoOfTheDay")
    if language == "Marathi":
        caption = "सुंदर क्षण, सुंदर आठवणी ✨"
        hashtags = "#मराठी #आठवणी #VisionAIStudio #PhotoOfTheDay"
    elif language == "Hindi":
        caption = "छोटे-छोटे पल, खूबसूरत यादें ✨"
        hashtags = "#यादें #खूबसूरतपल #VisionAIStudio #PhotoOfTheDay"
    elif language == "English":
        pass
    return caption, hashtags, "\n".join(f"Option {i+1}: {x}" for i, x in enumerate(choices))


def prompt_to_recipe(prompt):
    """Translate simple natural-language edit requests into a filter recipe."""
    text = (prompt or "").lower()
    if not text.strip():
        return "Vintage", 70, "No prompt provided. Applying a vintage look as a sample."
    rules = [
        (["vintage", "old film", "film camera", "retro"], "Vintage", 75),
        (["cinematic", "movie", "dramatic"], "Cinematic", 75),
        (["blur", "soft focus"], "Soft Blur", 55),
        (["dreamy", "glow", "soft glow"], "Dreamy Glow", 65),
        (["black and white", "black & white", "monochrome", "grayscale"], "Black & White", 85),
        (["sunset", "golden hour", "warm", "golden"], "Warm Sunset", 70),
        (["blue", "cool tone", "icy"], "Cool Blue", 65),
        (["bright", "clean", "natural"], "Bright & Clean", 65),
        (["matte", "muted"], "Matte", 70),
        (["sharp", "clear"], "Sharp", 75),
        (["pink", "pink aesthetic"], "Pink Aesthetic", 65),
    ]
    for keywords, filter_name, strength in rules:
        if any(word in text for word in keywords):
            return filter_name, strength, f"Prompt matched the **{filter_name}** filter recipe."
    return "Cinematic", 65, "No exact recipe matched; try words like vintage, blur, warm, pink, cinematic, or black and white."


def add_text_to_image(image, text, position):
    image = prepare_image(image).copy()
    if not text or not text.strip():
        return image
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", max(22, image.width // 18))
    except OSError:
        font = ImageFont.load_default()
    padding = max(12, image.width // 45)
    bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=5, stroke_width=1)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = max(padding, (image.width - tw) // 2)
    if position == "Top":
        y = padding
    elif position == "Bottom":
        y = image.height - th - padding
    else:
        y = (image.height - th) // 2
    # translucent-ish dark backing for readable text
    draw.rectangle((x - 10, y - 7, x + tw + 10, y + th + 7), fill=(0, 0, 0))
    draw.multiline_text((x, y), text, font=font, fill="white", spacing=5,
                        stroke_width=1, stroke_fill="black")
    return image


def make_social_canvas(image, format_name):
    image = prepare_image(image)
    dimensions = {
        "Square Post (1:1)": (1080, 1080),
        "Portrait Post (4:5)": (1080, 1350),
        "Story / Reel Cover (9:16)": (1080, 1920),
        "Facebook Landscape (1.91:1)": (1200, 628)
    }
    size = dimensions[format_name]
    fitted = ImageOps.fit(image, size, method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))
    return fitted


def save_image(image, prefix="visionai"):
    if image is None:
        raise gr.Error("There is no edited image to download.")
    path = os.path.join(tempfile.gettempdir(), f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png")
    image.save(path, format="PNG")
    return path


def run_studio(image, filter_name, intensity, prompt, overlay_text, text_position, format_name):
    if image is None:
        raise gr.Error("Upload an image first.")
    chosen_filter = filter_name
    prompt_note = ""
    if prompt and prompt.strip():
        chosen_filter, prompt_strength, prompt_note = prompt_to_recipe(prompt)
        # prompt recipe strength is used if user hasn't changed default intensity
        if intensity is None:
            intensity = prompt_strength
    edited = apply_filter(image, chosen_filter, intensity)
    if overlay_text and overlay_text.strip():
        edited = add_text_to_image(edited, overlay_text, text_position)
    final = make_social_canvas(edited, format_name)
    description = analyze_image(image)
    recommendations = suggest_filters(description)
    info = (
        f"AI image description: {description}\n\n"
        f"Applied filter: {chosen_filter}\n"
        f"Filter strength: {intensity}%\n"
        f"Recommended filters: {', '.join(recommendations)}\n"
        f"Export format: {format_name}\n"
        f"{prompt_note}"
    )
    output_path = save_image(final)
    return final, output_path, info, description, ", ".join(recommendations), chosen_filter


def recommend_only(image):
    if image is None:
        raise gr.Error("Upload an image first.")
    description = analyze_image(image)
    recommendations = suggest_filters(description)
    return description, ", ".join(recommendations)


# -------------------- UI --------------------
CSS = """
.gradio-container {max-width: 1250px !important; margin: auto !important; font-family: 'Segoe UI', sans-serif !important;}
#hero {padding: 28px 20px; text-align: center; border-radius: 22px; color: white; background: linear-gradient(120deg,#101b3d,#344bb0,#7955e8); margin-bottom: 16px;}
#hero h1 {font-size: 38px !important; color: white !important; margin-bottom: 6px;}
#hero p {color: #e5eaff !important;}
.panel {border: 1px solid #dfe5f1 !important; border-radius: 16px !important; padding: 16px !important;}
.primary-btn {background: linear-gradient(90deg,#465de8,#7955e8) !important; color: white !important;}
"""

with gr.Blocks(title="VisionAI Studio") as demo:
    gr.HTML("""
    <div id="hero">
      <h1>✦ VisionAI Studio</h1>
      <p>AI-powered photo filters, smart recommendations, captions and social media designs</p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1, elem_classes="panel"):
            gr.Markdown("## 1. Upload your photo")
            input_image = gr.Image(type="pil", label="Choose an image", height=300)
            analyze_btn = gr.Button("🔎 Analyze & Recommend Filters")
            description_out = gr.Textbox(label="AI image description", lines=2, interactive=False)
            recommendations_out = gr.Textbox(label="Recommended filters", lines=2, interactive=False)

            gr.Markdown("## 2. Edit your photo")
            filter_choice = gr.Dropdown(
                choices=["Original", "Vintage", "Cinematic", "Retro", "Black & White",
                         "Dreamy Glow", "Soft Blur", "Warm Sunset", "Cool Blue",
                         "Bright & Clean", "Matte", "Sharp", "Pink Aesthetic"],
                value="Vintage", label="Choose a filter"
            )
            intensity = gr.Slider(minimum=0, maximum=100, value=70, step=5, label="Filter strength (%)")
            prompt = gr.Textbox(
                label="Describe the look you want",
                placeholder="Example: make it warm and vintage, soft pink aesthetic, cinematic...",
                lines=2
            )
            gr.Markdown("*Prompt-based editing maps your words to available filter recipes. Full generative background replacement is not included in this local version.*")

            gr.Markdown("## 3. Social post design")
            format_choice = gr.Dropdown(
                choices=["Square Post (1:1)", "Portrait Post (4:5)", "Story / Reel Cover (9:16)", "Facebook Landscape (1.91:1)"],
                value="Story / Reel Cover (9:16)", label="Export layout"
            )
            overlay_text = gr.Textbox(label="Optional text on image", placeholder="Your quote or story title")
            text_position = gr.Radio(choices=["Top", "Center", "Bottom"], value="Bottom", label="Text position")
            edit_btn = gr.Button("✨ Apply Edit & Create Design", variant="primary", elem_classes="primary-btn")

        with gr.Column(scale=1, elem_classes="panel"):
            gr.Markdown("## Preview & download")
            edited_image = gr.Image(label="Edited preview", type="pil", height=420)
            download_image = gr.File(label="Download edited PNG")
            edit_info = gr.Textbox(label="Edit summary", lines=7, interactive=False)

    analyze_btn.click(
        fn=recommend_only,
        inputs=[input_image],
        outputs=[description_out, recommendations_out]
    )
    edit_btn.click(
        fn=run_studio,
        inputs=[input_image, filter_choice, intensity, prompt, overlay_text, text_position, format_choice],
        outputs=[edited_image, download_image, edit_info, description_out, recommendations_out, filter_choice]
    )

    gr.Markdown("---")
    gr.Markdown("## 4. Social media caption assistant")
    with gr.Row():
        with gr.Column(elem_classes="panel"):
            platform = gr.Dropdown(choices=["Instagram", "Facebook", "Both"], value="Instagram", label="Platform")
            mood = gr.Dropdown(choices=["Aesthetic", "Funny", "Travel", "Professional", "Romantic", "Motivational", "Attitude"], value="Aesthetic", label="Caption mood")
            caption_language = gr.Dropdown(choices=["English", "Hindi", "Marathi"], value="English", label="Caption language")
            caption_btn = gr.Button("💬 Generate Captions & Hashtags")
        with gr.Column(elem_classes="panel"):
            caption_out = gr.Textbox(label="Suggested caption", lines=3)
            hashtags_out = gr.Textbox(label="Hashtags", lines=2)
            options_out = gr.Textbox(label="More caption ideas", lines=5)

    caption_btn.click(
        fn=generate_social_content,
        inputs=[description_out, platform, mood, caption_language],
        outputs=[caption_out, hashtags_out, options_out]
    )

    gr.Markdown("""
    ---
    **How it works:** BLIP describes the uploaded image. VisionAI uses that description to recommend filters. Pillow applies the visual effects and creates the export layout. Caption suggestions use selectable styles and templates.

    **Limitations:** The local version uses prompt-to-filter recipes rather than a full diffusion-based image-generation model. Caption suggestions are template-based. Translation, social platform posting, and live trend detection are not included. Review edited images before sharing.
    """)

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="blue", neutral_hue="slate"), css=CSS)

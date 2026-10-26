import io
import streamlit as st
import torch
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageDraw, ImageFont
from transformers import BlipProcessor, BlipForConditionalGeneration

MODEL_NAME = "Salesforce/blip-image-captioning-base"
MAX_IMAGE_SIZE = 1600

FILTERS = [
    "Original", "Vintage", "Cinematic", "Retro", "Black & White",
    "Dreamy Glow", "Soft Blur", "Warm Sunset", "Cool Blue",
    "Bright & Clean", "Matte", "Sharp", "Pink Aesthetic"
]
FORMATS = [
    "Square Post (1:1)", "Portrait Post (4:5)",
    "Story / Reel Cover (9:16)", "Facebook Landscape (1.91:1)"
]

st.set_page_config(page_title="VisionAI Studio", page_icon="✨", layout="wide")

@st.cache_resource(show_spinner="Loading VisionAI image understanding model (first launch may take a few minutes)...")
def load_model():
    processor = BlipProcessor.from_pretrained(MODEL_NAME)
    model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME)
    model.eval()
    return processor, model

def prepare_image(image):
    if image is None:
        raise ValueError("Please upload a photo first.")
    image = image.convert("RGB")
    image.thumbnail((MAX_IMAGE_SIZE, MAX_IMAGE_SIZE))
    return image

def analyze_image(image):
    image = prepare_image(image)
    processor, model = load_model()
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=35, num_beams=3)
    caption = processor.decode(output[0], skip_special_tokens=True).strip()
    return caption[:1].upper() + caption[1:] if caption else "A photo ready for creative editing."

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
        result = Image.blend(result, Image.new("RGB", result.size, (15, 36, 48)), 0.12 * strength)
    elif filter_name == "Retro":
        result = ImageEnhance.Color(original).enhance(0.70)
        result = ImageEnhance.Contrast(result).enhance(1 + 0.25 * strength)
        result = Image.blend(result, Image.new("RGB", result.size, (205, 120, 65)), 0.14 * strength)
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
        result = Image.blend(original, Image.new("RGB", original.size, (255, 145, 65)), 0.22 * strength)
        result = ImageEnhance.Color(result).enhance(1 + 0.20 * strength)
        result = ImageEnhance.Brightness(result).enhance(1 + 0.06 * strength)
    elif filter_name == "Cool Blue":
        result = Image.blend(original, Image.new("RGB", original.size, (55, 120, 210)), 0.18 * strength)
        result = ImageEnhance.Contrast(result).enhance(1 + 0.12 * strength)
    elif filter_name == "Bright & Clean":
        result = ImageEnhance.Brightness(original).enhance(1 + 0.24 * strength)
        result = ImageEnhance.Contrast(result).enhance(1 + 0.10 * strength)
        result = ImageEnhance.Color(result).enhance(1 + 0.18 * strength)
    elif filter_name == "Matte":
        result = ImageEnhance.Contrast(original).enhance(1 - 0.25 * strength)
        result = ImageEnhance.Color(result).enhance(1 - 0.10 * strength)
        result = Image.blend(result, Image.new("RGB", original.size, (220, 205, 185)), 0.08 * strength)
    elif filter_name == "Sharp":
        result = original.filter(ImageFilter.UnsharpMask(radius=2, percent=int(80 + 140 * strength), threshold=3))
    elif filter_name == "Pink Aesthetic":
        result = Image.blend(original, Image.new("RGB", original.size, (245, 145, 190)), 0.18 * strength)
        result = ImageEnhance.Brightness(result).enhance(1 + 0.07 * strength)
    else:
        result = original
    return result

def suggest_filters(description):
    text = (description or "").lower()
    suggestions = []
    if any(w in text for w in ["sunset", "sunrise", "beach", "golden", "sky", "summer"]):
        suggestions += ["Warm Sunset", "Vintage", "Cinematic"]
    if any(w in text for w in ["person", "woman", "man", "girl", "boy", "portrait", "face", "selfie"]):
        suggestions += ["Dreamy Glow", "Bright & Clean", "Vintage"]
    if any(w in text for w in ["food", "cake", "coffee", "pizza", "fruit", "meal"]):
        suggestions += ["Bright & Clean", "Warm Sunset", "Sharp"]
    if any(w in text for w in ["building", "city", "street", "car", "night"]):
        suggestions += ["Cinematic", "Cool Blue", "Retro"]
    if any(w in text for w in ["dog", "cat", "bird", "animal", "pet"]):
        suggestions += ["Bright & Clean", "Dreamy Glow", "Vintage"]
    if any(w in text for w in ["forest", "tree", "flower", "garden", "nature", "mountain"]):
        suggestions += ["Vintage", "Matte", "Cinematic"]
    suggestions += ["Cinematic", "Vintage", "Bright & Clean"]
    return list(dict.fromkeys(suggestions))[:4]

def prompt_to_recipe(prompt):
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
    for keywords, name, strength in rules:
        if any(word in text for word in keywords):
            return name, strength, f"Prompt matched the {name} filter recipe."
    return "Cinematic", 65, "No exact recipe matched; try vintage, blur, warm, pink, cinematic, or black and white."

def add_text_to_image(image, text, position):
    image = prepare_image(image).copy()
    if not text or not text.strip():
        return image
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", max(22, image.width // 18))
    except OSError:
        font = ImageFont.load_default()
    padding = max(12, image.width // 45)
    bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=5, stroke_width=1)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = max(padding, (image.width - tw) // 2)
    y = padding if position == "Top" else image.height - th - padding if position == "Bottom" else (image.height - th) // 2
    draw.rectangle((x - 10, y - 7, x + tw + 10, y + th + 7), fill=(0, 0, 0))
    draw.multiline_text((x, y), text, font=font, fill="white", spacing=5, stroke_width=1, stroke_fill="black")
    return image

def make_social_canvas(image, format_name):
    dimensions = {
        "Square Post (1:1)": (1080, 1080),
        "Portrait Post (4:5)": (1080, 1350),
        "Story / Reel Cover (9:16)": (1080, 1920),
        "Facebook Landscape (1.91:1)": (1200, 628),
    }
    return ImageOps.fit(prepare_image(image), dimensions[format_name], method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))

def generate_social_content(description, platform, mood, language):
    desc = (description or "this moment").strip()
    mood_templates = {
        "Aesthetic": [f"Little moments, beautiful memories ✨ — {desc}", f"Finding beauty in the ordinary 🌿 — {desc}", f"A frame full of feelings 🤍 — {desc}"],
        "Funny": ["Camera roll said this one deserved a post 😂", "Proof that I went outside 📸", "Main character energy… on a budget 😎"],
        "Travel": ["Collecting moments, not things 🌍", "New view, same wonder ✈️", "Somewhere between lost and loving it 🧭"],
        "Professional": [f"Sharing a moment from my journey: {desc}.", "Small details. Meaningful progress. ✨", "Creating, learning, and moving forward."],
        "Romantic": ["Some moments speak straight to the heart ❤️", "A little magic in an ordinary moment 🌹", "Memories that feel like a warm hug 🤍"],
        "Motivational": ["Keep showing up. Your story is still being written. 💫", "Progress over perfection. One moment at a time. 🚀", "Make today a memory worth keeping. ✨"],
        "Attitude": ["Own your vibe. No explanation needed 😎", "Different energy, same goals 🔥", "Not here to fit in. Here to be me."],
    }
    choices = mood_templates.get(mood, mood_templates["Aesthetic"])
    caption = choices[0]
    hashtags = {
        "Instagram": "#VisionAIStudio #InstaDaily #PhotoOfTheDay #Aesthetic #Explore",
        "Facebook": "#Memories #LifeMoments #VisionAIStudio #PhotoShare",
        "Both": "#VisionAIStudio #PhotoOfTheDay #Memories #Aesthetic #Creative",
    }.get(platform, "#VisionAIStudio #PhotoOfTheDay")
    if language == "Marathi":
        caption, hashtags = "सुंदर क्षण, सुंदर आठवणी ✨", "#मराठी #आठवणी #VisionAIStudio #PhotoOfTheDay"
    elif language == "Hindi":
        caption, hashtags = "छोटे-छोटे पल, खूबसूरत यादें ✨", "#यादें #खूबसूरतपल #VisionAIStudio #PhotoOfTheDay"
    return caption, hashtags, "\n".join(f"Option {i+1}: {item}" for i, item in enumerate(choices))

st.markdown("""
<style>
.hero {padding: 1.5rem; text-align:center; border-radius:18px; color:white;
background: linear-gradient(120deg,#101b3d,#344bb0,#7955e8); margin-bottom:1rem;}
.hero h1 {color:white;}
</style>
<div class="hero"><h1>✨ VisionAI Studio</h1>
<p>AI-powered photo filters, smart recommendations, captions and social media designs</p></div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("VisionAI Studio")
    st.write("Upload a photo, choose a filter, and create a social-ready image.")
    uploaded = st.file_uploader("Choose a photo", type=["png", "jpg", "jpeg", "webp"])
    filter_choice = st.selectbox("Choose a filter", FILTERS, index=1)
    intensity = st.slider("Filter strength (%)", 0, 100, 70, 5)
    prompt = st.text_area("Describe the look you want", placeholder="e.g. warm vintage, soft pink, cinematic")
    format_choice = st.selectbox("Export layout", FORMATS, index=2)
    overlay_text = st.text_input("Optional text on image")
    text_position = st.selectbox("Text position", ["Top", "Center", "Bottom"], index=2)
    edit_button = st.button("✨ Apply Edit & Create Design", type="primary", use_container_width=True)

left, right = st.columns(2)
image = None
if uploaded:
    try:
        image = Image.open(uploaded).convert("RGB")
        image = prepare_image(image)
        left.subheader("Original photo")
        left.image(image, use_container_width=True)
    except Exception as exc:
        st.error(f"Could not open that image: {exc}")

if "description" not in st.session_state:
    st.session_state.description = ""
if "edited_image" not in st.session_state:
    st.session_state.edited_image = None
if "edit_info" not in st.session_state:
    st.session_state.edit_info = ""

if image is not None:
    if st.button("🔎 Analyze & Recommend Filters"):
        try:
            with st.spinner("Analyzing your photo..."):
                st.session_state.description = analyze_image(image)
                st.session_state.recommendations = suggest_filters(st.session_state.description)
        except Exception as exc:
            st.error(f"Image analysis failed: {exc}")
    if st.session_state.description:
        st.subheader("AI image description")
        st.write(st.session_state.description)
        st.write("**Recommended filters:** " + ", ".join(st.session_state.get("recommendations", [])))

    if edit_button:
        try:
            chosen_filter = filter_choice
            prompt_note = ""
            chosen_intensity = intensity
            if prompt.strip():
                chosen_filter, prompt_strength, prompt_note = prompt_to_recipe(prompt)
                if intensity == 70:
                    chosen_intensity = prompt_strength
            with st.spinner("Editing your photo..."):
                edited = apply_filter(image, chosen_filter, chosen_intensity)
                if overlay_text.strip():
                    edited = add_text_to_image(edited, overlay_text, text_position)
                final_image = make_social_canvas(edited, format_choice)
                description = analyze_image(image)
                recommendations = suggest_filters(description)
            st.session_state.edited_image = final_image
            st.session_state.description = description
            st.session_state.recommendations = recommendations
            st.session_state.edit_info = (
                f"AI image description: {description}\n\nApplied filter: {chosen_filter}\n"
                f"Filter strength: {chosen_intensity}%\nRecommended filters: {', '.join(recommendations)}\n"
                f"Export format: {format_choice}\n{prompt_note}"
            )
        except Exception as exc:
            st.error(f"Editing failed: {exc}")

if st.session_state.edited_image is not None:
    right.subheader("Edited preview")
    right.image(st.session_state.edited_image, use_container_width=True)
    right.text(st.session_state.edit_info)
    buffer = io.BytesIO()
    st.session_state.edited_image.save(buffer, format="PNG")
    st.download_button("⬇️ Download edited PNG", data=buffer.getvalue(), file_name="visionai_edited.png", mime="image/png")

st.divider()
st.header("💬 Social media caption assistant")
c1, c2 = st.columns(2)
with c1:
    platform = st.selectbox("Platform", ["Instagram", "Facebook", "Both"])
    mood = st.selectbox("Caption mood", ["Aesthetic", "Funny", "Travel", "Professional", "Romantic", "Motivational", "Attitude"])
    language = st.selectbox("Caption language", ["English", "Hindi", "Marathi"])
    caption_description = st.text_input("Describe the photo or moment", value=st.session_state.get("description", ""))
    caption_button = st.button("Generate Captions & Hashtags")
with c2:
    if caption_button:
        caption, hashtags, options = generate_social_content(caption_description, platform, mood, language)
        st.text_area("Suggested caption", caption, height=90)
        st.text_area("Hashtags", hashtags, height=70)
        st.text_area("More caption ideas", options, height=140)

st.caption("How it works: BLIP describes uploaded images; Pillow applies image filters and export layouts. Caption suggestions use templates. Prompt-based editing selects an existing filter recipe; it does not generate new pixels or replace backgrounds.")

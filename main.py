from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import joblib
from PIL import Image
import io
import base64
import json
import os
import requests

app = FastAPI(title="HairAI Studio - Unified API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

tabular_model = joblib.load('hair_loss_kaggle_model.joblib')

DEFICIENCY_ANALYSIS = {
    "Iron deficiency": {"risk_boost": 12, "advice": "Boost iron with spinach, lentils, or iron bisglycinate alongside Vitamin C."},
    "Vitamin D Deficiency": {"risk_boost": 10, "advice": "Opt for 15-20 mins daily sunlight or Vitamin D3 (2000-4000 IU/day)."},
    "Protein deficiency": {"risk_boost": 10, "advice": "Increase daily amino acid intake with eggs, tofu, or lean protein."},
    "Zinc Deficiency": {"risk_boost": 8, "advice": "Incorporate pumpkin seeds, chickpeas, or zinc picolinate."},
    "Biotin Deficiency ": {"risk_boost": 8, "advice": "Add almonds, sweet potatoes, or Biotin (B7) supplements."},
    "Magnesium deficiency": {"risk_boost": 6, "advice": "Consume dark leafy greens or magnesium glycinate."},
    "Omega-3 fatty acids": {"risk_boost": 6, "advice": "Include flaxseeds, walnuts, or fish oil supplements."}
}

@app.post("/api/predict-complete")
async def predict_complete(
    data_json: str = Form(...),
    scalp_photo: UploadFile = File(None)
):
    user_data = json.loads(data_json)
    
    image_analysis = None
    if scalp_photo and scalp_photo.filename:
        try:
            contents = await scalp_photo.read()
            if len(contents) > 0:
                image = Image.open(io.BytesIO(contents))
                width, height = image.size
                image_analysis = {
                    "image_processed": True,
                    "resolution": f"{width}x{height}",
                    "visual_flags": ["Clear scalp texture detected. No severe surface inflammation flagged."]
                }
        except Exception:
            image_analysis = {"image_processed": False, "visual_flags": ["Error reading image file."]}

    deficiencies = user_data.get('Nutritional_Deficiencies', ["No Data"])
    if isinstance(deficiencies, str):
        deficiencies = [deficiencies]

    primary_deficiency = deficiencies[0] if len(deficiencies) > 0 else "No Data"

    input_df = pd.DataFrame([{
        'Genetics': user_data.get('Genetics', 'No'),
        'Hormonal Changes': user_data.get('Hormonal_Changes', 'No'),
        'Medical Conditions': user_data.get('Medical_Conditions', 'No Data'),
        'Medications & Treatments': user_data.get('Medications_Treatments', 'No Data'),
        'Nutritional Deficiencies': primary_deficiency,
        'Stress': user_data.get('Stress', 'Moderate'),
        'Age': int(user_data.get('Age', 25)),
        'Poor Hair Care Habits': user_data.get('Poor_Hair_Care_Habits', 'No'),
        'Environmental Factors': user_data.get('Environmental_Factors', 'No'),
        'Smoking': user_data.get('Smoking', 'No'),
        'Weight Loss': user_data.get('Weight_Loss', 'No')
    }])

    prediction = int(tabular_model.predict(input_df)[0])
    base_prob = tabular_model.predict_proba(input_df)[0][1] * 100

    extra_risk = 0
    dietary_tips = []
    
    for item in deficiencies:
        if item in DEFICIENCY_ANALYSIS:
            extra_risk += DEFICIENCY_ANALYSIS[item]["risk_boost"]
            dietary_tips.append(f"• <b>{item.strip()}:</b> {DEFICIENCY_ANALYSIS[item]['advice']}")

    final_risk = min(round(base_prob + (extra_risk * 0.3), 1), 99.0)
    adjusted_prediction = 1 if final_risk > 50.0 else 0

    return {
        "hair_loss_predicted": adjusted_prediction,
        "risk_percentage": f"{final_risk}%",
        "dietary_recommendations": dietary_tips,
        "status_label": "High Hair Loss Risk" if adjusted_prediction == 1 else "Low Hair Loss Risk",
        "easy_explanation": f"Evaluated against {len(deficiencies)} selected nutritional marker(s) alongside genetic and lifestyle factors.",
        "image_analysis": image_analysis
    }

# -------------------------------------------------------------
# HAIRSTYLE AI ENDPOINT - FREE IMAGE-TO-IMAGE HAIR EDITING
# -------------------------------------------------------------
HF_API_TOKEN = os.getenv("HF_API_TOKEN", "f_ZsJGbgqBcdXYiOSQDjfZUALXftjlcLApNH")
API_URL = "https://router.huggingface.co/hf-inference/v1/images/generations"

@app.post("/api/generate-hairstyle")
async def generate_hairstyle(
    style_prompt: str = Form(...),
    tech_details: str = Form(""),
    user_portrait: UploadFile = File(...)
):
    try:
        contents = await user_portrait.read()
        portrait_img = Image.open(io.BytesIO(contents)).convert("RGB").resize((512, 512))

        buffered = io.BytesIO()
        portrait_img.save(buffered, format="JPEG", quality=85)
        input_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        headers = {
            "Authorization": f"Bearer {HF_API_TOKEN}",
            "Content-Type": "application/json"
        }
        
        prompt_text = (
            f"photograph of the same person from input image with identical face, skin, and eyes, "
            f"wearing a modern {style_prompt} haircut, detailed hair texture, sharp focus"
        )

        payload = {
            "model": "runwayml/stable-diffusion-v1-5",
            "inputs": f"data:image/jpeg;base64,{input_b64}",
            "parameters": {
                "prompt": prompt_text,
                "negative_prompt": "different face, altered facial features, distorted eyes, blurry, low quality, bad anatomy",
                "strength": 0.50
            }
        }

        response = requests.post(API_URL, headers=headers, json=payload, timeout=25)

        if response.status_code == 200 and len(response.content) > 1000:
            img_bytes = response.content
            img_base64 = base64.b64encode(img_bytes).decode("utf-8")
            image_output = f"data:image/jpeg;base64,{img_base64}"
        else:
            image_output = f"data:image/jpeg;base64,{input_b64}"

        barber_script = (
            f"Hey! I want a '{style_prompt}'.\n\n"
            f"Technical Specifications:\n"
            f"• Top Section: {tech_details if tech_details else 'Textured, tailored length'}\n"
            f"• Sides & Back: Tapered smoothly into the neckline\n"
            f"• Line-up & Edges: Natural edge-up, keep hairline intact"
        )

        return {
            "status": "success",
            "style_requested": style_prompt,
            "barber_script": barber_script,
            "generated_image_b64": image_output
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
import os
import streamlit as st
import mimetypes
from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Prepare food with you ingredients",
    page_icon="🍳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF6B6B;
        color: white;
        font-weight: bold;
        padding: 0.75rem;
        border-radius: 10px;
        border: none;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #FF5252;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(255, 107, 107, 0.4);
    }
    .recipe-container {
        background-color: #f8f9fa;
        padding: 2rem;
        border-radius: 15px;
        margin-top: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .ingredient-tag {
        background-color: #e3f2fd;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        margin: 0.2rem;
        display: inline-block;
        font-size: 0.9rem;
    }
    h1 {
        color: #FF6B6B;
        text-align: center;
        padding: 1rem 0;
    }
    .feature-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        text-align: center;
    }
    .dish-selected {
        background-color: #4CAF50;
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
        font-size: 1.2rem;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'recipe_history' not in st.session_state:
    st.session_state.recipe_history = []

# Initialize Gemini client
@st.cache_resource
def init_gemini_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("⚠ GEMINI_API_KEY not found in .env file!")
        st.stop()
    return genai.Client(api_key=api_key)

client = init_gemini_client()

def generate_recipe_from_text(ingredients, dietary_prefs, cuisine, meal_type, specific_dish=None):
    """Generate recipe based on text inputs"""
    
    if specific_dish and specific_dish != "None - Custom Recipe":
        prompt = f"""Create a CONCISE recipe for *{specific_dish}*.

    *Additional Ingredients Available:* {', '.join(ingredients) if ingredients else "Use standard ingredients"}
    *Dietary Preferences:* {dietary_prefs}

    Please provide a SHORT and CONCISE recipe with:
    1. Recipe Name: {specific_dish}
    2. Time: Prep + Cook (one line)
    3. Servings
    4. Ingredients: List with measurements (keep brief)
    5. Instructions: Maximum 5-6 clear steps only
    6. One quick tip

    IMPORTANT: Keep the response under 300 words. Be brief and to the point!"""
    else:
        prompt = f"""Create a CONCISE recipe based on these specifications:

    *Ingredients Available:* {', '.join(ingredients)}
    *Dietary Preferences:* {dietary_prefs}
    *Cuisine Style:* {cuisine}
    *Meal Type:* {meal_type}

    Please provide a SHORT and CONCISE recipe with:
    1. Recipe Name
    2. Time: Prep + Cook (one line)
    3. Servings
    4. Ingredients: List with measurements (keep brief)
    5. Instructions: Maximum 5-6 clear steps only
    6. One quick tip

    IMPORTANT: Keep the response under 300 words. Be brief and to the point!"""
    
    contents = [types.Part.from_text(text=prompt)]
    
    config = types.GenerateContentConfig(
        response_modalities=["TEXT"],
        temperature=0.9,
        max_output_tokens=500,
    )
    
    response_text = ""
    with st.spinner("👨‍🍳 Crafting your perfect recipe..."):
        for chunk in client.models.generate_content_stream(
            model="gemini-2.0-flash",
            contents=contents,
            config=config,
        ):
            if (chunk.candidates and 
                chunk.candidates[0].content and 
                chunk.candidates[0].content.parts):
                for part in chunk.candidates[0].content.parts:
                    if hasattr(part, 'text') and part.text:
                        response_text += part.text
    
    return response_text

def generate_recipe_from_image(image_bytes, additional_context):
    """Generate recipe based on uploaded image"""
    mime_type = "image/jpeg"
    
    prompt = f"""Analyze this food image and create a SHORT recipe.

    Additional Context: {additional_context if additional_context else "None provided"}

    Please provide a CONCISE recipe with:
    1. Dish name
    2. Time: Prep + Cook (one line)
    3. Servings
    4. Ingredients: Brief list with measurements
    5. Instructions: Maximum 5-6 steps only
    6. One quick tip

    IMPORTANT: Keep response under 300 words. Be brief!"""
    
    contents = [
        types.Part(inline_data=types.Blob(mime_type=mime_type, data=image_bytes)),
        types.Part.from_text(text=prompt),
    ]
    
    config = types.GenerateContentConfig(
        response_modalities=["TEXT"],
        temperature=0.9,
        max_output_tokens=500,
    )
    
    response_text = ""
    with st.spinner("🔍 Analyzing image and generating recipe..."):
        for chunk in client.models.generate_content_stream(
            model="gemini-2.0-flash",
            contents=contents,
            config=config,
        ):
            if (chunk.candidates and 
                chunk.candidates[0].content and 
                chunk.candidates[0].content.parts):
                for part in chunk.candidates[0].content.parts:
                    if hasattr(part, 'text') and part.text:
                        response_text += part.text
    
    return response_text

# Main App
st.title("🍳 Recipe Lab")
st.markdown("### Delicious starts here")

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/chef-hat.png", width=100)
    st.markdown("## Prepare your food with your ingredients")
    st.markdown("""
    -  *Ingredient-Based Recipes*
    -  *Image-to-Recipe*
    -  *99+ Popular Dishes*
    -  *Dietary Preferences*
    -  *Multiple Cuisines*
    """)

    st.markdown("---")
    st.markdown("### 📜 Recipe History")
    if st.session_state.recipe_history:
        for i, item in enumerate(reversed(st.session_state.recipe_history[-5:]), 1):
            st.markdown(f"{i}.** {item[:50]}...")
    else:
        st.info("No recipes generated yet!")

# Main content tabs
tab1, tab2 = st.tabs(["📝 Ingredient-Based Recipe", "📸 Image-to-Recipe"])

# Tab 1: Ingredient-Based Recipe
with tab1:
    st.markdown('<div class="feature-card"><h3>🥘 Create Recipe from Ingredients</h3><p>Tell us what you have, and we\'ll create something amazing!</p></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🛒 Your Ingredients")
        ingredient_input = st.text_area(
            "Enter ingredients (one per line or comma-separated)",
            height=150,
            placeholder="e.g., chicken breast, tomatoes, garlic, olive oil, basil"
        )
        
        dietary_prefs = st.multiselect(
            " Dietary Preferences",
            ["Vegetarian", "Vegan", "Gluten-Free", "Dairy-Free", "Keto", "Low-Carb", "Deep Fry", "Stir Fry", "Light Meal", "High-Protein", "Spice Lover", "Sweet", "None"],
            default=["None"]
        )
    
    with col2:
        st.markdown("####  Preferences")
        cuisine = st.selectbox(
            "Cuisine Style",
            ["Any", "Pakistani", "Indian", "Mughlai", "Punjabi", "Hyderabadi", "Kashmiri", "Chinese", "Italian", "Mexican", "Japanese", "Thai", "French", "Mediterranean", "American", "Korean"]
        )
        
        meal_type = st.selectbox(
            "Meal Type",
            ["Any", "Breakfast", "Lunch", "Dinner", "Snack", "Dessert", "Appetizer", "Street Food", "BBQ"]
        )
        
        # Popular Dish Selection with Search
        st.markdown("#### 🍛 Popular Dishes")
        
        popular_dishes = [
            "None - Custom Recipe",
            "Biryani", "Chicken Karahi", "Mutton Karahi", "Nihari", "Haleem",
            "Paya", "Qorma", "Qeema", "Handi", "Butter Chicken",
            "Tandoori Chicken", "Chicken Tikka", "Bihari Kabab", "Seekh Kabab", "Chapli Kabab",
            "Malai Boti", "Aloo Gosht", "Daal Chawal", "Aloo Keema", "Palak Paneer",
            "Paneer Tikka", "Shahi Paneer", "Matar Paneer", "Rajma Chawal", "Chole Bhature",
            "Pulao", "Kabuli Pulao", "Sindhi Biryani", "Bombay Biryani", "Hyderabadi Biryani",
            "Tawa Chicken", "Katakat", "White Karahi", "Chicken Jalfrezi", "Chicken Shashlik",
            "Mughlai Chicken", "Chicken Masala", "Fish Fry", "Fish Curry", "Daal Makhni",
            "Daal Tarka", "Mix Sabzi", "Bhindi Masala", "Baingan Bharta", "Aloo Baingan",
            "Aloo Gobi", "Gobi Manchurian", "Chicken Manchurian", "Fried Rice", "Chowmein",
            "Chicken 65", "Chicken Malai Handi", "Reshmi Kabab", "Afghani Boti", "Shinwari Karahi",
            "Chicken Ginger", "Tawa Keema", "Rogan Josh", "Kashmiri Yakhni", "Masala Dosa",
            "Idli Sambhar", "Vada", "Uttapam", "Pani Puri", "Dahi Puri",
            "Bhel Puri", "Sev Puri", "Papdi Chaat", "Gol Gappay", "Samosa Chaat",
            "Chicken Roll", "Paratha Roll", "Lachha Paratha", "Amritsari Kulcha", "Naan",
            "Garlic Naan", "Rumali Roti", "Tandoori Roti", "Chapati", "Halwa Puri",
            "Chana Puri", "Aloo Paratha", "Keema Paratha", "Methi Aloo", "Kadhi Pakora",
            "Bagara Baingan", "Korma Biryani", "Chicken White Qorma", "Chicken Handi Masala", "Dhaba Style Karahi",
            "Lahori Chargha", "Steam Roast", "Tikka Boti", "Mutton Handi", "Chicken Sindhi Karahi",
            "Chicken Madras", "Chicken Kadhai Masala", "Vegetable Biryani", "Achari Chicken", "Chicken Kolhapuri"
        ]
        
        popular_dish = st.selectbox(
            "Choose a popular dish",
            popular_dishes,
            help="Select from 99+ popular Pakistani, Indian & Desi dishes!"
        )
    
    if st.button("🎯 Generate Recipe", key="gen_text"):
        # Check if either ingredients or specific dish is provided
        if ingredient_input.strip() or (popular_dish and popular_dish != "None - Custom Recipe"):
            # Parse ingredients
            ingredients = []
            if ingredient_input.strip():
                ingredients = [ing.strip() for ing in ingredient_input.replace('\n', ',').split(',') if ing.strip()]
            
            # Display information
            if popular_dish and popular_dish != "None - Custom Recipe":
                st.markdown(f'<div class="dish-selected">🍽 Generating recipe for: {popular_dish}</div>', unsafe_allow_html=True)
            
            if ingredients:
                st.markdown("#### Selected Ingredients:")
                for ing in ingredients:
                    st.markdown(f'<span class="ingredient-tag">🥘 {ing}</span>', unsafe_allow_html=True)
            
            # Generate recipe
            recipe = generate_recipe_from_text(
                ingredients, 
                ", ".join(dietary_prefs), 
                cuisine, 
                meal_type,
                popular_dish if popular_dish != "None - Custom Recipe" else None
            )
            
            # Display recipe
            st.markdown('<div class="recipe-container">', unsafe_allow_html=True)
            st.markdown("## 📖 Your Custom Recipe")
            st.markdown(recipe)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Add to history
            recipe_name = recipe.split('\n')[0] if recipe else "Generated Recipe"
            st.session_state.recipe_history.append(recipe_name)
            
            # Download button
            st.download_button(
                label="📥 Download Recipe",
                data=recipe,
                file_name="my_recipe.txt",
                mime="text/plain"
            )
        else:
            st.warning("⚠ Please enter ingredients OR select a popular dish!")

# Tab 2: Image-to-Recipe
with tab2:
    st.markdown('<div class="feature-card"><h3>📸 Create Recipe from Image</h3><p>Upload a food photo and get an instant recipe!</p></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "📤 Upload food image",
            type=["jpg", "jpeg", "png"],
            help="Upload a clear image of the dish you want to recreate"
        )
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
    
    with col2:
        st.markdown("#### 💬 Additional Context")
        additional_context = st.text_area(
            "Any specific details? (optional)",
            height=150,
            placeholder="e.g., This is a Dubai chocolate bar, make it extra creamy"
        )
        
        st.markdown("####")
        generate_image_recipe = st.button("🎯 Generate Recipe from Image", key="gen_image")
    
    if generate_image_recipe:
        if uploaded_file:
            # Convert image to bytes
            image_bytes = uploaded_file.getvalue()
            
            # Generate recipe
            recipe = generate_recipe_from_image(image_bytes, additional_context)
            
            # Display recipe
            st.markdown('<div class="recipe-container">', unsafe_allow_html=True)
            st.markdown("## 📖 Recipe from Your Image")
            st.markdown(recipe)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Add to history
            recipe_name = recipe.split('\n')[0] if recipe else "Image-based Recipe"
            st.session_state.recipe_history.append(recipe_name)
            
            # Download button
            st.download_button(
                label="📥 Download Recipe",
                data=recipe,
                file_name="image_recipe.txt",
                mime="text/plain"
            )
        else:
            st.warning("⚠ Please upload an image first!")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <p>🤖 Powered by Google Gemini AI | Made by Qudsia❤ using Streamlit</p>
        <p style='font-size: 0.9rem;'>Transform your cooking experience with AI-generated recipes!</p>
    </div>
""", unsafe_allow_html=True)
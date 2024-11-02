from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
data = pd.read_csv('dataset.csv')
data['descriptionText'].fillna('', inplace=True)

# Vectorization
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(data['descriptionText'])

# Initialize FastAPI
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (adjust for production)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
async def root():
    return {"message": "Hello World"}

# Define request schema
class UserProfile(BaseModel):
    highest_degree: str
    field_of_study: str
    work_env: str
    skills: str
    career_interests: str
    work_values: str


# Create user profile vector based on user inputs
def create_user_profile_vector(user_profile: UserProfile):
    user_input = ' '.join([
        user_profile.highest_degree,
        user_profile.field_of_study,
        user_profile.work_env,
        user_profile.skills,
        user_profile.career_interests,
        user_profile.work_values
    ])
    user_input_vector = tfidf_vectorizer.transform([user_input])
    return user_input_vector


# API endpoint for recommendations
@app.post("/recommend")
async def recommend(user_profile: UserProfile):
    # Create user profile vector
    user_vector = create_user_profile_vector(user_profile)

    # Calculate cosine similarities
    cosine_similarities = cosine_similarity(user_vector, tfidf_matrix).flatten()

    # Get top 5 recommendations
    top_indices = cosine_similarities.argsort()[-5:][::-1]
    top_jobs = data.iloc[top_indices][["id","title","companyName","companyLogo","tertiaryDescription","descriptionText","employmentStatus","formattedLocation","link","standardizedTitle"]].to_dict(orient="records")

    return {"recommendations": top_jobs}

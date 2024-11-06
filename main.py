from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

data = pd.read_csv('dataset.csv')
data['descriptionText'].fillna('', inplace=True)

tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(data['descriptionText'])

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root() -> dict:
    return {"message": "Hello World"}

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


def filter_jobs_based_on_profile(user_profile: UserProfile):
    profile_fields = {
        "highest_degree": user_profile.highest_degree,
        "field_of_study": user_profile.field_of_study,
        "work_env": user_profile.work_env,
        "skills": user_profile.skills,
        "career_interests": user_profile.career_interests,
        "work_values": user_profile.work_values
    }

    data['match_score'] = data['descriptionText'].apply(lambda desc: sum(
        1 for field_value in profile_fields.values() if field_value.lower() in desc.lower()
    ))

    matched_jobs = data[data['match_score'] > 0].sort_values(by='match_score', ascending=False)

    top_jobs = matched_jobs.head(15)[
        ["id", "title", "companyName", "companyLogo", "tertiaryDescription",
         "descriptionText", "employmentStatus", "formattedLocation", "link",
         "standardizedTitle", "match_score"]
    ].to_dict(orient="records")

    return top_jobs

@app.post("/recommend-collab")
def recommend(user_profile: UserProfile) -> dict:
    try:
        user_vector = create_user_profile_vector(user_profile)
        cosine_similarities = cosine_similarity(user_vector, tfidf_matrix).flatten()
        top_indices = cosine_similarities.argsort()[-15:][::-1]

        top_jobs = data.iloc[top_indices][
            ["id", "title", "companyName", "companyLogo", "tertiaryDescription",
             "descriptionText", "employmentStatus", "formattedLocation", "link",
             "standardizedTitle"]
        ].to_dict(orient="records")

        return {"recommendations": top_jobs}
    except Exception as e:
        logger.error(f"Error in /recommend endpoint: {e}")  # Log the error
        raise HTTPException(status_code=500, detail="Internal Server Error")



@app.post("/recommend-knowledge")
def recommend(user_profile: UserProfile) -> dict:
    try:
        recommendations = filter_jobs_based_on_profile(user_profile)
        return {"recommendations": recommendations}
    except Exception as e:
        logger.error(f"Error in /recommend endpoint: {e}")  # Log the error
        raise HTTPException(status_code=500, detail="Internal Server Error")


# user_vector = create_user_profile_vector(user_profile)
#
# # Compute cosine similarities between the user profile and all job descriptions
# cosine_similarities = cosine_similarity(user_vector, tfidf_matrix).flatten()
#
# # Get the indices of the top 15 most similar job descriptions
# top_indices = cosine_similarities.argsort()[-15:][::-1]
#
# # Fetch the corresponding job details based on top indices
# top_jobs = data.iloc[top_indices][
#     ["id", "title", "companyName", "companyLogo", "tertiaryDescription",
#      "descriptionText", "employmentStatus", "formattedLocation", "link",
#      "standardizedTitle"]
# ].to_dict(orient="records")
#
# return {"recommendations": top_jobs}
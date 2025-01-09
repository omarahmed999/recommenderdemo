from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load and prepare the dataset
df = pd.read_csv('edx.csv')  # Replace with the correct file path
df['combined_text'] = df['Name'] + ' ' + df['Course Description'] + ' ' + df['Difficulty Level']
df = df.dropna(subset=['combined_text'])

# Initialize TF-IDF Vectorizer
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
tfidf_matrix = tfidf.fit_transform(df['combined_text'])

# Compute cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Create a mapping from course title to index
indices = pd.Series(df.index, index=df['Name']).drop_duplicates()

# Create Flask app
app = Flask(__name__)

# Function to get recommendations
def get_recommendations(Name, cosine_sim=cosine_sim, top_n=10):
    idx = indices[Name]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    course_indices = [i[0] for i in sim_scores]
    return df[['Name', 'University', 'Difficulty Level', 'Link', 'Course Description']].iloc[course_indices]

@app.route('/recommend', methods=['GET', 'POST'])
def recommend():
    courses = df['Name'].tolist()  # List of all course names for the dropdown
    recommendations = None
    error = None
    selected_course = None

    if request.method == 'POST':
        selected_course = request.form.get('course_title')
        try:
            recommendations = get_recommendations(selected_course)
        except KeyError:
            error = "Course not found or invalid selection!"

    return render_template(
        'recommendations.html', 
        courses=courses, 
        recommendations=recommendations, 
        error=error, 
        course_title=selected_course
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

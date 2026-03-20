import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_data(file_path):
    """
    Load dataset and create 'tags' column
    """
    try:
        data = pd.read_csv(file_path)
    except FileNotFoundError:
        print("❌ Dataset file not found!")
        return None

    data.fillna('', inplace=True)
    data['tags'] = data['genres'] + " " + data['keywords']

    return data


def create_similarity_matrix(data):
    """
    Convert text into vectors and compute similarity matrix
    """
    vectorizer = CountVectorizer(max_features=5000, stop_words='english')
    vectors = vectorizer.fit_transform(data['tags']).toarray()

    similarity = cosine_similarity(vectors)
    return similarity


def recommend_movies(movie_name, data, similarity, top_n=5):
    """
    Recommend similar movies based on input
    """
    movie_name = movie_name.lower()
    data['title_lower'] = data['title'].str.lower()

    if movie_name not in data['title_lower'].values:
        return None

    index = data[data['title_lower'] == movie_name].index[0]

    distances = list(enumerate(similarity[index]))
    sorted_movies = sorted(distances, key=lambda x: x[1], reverse=True)

    recommendations = []
    for i in sorted_movies[1:top_n + 1]:
        recommendations.append(data.iloc[i[0]].title)

    return recommendations


def display_recommendations(recommendations):
    """
    Display recommendations nicely
    """
    print("\n🎬 Recommended Movies:\n")

    for i, movie in enumerate(recommendations, start=1):
        print(f"{i}. {movie}")


def main():
    print("=" * 40)
    print("🎬 Movie Recommendation System")
    print("=" * 40)

    data = load_data("movies.csv")
    if data is None:
        return

    similarity = create_similarity_matrix(data)

    while True:
        movie = input("\nEnter movie name (or type 'exit'): ").strip()

        if movie.lower() == "exit":
            print("👋 Exiting... Goodbye!")
            break

        if not movie:
            print("⚠ Please enter a valid movie name.")
            continue

        recommendations = recommend_movies(movie, data, similarity)

        if recommendations is None:
            print("❌ Movie not found. Try another one.")
        else:
            display_recommendations(recommendations)


if __name__ == "__main__":
    main()

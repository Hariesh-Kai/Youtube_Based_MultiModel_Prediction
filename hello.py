import isodate
from googleapiclient.discovery import build

# Emotion-to-Music Mapping
emotion_music_mapping = {
    "angry": ["rock", "metal", "energetic", "animated movie"],
    "disgust": ["calm", "relaxing", "ambient", "animated movie"],
    "fear": ["uplifting", "motivational", "soft", "animated movie"],
    "happiness": ["pop", "dance", "party", "animated movie"],
    "neutrality": ["acoustic", "lofi", "chill", "animated movie"],
    "sadness": ["sad", "melancholic", "slow", "animated movie"],
    "surprise": ["exciting", "upbeat", "electronic", "animated movie"]
}

def fetch_trending_songs(emotion, age, gender, languages, recommended_genre, api_key):
    # Use the recommended_genre if provided; otherwise, fall back on emotion mapping
    genres = [recommended_genre] if recommended_genre else emotion_music_mapping.get(emotion, ["Trending"])
    youtube = build('youtube', 'v3', developerKey=api_key)

    genres = emotion_music_mapping.get(emotion, ["Trending"])
    songs = []

    gender_specific = "boys" if gender == "Male" else "girls" if age in ["Child", "Teenager"] else "men" if gender == "Male" else "women"
    
    for lang in languages:
        for genre in genres:
            if age in ["Child", "Teenager"]:
                search_query = f"latest {genre} songs in {lang} for {gender_specific}"
            elif age == "Adult":
                search_query = f"top trending {genre} songs in {lang} for {gender_specific}"
            else:  # Older Adult
                search_query = f"classic {genre} songs in {lang}"

            request = youtube.search().list(
                q=search_query,
                part="snippet",
                maxResults=6,  # Fetch 6 songs instead of 5
                type="video"
            )
            response = request.execute()
            for item in response['items']:
                video_id = item['id']['videoId']
                video_details = youtube.videos().list(
                    part="contentDetails",
                    id=video_id
                ).execute()

                # Get the duration in seconds
                duration = isodate.parse_duration(video_details['items'][0]['contentDetails']['duration']).total_seconds()
                if 120 <= duration <= 600:  # Filter songs between 2 and 10 minutes
                    title = item['snippet']['title']
                    songs.append({"title": title, "video_id": video_id})

    return songs[:6]  # Return only the top 6 songs

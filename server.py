import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from urllib.parse import parse_qs, urlparse
import json
import pandas as pd
from datetime import datetime
import uuid
import os
from typing import Callable, Any
from wsgiref.simple_server import make_server

nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('stopwords', quiet=True)

adj_noun_pairs_count = {}
sia = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))

reviews = pd.read_csv('data/reviews.csv').to_dict('records')

VALID_LOCATIONS = [
    "Albuquerque, New Mexico", "Carlsbad, California", "Chula Vista, California",
    "Colorado Springs, Colorado", "Denver, Colorado", "El Cajon, California",
    "El Paso, Texas", "Escondido, California", "Fresno, California",
    "La Mesa, California", "Las Vegas, Nevada", "Los Angeles, California",
    "Oceanside, California", "Phoenix, Arizona", "Sacramento, California",
    "Salt Lake City, Utah", "San Diego, California", "Tucson, Arizona"
]

class ReviewAnalyzerServer:
    def __init__(self) -> None:
        # This method is a placeholder for future initialization logic
        pass

    def analyze_sentiment(self, review_body):
        sentiment_scores = sia.polarity_scores(review_body)
        return sentiment_scores

    def __call__(self, environ: dict[str, Any], start_response: Callable[..., Any]) -> bytes:
        """
        The environ parameter is a dictionary containing some useful
        HTTP request information such as: REQUEST_METHOD, CONTENT_LENGTH, QUERY_STRING,
        PATH_INFO, CONTENT_TYPE, etc.
        """

        if environ["REQUEST_METHOD"] == "GET":
            query_params = parse_qs(environ["QUERY_STRING"])
            
            # Filter by location
            location = query_params.get("location", [None])[0]
            start_date = query_params.get("start_date", [None])[0]
            end_date = query_params.get("end_date", [None])[0]
            
            filtered_reviews = reviews
            
            if location:
                filtered_reviews = [review for review in filtered_reviews if review["Location"] == location]
            
            if start_date:
                start_date_dt = datetime.strptime(start_date, "%Y-%m-%d")
                filtered_reviews = [review for review in filtered_reviews if datetime.strptime(review["Timestamp"], "%Y-%m-%d %H:%M:%S") >= start_date_dt]
            
            if end_date:
                end_date_dt = datetime.strptime(end_date, "%Y-%m-%d")
                filtered_reviews = [review for review in filtered_reviews if datetime.strptime(review["Timestamp"], "%Y-%m-%d %H:%M:%S") <= end_date_dt]
            
            # Add sentiment analysis to each review
            for review in filtered_reviews:
                review["sentiment"] = self.analyze_sentiment(review["ReviewBody"])
            
            # Sort by compound sentiment score in descending order
            filtered_reviews.sort(key=lambda x: x["sentiment"]["compound"], reverse=True)
            
            response_body = json.dumps(filtered_reviews, indent=2).encode("utf-8")
            
            start_response("200 OK", [
                ("Content-Type", "application/json"),
                ("Content-Length", str(len(response_body)))
            ])
            
            return [response_body]


        if environ["REQUEST_METHOD"] == "POST":
            try:
                content_length = int(environ.get("CONTENT_LENGTH", 0))
                request_body = environ["wsgi.input"].read(content_length).decode("utf-8")
                post_data = parse_qs(request_body)

                review_body = post_data.get("ReviewBody", [None])[0]
                location = post_data.get("Location", [None])[0]

                # Validate the review body and location
                if not review_body or not location:
                    start_response("400 Bad Request", [("Content-Type", "application/json")])
                    return [b'{"error": "ReviewBody and Location are required."}']

                # Check if the location is valid
                if location not in VALID_LOCATIONS:
                    start_response("400 Bad Request", [("Content-Type", "application/json")])
                    return [b'{"error": "Invalid location."}']

                new_review = {
                    "ReviewId": str(uuid.uuid4()),
                    "ReviewBody": review_body,
                    "Location": location,
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }

                new_review["sentiment"] = self.analyze_sentiment(new_review["ReviewBody"])
                reviews.append(new_review)

                response_body = json.dumps(new_review, indent=2).encode("utf-8")
                start_response("201 Created", [
                    ("Content-Type", "application/json"),
                    ("Content-Length", str(len(response_body)))
                ])

                return [response_body]

            except Exception as e:
                start_response("500 Internal Server Error", [("Content-Type", "application/json")])
                return [f'{{"error": "{str(e)}"}}'.encode("utf-8")]


if __name__ == "__main__":
    app = ReviewAnalyzerServer()
    port = os.environ.get('PORT', 8000)
    with make_server("", port, app) as httpd:
        print(f"Listening on port {port}...")
        httpd.serve_forever()
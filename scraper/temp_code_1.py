# Import libraries
import requests
import json

# Define the video ID and the API key
video_id = "dQw4w9WgXcQ" # Change this to the video ID you want
api_key = "AIzaSyD-9tSrke72PouQMnMX-a7eZSW0jkFMBWY" # Change this to your own API key

# Define the base URL and the parameters
base_url = "https://www.googleapis.com/youtube/v3/commentThreads"
params = {
    "part": "snippet,replies",
    "videoId": video_id,
    "key": api_key,
    "maxResults": 100 # Change this to the number of comments you want
}

# Make a request to the API and get the response
response = requests.get(base_url, params=params)
data = response.json()

# Loop through the data and print the comments
for item in data["items"]:
    # Get the top level comment
    comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
    print(comment)
    
    # Get the replies if any
    if "replies" in item:
        for reply in item["replies"]["comments"]:
            comment = reply["snippet"]["textDisplay"]
            print(comment)

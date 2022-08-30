"""
todo
- save to cloud
"""

import requests
import pandas as pd
from datetime import datetime
import os


BASE_URL = "https://openapi.radiofrance.fr/v1/graphql"
API_TOKEN = os.environ.get("API_TOKEN")
endpoint = f"{BASE_URL}?x-token={API_TOKEN}"


def get_show_url(show_id):
    """Get the URL of a show, given its ID"""

    query = """
    {
        show(id: "SHOW_ID") {
            id
            url
            title
            }
    }
    """

    query = query.replace("SHOW_ID", show_id)

    response = requests.post(endpoint, json={"query": query})
    response = response.json()

    show_url = response["data"]["show"]["url"]

    return show_url


def get_last_diffusions(show_url, number_diffusions):
    """Retrieve MP3 URLs for given number of last diffusions and given show by show URL"""

    query = """
    {
        diffusionsOfShowByUrl(url: "SHOW_URL", first: NUMBER_DIFFUSIONS) {
            edges {
                cursor
                node {
                    id
                    title
                    published_date
                    podcastEpisode {
                        url
                    }
                }
            }
        }
    }
    """

    query = query.replace("SHOW_URL", show_url).replace("NUMBER_DIFFUSIONS", number_diffusions)

    response = requests.post(endpoint, json={"query": query})
    response = response.json()

    # Clean a bit within JSON response

    diffusions = response["data"]["diffusionsOfShowByUrl"]["edges"]

    # From cleaned response, retrieve diffusion ID, title, date and MP3 URL
    # Additional condition to not include elements when diffusions didn't happen (because of strikes)

    diffusions_list = [
        {
            "id": diffusion["node"]["id"],
            "title": diffusion["node"]["title"],
            "date": diffusion["node"]["published_date"],
            "url": diffusion["node"]["podcastEpisode"]["url"]
        } for diffusion in diffusions if diffusion["node"]["podcastEpisode"] is not None]

    # Put into a DataFrame and convert POSIX timestamp into YYYYMMDD format

    diffusions_df = pd.DataFrame(diffusions_list)
    diffusions_df["date"] = pd.to_datetime(diffusions_df["date"], unit = "s").dt.strftime("%Y%m%d")

    return diffusions_df


def read_diffusion_history(history_path):
    """Read CSV containing history of diffusions"""
    history_df = pd.read_csv(history_path)

    return history_df


def save_diffusion_to_history(diffusions_df, history_path):
    """Append newest diffusions to history of diffusions"""

    history_df = read_diffusion_history(history_path)

    today = datetime.today().strftime("%Y%m%d") # Get supposed last diffusion date
    today = int(today)

    if today in history_df["date"].values:
        print(f"🟩 Last diffusion of {today} already in history file")
    else:
        diffusions_df.to_csv(history_path, index = False, mode = "a", header = False)
        print(f"🟩 Last diffusion of {today} added to history file")

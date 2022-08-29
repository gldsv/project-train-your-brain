import requests
import pandas as pd
import os

BASE_URL = "https://openapi.radiofrance.fr/v1/graphql"
API_TOKEN = os.environ.get("API_TOKEN")
SHOW_ID = "48f74c6a-eb98-11e1-a7b7-782bcb76618d_1" # Jeu des 1000 € unique ID
number_diffusions = "1" # Last diffusion

endpoint = f"{BASE_URL}?x-token={API_TOKEN}"

# Retrieve information (including url) about given show via id of the show

query_show = """
{
  show(id: "SHOW_ID") {
    id
    url
    title
    }
}
"""

query_show = query_show.replace("SHOW_ID", SHOW_ID)

response = requests.post(endpoint, json={"query": query_show})
show_json = response.json()

# Get show URL (via API in case URL changes - hoping ID won't)

show_url = show_json["data"]["show"]["url"]
show_url


# Retrieve URLs for MP3 diffusions for given number of diffusions for given show by show URL

query_diffusions = """
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

query_diffusions = query_diffusions.replace("SHOW_URL", show_url).replace("NUMBER_DIFFUSIONS", number_diffusions)

response = requests.post(endpoint, json={"query": query_diffusions})
diffusions_json = response.json()

# Clean a bit within JSON response

diffusions = diffusions_json["data"]["diffusionsOfShowByUrl"]["edges"]

# From cleaned response, retrieve diffusion ID and MP3 URL
# Additional condition to not include elements when diffusions didn't happen (because of strikes)

diffusions_list = [
    {
        "id": diffusion["node"]["id"],
        "title": diffusion["node"]["title"],
        "date": diffusion["node"]["published_date"],
        "url": diffusion["node"]["podcastEpisode"]["url"]
    } for diffusion in diffusions if diffusion["node"]["podcastEpisode"] is not None]

# Put into a DataFrame

diffusions_df = pd.DataFrame(diffusions_list)
diffusions_df["date"] = pd.to_datetime(diffusions_df["date"], unit = "s").dt.strftime("%Y%m%d")

header = diffusions_df.columns

filename = os.path.join(os.environ.get("LOCAL_DATA_PATH"),
        "daily_diffusions")

diffusions_df.to_csv(filename+".csv", index = False, mode = "a", header = False)
diffusions_df

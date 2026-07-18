import os
import requests

def download_images(urls, folder):
    os.makedirs(folder, exist_ok=True)
    for i, url in enumerate(urls):
        try:
            response = requests.get(url, timeout=10)
            with open(os.path.join(folder, f"image_{i}.jpg"), 'wb') as f:
                f.write(response.content)
            print(f"Downloaded: {i+1}/{len(urls)}")
        except Exception as e:
            print(f"Failed to download {url}: {e}")

# Add your collected URLs here
bullfrog_urls = [
    "https://inaturalist-open-data.s3.amazonaws.com/photos/12345/original.jpg",
    "https://inaturalist-open-data.s3.amazonaws.com/photos/67890/original.jpg"
]
cane_toad_urls = [
    "https://example.com/image3.jpg",
    "https://example.com/image4.jpg"
]

download_images(bullfrog_urls, "data/bullfrog")
download_images(cane_toad_urls, "data/cane_toad")
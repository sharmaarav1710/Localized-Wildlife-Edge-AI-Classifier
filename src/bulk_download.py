import os
import requests
import time

TAXA = {
    "bullfrog": 24338,
    "cane_toad": 64366
}

def fetch_inat_images(name, taxon_id, limit=60, output_dir="data"):
    target_folder = os.path.join(output_dir, name)
    os.makedirs(target_folder, exist_ok=True)
    
    # Use the /v1/observations endpoint
    base_url = "https://api.inaturalist.org/v1/observations"
    
    # Explicitly override defaults to ensure we get results
    params = {
        "taxon_id": taxon_id,
        "per_page": limit,
        "media": "photo",
        "quality_grade": "any",      # Override default 'research' grade if it's too restrictive
        "verifiable": "any"          # Override default 'verifiable=true'
    }
    
    print(f"Fetching {limit} images for {name} (ID: {taxon_id})...")
    response = requests.get(base_url, params=params)
    print(f"DEBUG: Status Code {response.status_code}")
    
    if response.status_code != 200:
        return

    data = response.json()
    count = 0
    for obs in data.get('results', []):
        for photo in obs.get('photos', []):
            # 'large' is a valid size
            img_url = photo['url'].replace("square", "large") 
            try:
                img_data = requests.get(img_url).content
                file_path = os.path.join(target_folder, f"{obs['id']}.jpg")
                with open(file_path, 'wb') as f:
                    f.write(img_data)
                count += 1
                print(f"[{count}/{limit}] Saved to {file_path}")
                time.sleep(0.5) 
                break 
            except Exception as e:
                print(f"Failed {obs['id']}: {e}")
        if count >= limit:
            break

for name, tid in TAXA.items():
    fetch_inat_images(name, tid, limit=60)
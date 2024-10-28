import requests

# Base URL for downloading files
base_url = "https://raw.githubusercontent.com/opencv/opencv/refs/heads/4.x/data/haarcascades/"

# List of file names to download
haarcascades = [
    "haarcascade_frontalface_default.xml",
    "haarcascade_eye.xml",
    "haarcascade_smile.xml",
    "haarcascade_eye_tree_eyeglasses.xml",
    "haarcascade_frontalface_alt.xml",
    "haarcascade_frontalface_alt2.xml",
    "haarcascade_frontalface_alt_tree.xml",
    "haarcascade_fullbody.xml",
    "haarcascade_lefteye_2splits.xml",
    "haarcascade_licence_plate_rus_16stages.xml",
    "haarcascade_lowerbody.xml",
    "haarcascade_profileface.xml",
    "haarcascade_righteye_2splits.xml",
    "haarcascade_russian_plate_number.xml",
    "haarcascade_upperbody.xml"
]

# Loop through each file and download it
for file_name in haarcascades:
    print(f"Downloading {file_name}...")
    url = f"{base_url}{file_name}"
    response = requests.get(url)
    
    # Check if the request was successful
    if response.status_code == 200:
        with open(file_name, 'wb') as file:
            file.write(response.content)
        print(f"{file_name} downloaded successfully.")
    else:
        print(f"Failed to download {file_name}. Status code: {response.status_code}")

print("Download complete!")

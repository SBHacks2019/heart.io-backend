from google_images_download import google_images_download
import json

all_diseases = [
    'Actinic keratoses',
    'Basal cell carcinoma',
    'Benign keratosis-like lesions',
    'Dermatofibroma',
    'Melanocytic nevi',
    'Melanoma',
    'Vascular lesions'
]

image_scraper = google_images_download.googleimagesdownload()
for disease_type in all_diseases:
    image_scraper.download({
        'keywords': f'"{disease_type}"',
        'extract_metadata': True,
        'language': 'English',
        'limit': 100,
        'no_directory': True,
        'output_directory': f'scraped_data/{disease_type}'.replace(' ', '_').lower()
    })
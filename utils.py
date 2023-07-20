import os
import io
import time
import base64
import requests
from PIL import Image


def create_base64_encoded_credentials(username, password):
    credentials = f"{username}:{password}"

    # Encode the credentials to base64
    base64_encoded_credentials = base64.b64encode(credentials.encode()).decode()

    print(base64_encoded_credentials)
    return base64_encoded_credentials


def text2img(api_url, params, base64_encoded_credentials, **kwargs):
    """
    http://127.0.0.1:53511/sdapi/v1/txt2img

    """
    headers = {
        "Authorization": f"Basic {base64_encoded_credentials}",
        "Content-Type": "application/json"  # Set the appropriate content type for your request
    }

    try:
        response = requests.post(api_url, json=params, headers=headers)

        if response.status_code == 200:
            # Successful response
            return response.json()
        else:
            # Handle other status codes or errors
            print(f"Error: {response.status_code} - {response.text}")

    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")


def text_to_image(image_text, save_path):
    image_bytes = base64.b64decode(image_text)
    image = Image.open(io.BytesIO(image_bytes))
    image.save(save_path)


if __name__ == '__main__':
    credential = create_base64_encoded_credentials('sijieapi', '92wsj___WIN')
    url = 'http://58.34.1.33:53511/sdapi/v1/txt2img'
    params = {'enable_hr': 'true',
              'denoising_strength': 0.6,
              'firstphase_width': 0,
              'firstphase_height': 0,
              'hr_scale': 2,
              'hr_upscaler': 'Latent',
              'hr_second_pass_steps': 0,
              'hr_resize_x': 2048,
              'hr_resize_y': 1024,
              'hr_sampler_name': 'DPM++ 2M Karras',
              'hr_prompt': '',
              'hr_negative_prompt': '',
              'prompt': 'large old city, view from hillside, landscape of a Absurd [Mexico|Bamboo Forest] and Ljubljana, Hurricane, Smiling, high quality, absurdres, masterpiece, <lora:add_detail:1>, scenery, outdoors, cloud, sky, day, solo, castle, building, standing, 1girl, landscape, tree, cloudy sky, cape, grass, mountain, from behind, water, blue sky, river, fantasy, red cape, horizon, city, facing away, hill, tower, very wide shot, nature, 1other, ambiguous gender, cliff',
              'styles': ['string'],
              'seed': 4236284727,
              'subseed': -1,
              'subseed_strength': 0,
              'seed_resize_from_h': -1,
              'seed_resize_from_w': -1,
              'sampler_name': 'DPM++ 2M Karras',
              'batch_size': 1,
              'n_iter': 1,
              'steps': 30,
              'cfg_scale': 7,
              'width': 1024,
              'height': 512,
              'restore_faces': 'true',
              'tiling': 'false',
              'do_not_save_samples': 'false',
              'do_not_save_grid': 'false',
              'negative_prompt': '(worst quality, low quality:1.4),CyberRealistic_Negative-neg,UnrealisticDream,aid291,an5,bad-artist,bad-artist-anime,bad-image-v2-39000,BadDream,badv4,V2 FastNegativeEmbedding lr NegfeetV2 nobg notxt Unspeakable-Horrors-Composition-4v  verybadimagenegative_v1.3,EasyNegative,',
              'eta': 0,
              's_min_uncond': 0,
              's_churn': 0,
              's_tmax': 0,
              's_tmin': 0,
              's_noise': 1,
              'override_settings': {},
              'override_settings_restore_afterwards': 'true',
              'script_args': [],
              'sampler_index': 'DPM++ 2M Karras',
              'script_name': '',
              'send_images': 'true',
              'save_images': 'false',
              'alwayson_scripts': {}}
    data = text2img(url, params=params, base64_encoded_credentials=credential)
    text_to_image(data['images'][0], f'./str({int(time.time())}).png')

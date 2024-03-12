import os
import io
import time
import base64
import requests
from PIL import Image
from typing import Dict


class StableDiffusionApi(object):
    USERNAME = 'xxx'
    PASSWORD = 'xxx'
    MAIN_URL = 'xxx'

    def __init__(self):
        self.headers = {
            "Authorization": f"Basic {self.create_base64_encoded_credentials()}",
            "Content-Type": "application/json"
        }

    def create_base64_encoded_credentials(self):
        credentials = f"{self.USERNAME}:{self.PASSWORD}"

        # Encode the credentials to base64
        base64_encoded_credentials = base64.b64encode(credentials.encode()).decode()

        print(base64_encoded_credentials)
        return base64_encoded_credentials

    def configure_sd(self, params: Dict):
        """
        /sdapi/v1/options
        """
        url = f'{self.MAIN_URL}/sdapi/v1/options'
        resp = requests.post(url, json=params, headers=self.headers)
        assert resp.status_code == 200

    def get_configuration(self):
        """
        /sdapi/v1/options
        """
        url = f'{self.MAIN_URL}/sdapi/v1/options'
        resp = requests.get(url, headers=self.headers)
        print(resp.json())
        if resp.status_code != 200:
            raise Exception(resp.text)
        return resp.json()

    def text2img(self, params, **kwargs):
        """
        /sdapi/v1/txt2img
        """
        url = f'{self.MAIN_URL}/sdapi/v1/txt2img'
        try:
            response = requests.post(url, json=params, headers=self.headers)

            if response.status_code == 200:
                # Successful response
                return response.json()
            else:
                # Handle other status codes or errors
                print(f"Error: {response.status_code} - {response.text}")

        except requests.exceptions.RequestException as e:
            print(f"Error: {e}")

    @staticmethod
    def text_to_image(image_text, save_path):
        image_bytes = base64.b64decode(image_text)
        image = Image.open(io.BytesIO(image_bytes))
        image.save(save_path)

    def get_sd_models(self):
        """
        /sdapi/v1/sd-models
        """
        url = f'{self.MAIN_URL}/sdapi/v1/sd-models'
        resp = requests.get(url, headers=self.headers)
        return resp.json()

    def get_sd_vaes(self):
        """
        /sdapi/v1/sd-vae
        """
        url = f'{self.MAIN_URL}/sdapi/v1/sd-vae'
        resp = requests.get(url, headers=self.headers)
        return resp.json()


if __name__ == '__main__':
    sd = StableDiffusionApi()
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

    # configuration = sd.get_configuration()
    # print(configuration)

    # print(sd.get_sd_vaes())

    # print(sd.get_sd_models())

    sd.configure_sd(
        {'sd_model_checkpoint': 'darkSushiMixMix_225D.safetensors [cca17b08da]'}
    )

    data = sd.text2img(params=params)
    sd.text_to_image(data['images'][0], f'./{str(int(time.time()))}.png')

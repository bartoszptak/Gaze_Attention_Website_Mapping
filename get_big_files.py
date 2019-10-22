import requests
import zipfile
import os


def download_file_from_google_drive(id, destination):
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768

        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)

    print('GET: {}'.format(destination))
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)

def download_unzip_and_remove(google_code, name):
    download_file_from_google_drive(
        google_code, name)

    with zipfile.ZipFile(name, 'r') as zip_ref:
        zip_ref.extractall('data')

    os.remove(name)


if __name__ == "__main__":
    import sys

    # face predictor
    download_file_from_google_drive(
        '1TXJn_tAKkgmg9aMAVUrY8E2A9xUpoxLl', os.path.join('data','shape_predictor_68_face_landmarks.dat'))

    # tensorflow model
    download_unzip_and_remove('1wPEBjl6NjpQOhh-J2ZoR3XxxUPb7do4B', os.path.join('data','model.zip'))

    # datasets
    download_unzip_and_remove('1AQ-ToGm4-PG2HlEdnvVEzX73sf-XOBL5', os.path.join('data','dataset.zip'))

    # train logs
    download_unzip_and_remove('1DyHAYc3qOjl4odaeI9YgW82PhTV5ZVHE', os.path.join('data','train.zip'))
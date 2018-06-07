import multiprocessing
import os
from io import BytesIO
import urllib
import pandas as pd
import re
import tqdm
from PIL import Image


# set files and dir
DATA_FRAME, OUT_DIR = pd.read_csv('index.csv'), 'index'  # recognition challenge
# DATA_FRAME, OUT_DIR = pd.read_csv('../input/index.csv'), '../input/index'  # retrieval challenge
# DATA_FRAME, OUT_DIR = pd.read_csv('../input/test.csv'), '../input/test'  # test data

# preferences
TARGET_SIZE = 128  # image resolution to be stored
IMG_QUALITY = 90  # JPG quality
NUM_WORKERS = 8 # Num of CPUs


DATA_FRAME.url.apply(lambda x: x.split('/')[-2]).value_counts().head()

def overwrite_urls(df):
    def reso_overwrite(url_tail, reso=TARGET_SIZE):
        pattern = 's[0-9]+'
        search_result = re.match(pattern, url_tail)
        if search_result is None:
            return url_tail
        else:
            return 's{}'.format(reso)

    def join_url(parsed_url, s_reso):
        parsed_url[-2] = s_reso
        return '/'.join(parsed_url)

    parsed_url = df.url.apply(lambda x: x.split('/'))
    train_url_tail = parsed_url.apply(lambda x: x[-2])
    resos = train_url_tail.apply(lambda x: reso_overwrite(x, reso=TARGET_SIZE))

    overwritten_df = pd.concat([parsed_url, resos], axis=1)
    overwritten_df.columns = ['url', 's_reso']
    df['url'] = overwritten_df.apply(lambda x: join_url(x['url'], x['s_reso']), axis=1)
    return df


def parse_data(df):
    key_url_list = [line[:2] for line in df.values]
    return key_url_list


def download_image(key_url):
    (key, url) = key_url
    filename = os.path.join(OUT_DIR, '{}.jpg'.format(key))

    if os.path.exists(filename):
        print('Image {} already exists. Skipping download.'.format(filename))
        return 0

    try:
        response = urllib.urlopen(url)
        image_data = response.read()
    except:
        print('Warning: Could not download image {} from {}'.format(key, url))
        return 1

    try:
        pil_image = Image.open(BytesIO(image_data))
    except:
        print('Warning: Failed to parse image {}'.format(key))
        return 1

    try:
        pil_image_rgb = pil_image.convert('RGB')
    except:
        print('Warning: Failed to convert image {} to RGB'.format(key))
        return 1

    try:
        pil_image_resize = pil_image_rgb.resize((TARGET_SIZE, TARGET_SIZE))
    except:
        print('Warning: Failed to resize image {}'.format(key))
        return 1

    try:
        pil_image_resize.save(filename, format='JPEG', quality=IMG_QUALITY)
    except:
        print('Warning: Failed to save image {}'.format(filename))
        return 1

    return 0


def loader(df):
    if not os.path.exists(OUT_DIR):
        os.mkdir(OUT_DIR)

    key_url_list = parse_data(df)
    pool = multiprocessing.Pool(processes=NUM_WORKERS)
    failures = sum(tqdm.tqdm(pool.imap_unordered(download_image, key_url_list),
                             total=len(key_url_list)))
    print('Total number of download failures:', failures)
    pool.close()
    pool.terminate()

if __name__ == '__main__':
    loader(overwrite_urls(DATA_FRAME))
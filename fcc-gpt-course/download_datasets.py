import os
import requests
import tarfile
import lzma
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_fixed

# 基础URL
base_url = "https://huggingface.co/datasets/Skylion007/openwebtext/resolve/main/subsets/"

# 定义重试机制：最多重试 5 次，每次间隔 2 秒
@retry(stop=stop_after_attempt(5), wait=wait_fixed(2))
def download_file(url, file_path):
    response = requests.get(url, stream=True)
    
    if response.status_code == 200:
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 每次读取的块大小 (1KB)
        
        with tqdm(total=total_size, unit='iB', unit_scale=True) as progress_bar:
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=block_size):
                    if chunk:  # 过滤掉keep-alive的新块
                        f.write(chunk)
                        progress_bar.update(len(chunk))
        print(f"Downloaded {file_path}")
    else:
        raise requests.exceptions.RequestException(f"Failed to download {url}. Status code: {response.status_code}")

# 下载 tar 文件
def download_openwebtext(save_path="openwebtext_subsets", range_num=21):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for i in range(range_num):
        subset_name = f"urlsf_subset{i:02d}.tar"
        url = base_url + subset_name + "?download=true"
        file_path = os.path.join(save_path, subset_name)
        try:
            print(f"Downloading {subset_name}...")
            download_file(url, file_path)
        except Exception as e:
            print(f"Error downloading {subset_name}: {e}")

def tar_files_in_dir(directory):
    files = []
    for filename in os.listdir(directory):
        if filename.endswith(".tar") and os.path.isfile(os.path.join(directory, filename)):
            files.append(filename)
    return files

def extract_tar(file_path, extract_to):
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)

    with tarfile.open(file_path, "r") as tar:
        members = tar.getmembers()
        with tqdm(total=len(members), unit="file", desc=f"Extracting {file_path}") as pbar:
            def progress(members):
                for member in members:
                    yield member
                    pbar.update(1)

            for member in progress(members):
                if member.name.endswith(".xz"):
                    member.name = os.path.basename(member.name)
                    tar.extract(member, path=extract_to)

def process_xz_files_in_folder(folder, output_file, vocab):
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith(".xz"):
                xz_file_path = os.path.join(root, file)
                try:
                    with lzma.open(xz_file_path, "rt", encoding="utf-8") as infile:
                        text = infile.read()
                        output_file.write(text)
                        characters = set(text)
                        vocab.update(characters)
                except Exception as e:
                    print(f"Error processing {xz_file_path}: {e}")

def xz_files_in_dir(directory):
    files = []
    for filename in os.listdir(directory):
        if filename.endswith("xz") and os.path.isfile(os.path.join(directory, filename)):
            files.append(filename)
    return files

if __name__ == "__main__":
    folder_path = "openwebtext_subsets"
    extract_folder = "openwebtext_xz"  # 统一解压路径
    output_file_train = "train_split.txt"
    output_file_val = "val_split.txt"
    vocab_file = "vocab.txt"

    # 第一个循环：解压所有 tar 文件到 "openwebtext_xz"
    files = tar_files_in_dir(folder_path)
    for filename in files:
        file_path = os.path.join(folder_path, filename)
        extract_tar(file_path, extract_folder)

    # 第二个循环：处理 "openwebtext_xz" 目录中的所有 .xz 文件
    vocab = set()

    xz_files = xz_files_in_dir(extract_folder)

    total_files = len(xz_files)
    split_index = int(total_files * 0.9)
    files_train = xz_files[:split_index]  # 修正为 xz_files 列表
    files_val = xz_files[split_index:]  # 修正为 xz_files 列表

    print('Start processing train data.')
    with open(output_file_train, "w", encoding="utf-8") as outfile:
        for filename in tqdm(files_train, total=len(files_train)):
            file_path = os.path.join(extract_folder, filename)
            with lzma.open(file_path, "rt", encoding="utf-8") as infile:
                text = infile.read()
                outfile.write(text)
                characters = set(text)
                vocab.update(characters)

    print('Start processing valid data.')
    with open(output_file_val, "w", encoding="utf-8") as outfile:
        for filename in tqdm(files_val, total=len(files_val)):
            file_path = os.path.join(extract_folder, filename)
            with lzma.open(file_path, "rt", encoding="utf-8") as infile:
                text = infile.read()
                outfile.write(text)
                characters = set(text)
                vocab.update(characters)

    print('Start processing vocab file.')
    with open(vocab_file, "w", encoding="utf-8") as vfile:
        for char in vocab:
            vfile.write(char + '\n')

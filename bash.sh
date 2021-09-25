
git clone https://github.com/oaeka/effective-journey.git
cd effective-journey/

rm -rf Dataset
mkdir Dataset
cd Dataset

wget https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip
wget https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip

unzip Flickr8k_Dataset.zip
unzip Flickr8k_text.zip

rm -rf  __MACOSX
cd ..

pip install -r requirements.txt

python3 data/inception_features.py
python3 train.py

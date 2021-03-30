pip install -r requirements.txt

python -m spacy download fr_core_news_sm
python -m spacy download en_core_web_sm

git clone https://github.com/multi30k/dataset.git

cp dataset/data/task1/raw/train.fr.gz .data/multi30k
cp dataset/data/task1/raw/val.fr.gz .data/multi30k

gzip -d .data/multi30k/train.fr.gz
gzip -d .data/multi30k/val.fr.gz
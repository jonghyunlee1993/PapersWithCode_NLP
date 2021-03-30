pip install -r requirements.txt

!mkdir -p .data/multi30k
!git clone https://github.com/multi30k/dataset.git

!cp dataset/data/task1/raw/train.fr.gz .data/multi30k
!cp dataset/data/task1/raw/train.en.gz .data/multi30k

!cp dataset/data/task1/raw/val.fr.gz .data/multi30k
!cp dataset/data/task1/raw/val.en.gz .data/multi30k

!cp dataset/data/task1/raw/test_2016_flickr.fr.gz .data/multi30k
!cp dataset/data/task1/raw/test_2016_flickr.en.gz .data/multi30k

!gzip -d .data/multi30k/train.fr.gz
!gzip -d .data/multi30k/train.en.gz

!gzip -d .data/multi30k/val.fr.gz
!gzip -d .data/multi30k/val.en.gz

!gzip -d .data/multi30k/test_2016_flickr.fr.gz
!gzip -d .data/multi30k/test_2016_flickr.en.gz

!mv .data/multi30k/test_2016_flickr.fr .data/multi30k/test2016.fr
!mv .data/multi30k/test_2016_flickr.en .data/multi30k/test2016.en

!python -m spacy download fr_core_news_sm
!python -m spacy download en_core_web_sm
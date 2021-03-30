pip3 install -r requirements.txt

mv .data/multi30k/test_2016_flickr.fr .data/multi30k/test2016.fr
mv .data/multi30k/test_2016_flickr.en .data/multi30k/test2016.en

python3 -m spacy download fr_core_news_sm
python3 -m spacy download en_core_web_sm
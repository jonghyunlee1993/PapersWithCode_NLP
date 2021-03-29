pip install -r requirements.txt

echo "Download spacy module ... "

python -m spacy download fr_core_news_sm
python -m spacy download en_core_web_sm

echo "spacy fr, en core successfully downloaded!"
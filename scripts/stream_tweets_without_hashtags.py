from __future__ import absolute_import, print_function

from tweepy.streaming import StreamListener
from tweepy import Stream
import json
import sys
from http.client import IncompleteRead
import time
import os
from urllib3.exceptions import ProtocolError
from tweepy import OAuthHandler
from readorsee.data import config
from dotenv import load_dotenv, find_dotenv

PUT_TO_SLEEP = False

original_file = open(config.PATH_TO_EXTERNAL_TWITTER_DATA,
                     "a+", encoding="utf-8")

stopwords = ["vc", ",", ".", "!", "vcs", "tô", "to", "n", "tbm", "tmb", "tá",
             "de", "a", "o", "que", "e", "do", "da", "em", "um", "para", "é",
             "com", "não", "uma", "os", "no", "se", "na", "por", "mais", "as",
             "dos", "como", "mas", "foi", "ao", "ele", "das", "tem", "à",
             "seu", "sua", "ou", "ser", "quando", "muito", "há", "nos", "já",
             "está", "eu", "também", "só", "pelo", "pela", "até", "isso",
             "ela", "entre", "era", "depois", "sem", "mesmo", "aos", "ter",
             "seus", "quem", "nas", "me", "esse", "eles", "estão", "você",
             "tinha", "foram", "essa", "num", "nem", "suas", "meu", "às",
             "minha", "têm", "numa", "pelos", "elas", "havia", "seja", "qual",
             "será", "nós", "tenho", "lhe", "deles", "essas", "esses", "pelas",
             "este", "fosse", "dele", "tu", "te", "vocês", "vos", "lhes",
             "meus", "minhas", "teu", "tua", "teus", "tuas", "nosso", "nossa",
             "nossos", "nossas", "dela", "delas", "esta", "estes", "estas",
             "aquele", "aquela", "aqueles", "aquelas", "isto", "aquilo",
             "estou", "está", "estamos", "estão", "estive", "esteve",
             "estivemos", "estiveram", "estava", "estávamos", "estavam",
             "estivera", "estivéramos", "esteja", "estejamos", "estejam",
             "estivesse", "estivéssemos", "estivessem", "estiver",
             "estivermos", "estiverem", "hei", "há", "havemos", "hão", "houve",
             "houvemos", "houveram", "houvera", "houvéramos", "haja",
             "hajamos", "hajam", "houvesse", "houvéssemos", "houvessem",
             "houver", "houvermos", "houverem", "houverei", "houverá",
             "houveremos", "houverão", "houveria", "houveríamos", "houveriam",
             "sou", "somos", "são", "era", "éramos", "eram", "fui", "foi",
             "fomos", "foram", "fora", "fôramos", "seja", "sejamos", "sejam",
             "fosse", "fôssemos", "fossem", "for", "formos", "forem", "serei",
             "será", "seremos", "serão", "seria", "seríamos", "seriam",
             "tenho", "tem", "temos", "tém", "tinha", "tínhamos", "tinham",
             "tive", "teve", "tivemos", "tiveram", "tivera", "tivéramos",
             "tenha", "tenhamos", "tenham", "tivesse", "tivéssemos",
             "tivessem", "tiver", "tivermos", "tiverem", "terei", "terá",
             "teremos", "terão", "teria", "teríamos", "teriam"]


class StdOutListener(StreamListener):
    """ A listener handles tweets that are received from the stream.
    This is a basic listener that just prints received tweets to stdout.
    """

    def on_data(self, data):
        global PUT_TO_SLEEP
        PUT_TO_SLEEP = False
        json_obj = json.loads(data)
        hashtags = json_obj.get("entities", {}).get("hashtags", [])
        if (not json_obj.get("retweeted", False)
                and not json_obj.get("truncated", False)
                and json_obj.get("retweeted_status", None) is None
                and len(hashtags) == 0):

            text = json_obj.get("text", "").replace(
                '\r', '').replace('\n', '').replace('	', ' ').strip()
            if text:
                original_file.write(text + "\n")
        return True

    def on_error(self, status):
        if status == 420:
            global PUT_TO_SLEEP
            PUT_TO_SLEEP = True


def get_twitter_auth():

    env_variables_path = config.ENV_VARIABLES
    dotenv_path = find_dotenv(env_variables_path)
    load_dotenv(dotenv_path)

    try:
        CONSUMER_KEY = os.environ.get("TWITTER_CONSUMER_KEY")
        CONSUMER_SECRET = os.environ.get("TWITTER_CONSUMER_SECRET")
        ACCESS_TOKEN = os.environ.get("TWITTER_ACCESS_TOKEN")
        ACCESS_SECRET = os.environ.get("TWITTER_ACCESS_SECRET")
        auth = OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
        auth.set_access_token(ACCESS_TOKEN, ACCESS_SECRET)

    except KeyError:
        print("TWITTER_* environment variables not set\n")
        sys.exit(1)

    return auth


if __name__ == '__main__':

    while True:
        stdout = StdOutListener()
        auth = get_twitter_auth()
        print("Collecting tweets...")
        while True:
            try:
                if PUT_TO_SLEEP:
                    print("Sleeping for 10 minutes... \n")
                    time.sleep(600)
                    print("Collecting tweets...")
                stream = Stream(auth, stdout)
                stream.filter(track=stopwords, languages=["pt"])
            except IncompleteRead:
                continue
            except ProtocolError:
                break

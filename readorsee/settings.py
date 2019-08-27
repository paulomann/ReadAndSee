from os.path import dirname, join

ROOT = dirname(dirname(__file__))

PATH_TO_QUESTIONNAIRE = join(ROOT, "data/raw/questionnaire.csv")
PATH_TO_INSTAGRAM_DATA = join(ROOT, "data/external/instagram")
PATH_TO_INTERIM_DATA = join(ROOT, "data/interim")
PATH_TO_PROCESSED_DATA = join(ROOT, "data/processed")
ENV_VARIABLES = join(ROOT, "instagram_access.env")
PATH_TO_CLFS_OPTIONS = join(ROOT, "models/options.json")
PATH_TO_EXPERIMENTS = join(ROOT, "models", "experiments")
PATH_TO_PT_LIWC = join(ROOT, "data", "LIWC2007_Portugues_win.dic")

# ELMO related variables
PATH_TO_EXTERNAL_TWITTER_DATA = join(ROOT,
    "data/external/twitter/tweets_without_hashtag.txt")
PATH_TO_PROCESSED_TO_TRAIN_TWEETS = join(ROOT,
    "data/processed/tweets/train_tweets/")
PATH_TO_PROCESSED_TO_HELDOUT_TWEETS = join(ROOT,
    "data/processed/tweets/heldout_tweets/")
PATH_TO_VOCABULARY_DATA = join(ROOT, "data/processed/tweets/vocabulary.txt")

# Normal elmo:
PATH_TO_ELMO_WEIGHTS = join(ROOT, "models/elmo/elmo_pt_weights.hdf5")
PATH_TO_ELMO_OPTIONS = join(ROOT, "models/elmo/elmo_pt_options.json")
    
# Fine tuned elmo:
PATH_TO_FT_ELMO_WEIGHTS = join(ROOT,
    "models/elmo/fine_tuned_elmo/elmo_pt_fine_tuned_weights.hdf5")
PATH_TO_FT_ELMO_OPTIONS = join(ROOT, 
    "models/elmo/fine_tuned_elmo/elmo_pt_fine_tuned_options.json")

# FastText related variables
PATH_TO_FASTTEXT_PT_EMBEDDINGS = join(ROOT, "models/fasttext/cc.pt.300.bin.gz")

# BoW variables
PATH_TO_SERIALIZED_TFIDF = join(ROOT, "models", "bow", "bow.model")
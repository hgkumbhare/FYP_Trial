import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer

EXAMPLE_TEXT = "Hello Mr. Smith, how are you doing today? The weather is great, and Python is awesome. The sky is pinkish-blue. You shouldn't eat cardboard."

EXAMPLE_TEXT="Such review of risk categorisation of customers should be carried out at a periodicity of not less than once in six months."

print(sent_tokenize(EXAMPLE_TEXT))

print(word_tokenize(EXAMPLE_TEXT))

ps = PorterStemmer();

example_words=["lover","lovable","loving"]
for w in word_tokenize(EXAMPLE_TEXT):
	print(ps.stem(w))

train_text = state_union.raw("2005-GWBush.txt")
#sample_text = state_union.raw("2006-GWBush.txt")
sample_text=EXAMPLE_TEXT;

custom_sent_tokenizer = PunktSentenceTokenizer(train_text);
tokenized = custom_sent_tokenizer.tokenize(sample_text);

def process_content():
    try:
        for i in tokenized[:5]:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            print(tagged)

    except Exception as e:
        print(str(e))


process_content()



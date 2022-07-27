import fasttext
import random

class LanguageIdentification:

    def __init__(self):
        pretrained_lang_model = "lid.176.bin"
        self.model = fasttext.load_model(pretrained_lang_model)

    def predict_lang(self, text):
        predictions = self.model.predict(text, k=2) # returns top 2 matching languages
        return predictions
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer



model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
#model.cuda()

def translate_m2m(text, lang_id , trans_lang):
    tokenizer.src_lang = lang_id  
    encoded_hi = tokenizer(text, return_tensors="pt")
    for key in encoded_hi:
        encoded_hi[key] = encoded_hi[key].cuda()
    generated_tokens = model.generate(**encoded_hi, forced_bos_token_id=tokenizer.get_lang_id(trans_lang))
    tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

if __name__ == "__main__":
    df = pandas.read_csv("bquxjob_1c5b100e_1823e259be4.csv")
    f = open("temp.txt", "w+")
    max_len = 5000
    min_len = 10
    df = df[df['body'].map(len) < max_len ]
    df = df[df['body'].map(len) > min_len ]
    for sent in df.body:
        sent = sent.replace("\n"," ")
        lang_list = ["af" ,"am" ,"ar" ,"ast" ,"az" ,"ba" ,"be" ,"bg" ,"bn" ,"br" ,"bs" ,"ca" ,"ceb" ,"cs" ,"cy" ,"da" ,"de" ,"el" ,"en" ,"es" ,"et" ,"fa" ,"ff" ,"fi" ,"fr" ,"fy" ,"ga" ,"gd" ,"gl" ,"gu" ,"ha" ,"he" ,"hi" ,"hr" ,"ht" ,"hu" ,"hy" ,"id" ,"ig" ,"ilo" ,"is" ,"it" ,"ja" ,"jv" ,"ka" ,"kk" ,"km" ,"kn" ,"ko" ,"lb" ,"lg" ,"ln" ,"lo" ,"lt" ,"lv" ,"mg" ,"mk" ,"ml" ,"mn" ,"mr" ,"ms" ,"my" ,"ne" ,"nl" ,"no" ,"ns" ,"oc" ,"or" ,"pa" ,"pl" ,"ps" ,"pt" ,"ro" ,"ru" ,"sd" ,"si" ,"sk" ,"sl" ,"so" ,"sq" ,"sr" ,"ss" ,"su" ,"sv" ,"sw" ,"ta" ,"th" ,"tl" ,"tn" ,"tr" ,"uk" ,"ur" ,"uz" ,"vi" ,"wo" ,"xh" ,"yi" ,"yo" ,"zh" ,"zu" ]
        trans_lang = (random.choices(lang_list, weights=(5, 5, 20, 5, 5, 5, 5, 5, 20, 5, 5, 5, 5, 20, 5, 100, 20, 5, 5, 5, 5, 20, 5, 5, 5, 5, 5, 5, 5, 20, 5, 5, 5, 5, 5, 5, 5, 20, 20, 5, 5, 5, 5, 20, 20, 5, 5, 5, 5, 5, 5, 5, 5, 20, 5, 5, 5, 5, 5, 5, 5, 5, 5, 20, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 20, 20, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 20, 5), k=1))[0]
        lang = LANGUAGE.predict_lang(sent)
        lang_id = lang[0][0].split("_")[-1]
        if not lang_id == "en":
            sent2 = translate_m2m(sent,lang_id=lang_id,trans_lang=trans_lang)
            text = sent + " || " +sent2 +  '\n'
        else:
            text = sent + '\n'
        f.writelines(text)


        

    
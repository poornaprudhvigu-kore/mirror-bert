import fasttext
import random
import sys,os
import pandas
from multiprocessing import Pool
def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

class LanguageIdentification:

    def __init__(self):
        pretrained_lang_model = "lid.176.bin"
        model = fasttext.load_model(pretrained_lang_model)
        self.model = model

    def predict_lang(self, text):
        text =text
        predictions = self.model.predict(text, k=2) # returns top 2 matching languages
        return predictions
    def predict_lang_df(self,df):
        lang = []
        for text in df.body:
            text = text.replace("\n"," ")
            lang_code = self.model.predict(text, k=1)
            lang_id = lang_code[0][0].split("_")[-1]
            lang.append(lang_id)
        df["lang_id"] = lang
        df.sort_values('lang_id')
        return df
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer



model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
#model.cuda()

def translate_m2m(text, lang_id , trans_lang):
    tokenizer.src_lang = lang_id  
    encoded_hi = tokenizer(text, return_tensors="pt",padding=True)
    # for key in encoded_hi:
    #     encoded_hi[key] = encoded_hi[key].cuda()
    generated_tokens = model.generate(**encoded_hi, forced_bos_token_id=tokenizer.get_lang_id(trans_lang),)
    trans_sent = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    print(trans_sent)
    return trans_sent


if __name__ == "__main__":
    in_file = sys.argv[1]
    df = pandas.read_csv(in_file)
    f = open("temp.txt", "w+")
    LANGUAGE = LanguageIdentification()
    max_len = 3000
    min_len = 10
    df = df[df['body'].notna()]
    df = df[df['body'].map(len) < max_len ]
    df = df[df['body'].map(len) > min_len ]
    df["body"].str.replace("\n"," ")
    df = LANGUAGE.predict_lang_df(df)

    nprocs = 4

    pool = Pool(nprocs)
    for chunk in chunker(df, nprocs):
        chunk_batch = (chunk.body).tolist()
        lang_list = ["af" ,"am" ,"ar"  ,"az"  ,"be" ,"bg" ,"bn"  ,"bs" ,"ca" ,"ceb" ,"cs" ,"cy" ,"da" ,"de" ,"el" ,"en" ,"es" ,"et" ,"fa" ,"ff" ,"fi" ,"fr" ,"fy" ,"ga" ,"gd" ,"gl" ,"gu" ,"ha" ,"he" ,"hi" ,"hr" ,"ht" ,"hu" ,"hy" ,"id" ,"ig"  ,"is" ,"it" ,"ja" ,"jv" ,"ka" ,"kk" ,"km" ,"kn" ,"ko" ,"lb" ,"lg" ,"ln" ,"lo" ,"lt" ,"lv" ,"mg" ,"mk" ,"ml" ,"mn" ,"mr" ,"ms" ,"my" ,"ne" ,"nl" ,"no"  ,"or" ,"pa" ,"pl"  ,"pt" ,"ro" ,"ru"  ,"si" ,"sk" ,"sl" ,"so" ,"sq" ,"sr" ,"su" ,"sv" ,"sw" ,"ta" ,"th" ,"tl"  ,"tr" ,"uk" ,"ur" ,"uz" ,"vi" ,"wo" ,"xh" ,"yi" ,"yo" ,"zh" ,"zu" ]
        trans_lang = (random.choices(lang_list, weights=(5, 5, 20, 5, 5, 5, 5, 5, 20, 5, 5, 5, 5, 20, 5, 100, 20, 5, 5, 5, 5, 20, 5, 5, 5, 5, 5, 5, 5, 20, 5, 5, 5, 5, 5, 5, 5, 20, 20, 5, 5, 5, 5, 20, 20, 5, 5, 5, 5, 5, 5, 5, 5, 20, 5, 5, 5, 5, 5, 5, 5, 5, 5, 20, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 20, 20, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 20, 5), k=1))[0]
        text = []
        total_sum = sum(chunk["body"].str.len())
        print(trans_lang)
        if total_sum > 4000:
            for sent,lang_id in zip(chunk.body,chunk.lang_id):
                if lang_id not in lang_list:
                    text = sent + '\n'
                elif not trans_lang == "en":
                    sent2 = translate_m2m(sent,lang_id=lang_id,trans_lang=trans_lang)
                    text = sent + " || " +sent2 +  '\n'
                else:
                    text = sent + '\n'
                f.writelines(text)

        elif len(df.lang_id.unique()) == 1:
            lang_id = chunk['lang_id']
            if lang_id not in lang_list:
                text = chunk_batch +"\n"
            elif not trans_lang == "en":
                trans_text = translate_m2m(chunk_batch,lang_id=lang_id,trans_lang=trans_lang)
                for sent1,sent2 in zip(chunk_batch,trans_text):
                    text.append(sent1 + " || " +sent2 +'\n')
            else:
                text = chunk_batch+'\n'
        else:
            for lang_id in df.lang_id.unique():
                text_sm = []
                chunk_batch_sm = chunk[chunk['lang_id'].str.contains(lang_id) ]
                if len((chunk_batch_sm.body).tolist()) >=1:
                    if lang_id not in lang_list:
                        text_sm = (chunk_batch_sm.body).tolist()
                    elif not trans_lang == "en":
                        trans_text = translate_m2m((chunk_batch_sm.body).tolist(),lang_id=lang_id,trans_lang=trans_lang)
                        for sent1,sent2 in zip((chunk_batch_sm.body).tolist(),trans_text):
                            text_sm.append(sent1 + " || " +sent2 )
                    else:
                        text_sm = (chunk_batch_sm.body).tolist()
                else:
                    continue
                for line in text_sm:
                    text.append(str(line+'\n'))
        f.write('\n'.join(text))
os.system("sed -i \'/^$/d\' temp.txt")


    # for sent in df.body:
    #     sent = sent.replace("\n"," ")
    #     lang_list = ["af" ,"am" ,"ar"  ,"az"  ,"be" ,"bg" ,"bn"  ,"bs" ,"ca" ,"ceb" ,"cs" ,"cy" ,"da" ,"de" ,"el" ,"en" ,"es" ,"et" ,"fa" ,"ff" ,"fi" ,"fr" ,"fy" ,"ga" ,"gd" ,"gl" ,"gu" ,"ha" ,"he" ,"hi" ,"hr" ,"ht" ,"hu" ,"hy" ,"id" ,"ig"  ,"is" ,"it" ,"ja" ,"jv" ,"ka" ,"kk" ,"km" ,"kn" ,"ko" ,"lb" ,"lg" ,"ln" ,"lo" ,"lt" ,"lv" ,"mg" ,"mk" ,"ml" ,"mn" ,"mr" ,"ms" ,"my" ,"ne" ,"nl" ,"no"  ,"or" ,"pa" ,"pl"  ,"pt" ,"ro" ,"ru"  ,"si" ,"sk" ,"sl" ,"so" ,"sq" ,"sr" ,"su" ,"sv" ,"sw" ,"ta" ,"th" ,"tl"  ,"tr" ,"uk" ,"ur" ,"uz" ,"vi" ,"wo" ,"xh" ,"yi" ,"yo" ,"zh" ,"zu" ]
    #     trans_lang = (random.choices(lang_list, weights=(5, 5, 20, 5, 5, 5, 5, 5, 20, 5, 5, 5, 5, 20, 5, 100, 20, 5, 5, 5, 5, 20, 5, 5, 5, 5, 5, 5, 5, 20, 5, 5, 5, 5, 5, 5, 5, 20, 20, 5, 5, 5, 5, 20, 20, 5, 5, 5, 5, 5, 5, 5, 5, 20, 5, 5, 5, 5, 5, 5, 5, 5, 5, 20, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 20, 20, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 20, 5), k=1))[0]
    #     lang = LANGUAGE.predict_lang(sent)
    #     lang_id = lang[0][0].split("_")[-1]
    #     if lang_id not in lang_list:
    #         text = sent + '\n'
    #     elif not trans_lang == "en":
    #         sent2 = translate_m2m(sent,lang_id=lang_id,trans_lang=trans_lang)
    #         text = sent + " || " +sent2 +  '\n'
    #     else:
    #         text = sent + '\n'
    #     f.writelines(text)


        

    
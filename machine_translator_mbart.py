import pickle
import warnings
warnings.filterwarnings("ignore")
import torch


class Translator:
    def __init__(self):
        self.languages = {"english":"en_XX", "hindi":"hi_IN","gujarati":"gu_IN", "bengali":"bn_IN", 
             "malayalam":"ml_IN", "marathi":"mr_IN", "tamil":"ta_IN", "telugu":"te_IN"}
        
        # self.model = pickle.load(open("/content/drive/MyDrive/IP/machine_trasnlator_api/models/mbart-large-50-many-to-many-mmt-model.sav","rb"))
        self.tokenizer = pickle.load(open("models/mbart-large-50-many-to-many-mmt-tokenizer.sav","rb"))
        
        # self.quantized_model = torch.quantization.quantize_dynamic(self.model,dtype=torch.qint8)
        self.quantized_model = pickle.load(open("models/mbart-large-50-many-to-many-mmt-quantized-model.sav","rb"))
        

    def get_supported_languages(self):
        return list(self.languages.keys())
    
    def translate(self,article, src_lang, target_lang):
        self.tokenizer.src_lang = self.languages[src_lang]
        encoded = self.tokenizer(article, return_tensors="pt")
        translated_tokens = self.quantized_model.generate(**encoded,forced_bos_token_id=self.tokenizer.lang_code_to_id[self.languages[target_lang]])
        translation = self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
        return translation
    
    def translate_original(self,article, src_lang, target_lang):
        self.tokenizer.src_lang = self.languages[src_lang]
        encoded = self.tokenizer(article, return_tensors="pt")
        translated_tokens = self.model.generate(**encoded,forced_bos_token_id=self.tokenizer.lang_code_to_id[self.languages[target_lang]])
        translation = self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
        return translation

"""
translator = Translator()

article="We  should not discriminate with TB patients. Rather patient should be advised to  take certain precaution to avoid spread of disease, he should cover his mouth  and nose while coughing or sneezing, ensure proper sputum disposal and take regular full medical treatment as advised by a medical physician ."
print(translator.get_supported_languages())
print(translator.translate(article, 'english', 'hindi'))
"""

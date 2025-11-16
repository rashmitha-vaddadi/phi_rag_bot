'''using the llm model phi2 and also creating the wrapper class function , this wrapper class function
helps in wrapping all the internal steps to load and run the gemma model'''
##imports
'''importing pytorch as it emables Gemma to run efficiently accross hardware , helps in felxible model loading 
autotokenizer: helps in converting the input data into tokens , automodelcausallm loads the model'''
import torch
from transformers import AutoTokenizer , AutoModelForCausalLM

class SimpleLLM:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ##calling the phi2 model
        self.model_used = "microsoft/phi-2"
        ##intializing autotokenizer , which will convert the chunks into token
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_used,
            use_fast = True
        )
        ##for initialising the model , we are using automodelforcasuallm
        self.model = AutoModelForCasualLM.from_pretrained(
            self.model_used,
            torch_dtype=torch.float32
        ).to(self.device)

    ##we are now creating a new method which will generate text , this will be the code function that
    ##flask api will call later
    def lets_chat(self,prompt):
        ##tokenization , taking text input and converting it into vectors
        encoded_input = self.tokenizer(prompt,return_tensors="pt",truncation=True).to(self.device)
        #return_tensors and truncation are predefined arguments,defined by hugging face inside tokenizer class"
        #return_tensors are saying to give the output in Pytorch tensor format"
        #truncation = true tells the model to cut the input if its much larging than crashing"
        encoded_output = self.model.generate(
             encoded_input,
             max_new_tokens = 150,
             do_sample = True,
             temperature = 0.8
        )

       
        #encoded_output generated the tokenised response to the input query'
        decoded_output = self.tokenizer.decode(encoded_output[0],skip_special_tokens=True)
        #after generation of output the model will return a batch of outputs we need only the first element"
        #and we need to decode the first element in it"
        #we are skipping special tokens as gemma will return tokens like<s> and </s> we need to skip them"

        ##return the decoded output
        return decoded_output




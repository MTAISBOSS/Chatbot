from flask import Flask, render_template, request
from transformers import AutoModelForCausalLM, AutoTokenizer,TextDataset
from transformers import DataCollatorForLanguageModeling,Trainer,TrainingArguments
import torch
from fine_tuner import fine_tune

# Load the tokenizer and model
#'microsoft/DialoGPT-small'
#'microsoft/DialoGPT-medium'
#''facebook/blenderbot-400M-distill'
model_name={'name':'microsoft/DialoGPT-medium'}
tokenizer = AutoTokenizer.from_pretrained(model_name["name"])
model = AutoModelForCausalLM.from_pretrained(model_name["name"])
max_sequence_length = model.config.max_position_embeddings

# Set the device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

#create datase
step={0:-1,}
chat_history_ids={}
#chatbot gui
app = Flask(__name__)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    step[0]+=1
    return get_Chat_response(input,step[0],model_name["name"])


def get_Chat_response(text,step,model_name):

    if model_name=='facebook/blenderbot-400M-distill':

        # encode the new user input, add the eos_token and return a tensor in Pytorch
        inputs= tokenizer([str(text)], return_tensors='pt')
        reply=model.generate(**inputs)
        return tokenizer.batch_decode(reply,skip_special_tokens=True)[0]

    else:


        # encode the new user input, add the eos_token and return a tensor in Pytorch
        new_user_input_ids = tokenizer.encode(str(text) + tokenizer.eos_token, return_tensors='pt')

        # append the new user input tokens to the chat history
        bot_input_ids = torch.cat([chat_history_ids[0], new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids

        # generated a response while limiting the total chat history to 1000 tokens, 
        chat_history_ids[0]= model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

        # pretty print last ouput tokens from bot
        return tokenizer.decode(chat_history_ids[0][:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)



#driver code
if __name__ == '__main__':
    app.run()

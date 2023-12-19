from transformers import AutoModelForCausalLM, AutoTokenizer,TextDataset
from transformers import DataCollatorForLanguageModeling,Trainer,TrainingArguments
import torch
def fine_tune(file_name,model_name):

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    max_sequence_length = model.config.max_position_embeddings

    # Set the device to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    #create dataset
    train_dataset=TextDataset(tokenizer=tokenizer,file_path=file_name,block_size=max_sequence_length)
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer,mlm=False)

    #create training argument
    training_args=TrainingArguments(
        output_dir='./fine-tuned-model',
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        save_steps=500,
        save_total_limit=2,
        prediction_loss_only=True,
        learning_rate=1e-5,
        weight_decay=0.01
        )

    #training model
    trainer=Trainer(model=model,args=training_args,data_collator=data_collator,train_dataset=train_dataset)
    trainer.train()

    #saving model
    model.save_pretrained("fine-tuned-model")
    tokenizer.save_pretrained("fine-tuned-model")
    result={'model':model,'tokenizer':tokenizer}
    return result

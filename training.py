import nltk
import pandas as pd

import datasets
from transformers import pipeline
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import DataLoader

from transformers import DistilBertForSequenceClassification, AdamW

df = pd.read_csv('training_data_sentences.csv')


from sklearn import preprocessing
le = preprocessing.LabelEncoder()
label_list = df['label']
le.fit(label_list)
encoded_label = le.transform(label_list)

df['encoded'] = encoded_label
training_set = df[['encoded', 'sentences']]
training_set.to_csv('encoded_training.csv')

from datasets import load_dataset, load_metric

train_ds, test_ds = datasets.Dataset.from_csv('encoded_training.csv', split=['train[:5000]', 'test[:2000]'])

splits = train_ds.train_test_split(test_size=0.1)

train_ds = splits['train']

val_ds = splits['test']

metric = load_metric("accuracy")
model_name = 'emilyalsentzer/Bio_ClinicalBERT'
model = AutoModelForSequenceClassification.from_pretrained(model_name) # num_labels= len(df['label'].unique()
tokenizer = AutoTokenizer.from_pretrained(model_name)

model.train()

class ViTForImageClassification2(nn.Module):

    def __init__(self, num_labels=10):

        super(ViTForImageClassification2, self).__init__()
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        self.classifier = nn.Linear(self.vit.config.hidden_size, num_labels)
        self.num_labels = num_labels

    def forward(self, pixel_values, labels):

        outputs = self.vit(pixel_values=pixel_values)
        logits = self.classifier(output)
        loss = None

        if labels is not None:

          loss_fct = nn.CrossEntropyLoss()
          loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


args = TrainingArguments(
    "test-cifar-10",
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=10,
    per_device_eval_batch_size=4,
    num_train_epochs=1,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name,
    logging_dir='logs',
)

trainer = Trainer(
    model,
    args,
    train_dataset = preprocessed_train_ds,
    eval_dataset = preprocessed_val_ds,
    data_collator = data_collator,
    compute_metrics = compute_metrics,
)

outputs = trainer.predict(preprocessed_test_ds)

trainer.train()
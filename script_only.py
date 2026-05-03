# **Please do not attempt to run all the codes at once, follow modularity or use the notebook code instead**

# =============================================================================
# SECTION 1: SETUP -- Mount Drive, install dependencies, create folder structure
# =============================================================================

from google.colab import drive
drive.mount('/content/drive')

# !pip install transformers datasets tokenizers evaluate wandb accelerate -q

import os

base = '/content/drive/MyDrive/ReligionBERT'

folders = [
    'datasets/raw/bible', 'datasets/raw/quran', 'datasets/processed',
    'models/baselines/bert-base-uncased', 'models/baselines/mbert',
    'models/baselines/xlm-roberta', 'models/religion-bert',
    'models/multi-religion-bert', 'models/finetuned/semantic-similarity',
    'models/finetuned/classification', 'models/finetuned/qa',
    'checkpoints/pretraining-english', 'checkpoints/pretraining-multilingual',
    'checkpoints/finetuning', 'results/metrics', 'results/figures', 'logs', 'scripts',
]

for folder in folders:
    os.makedirs(f'{base}/{folder}', exist_ok=True)
    print(f'Created: {folder}')


# =============================================================================
# SECTION 2: BIBLE CORPUS -- Download, inspect, extract, and combine
# =============================================================================

# -- 2a: Clone corpus --
import os
os.chdir('/content/drive/MyDrive/ReligionBERT/datasets/raw/bible')
# !git clone https://github.com/christos-c/bible-corpus.git .

# -- 2b: Corpus inspection --
bible_path = '/content/drive/MyDrive/ReligionBERT/datasets/raw/bible/bibles'
files = os.listdir(bible_path)
full_bibles = [f for f in files if '-NT' not in f and '-PART' not in f]
nt_only     = [f for f in files if '-NT' in f]
partial     = [f for f in files if '-PART' in f]
print(f'Full: {len(full_bibles)} | NT only: {len(nt_only)} | Partial: {len(partial)}')

# -- 2c: XML structure inspection --
import xml.etree.ElementTree as ET

tree = ET.parse(f'{bible_path}/English.xml')
root = tree.getroot()
body = root[1][0]
first_book    = body[0]
first_chapter = first_book[0]
for verse in first_chapter[:5]:
    print(f'{verse.tag} | {verse.attrib} | {verse.text}')

# -- 2d: Extract verses from all target languages --
import json

processed_path = '/content/drive/MyDrive/ReligionBERT/datasets/processed'

def extract_bible(filepath):
    try:
        tree = ET.parse(filepath)
        root = tree.getroot()
        body = root[1][0]
        verses = []
        for book in body:
            for chapter in book:
                for verse in chapter:
                    if verse.attrib.get('type') == 'verse' and verse.text:
                        text = verse.text.strip()
                        if text:
                            verses.append({'id': verse.attrib.get('id', ''), 'text': text})
        return verses
    except Exception as e:
        print(f'Error: {e}')
        return []

pretrain_languages = [
    'English.xml', 'English-WEB.xml', 'French.xml', 'Spanish.xml',
    'Portuguese.xml', 'German.xml', 'Amharic.xml', 'Shona.xml',
    'Xhosa.xml', 'Malagasy.xml', 'Somali.xml', 'Zarma.xml',
]
eval_languages = ['Ewe-NT.xml', 'Swahili-NT.xml']

for filename in pretrain_languages + eval_languages:
    filepath = os.path.join(bible_path, filename)
    if not os.path.exists(filepath):
        print(f'MISSING: {filename}')
        continue
    lang   = filename.replace('.xml', '')
    verses = extract_bible(filepath)
    with open(os.path.join(processed_path, f'{lang}.json'), 'w', encoding='utf-8') as f:
        json.dump(verses, f, ensure_ascii=False, indent=2)
    print(f'{lang}: {len(verses):,} verses')

# -- 2e: Combine into flat .txt training files --
import csv

english_files = ['English.json', 'English-WEB.json']
with open(f'{processed_path}/train_english.txt', 'w', encoding='utf-8') as out:
    for filename in english_files:
        with open(os.path.join(processed_path, filename), 'r', encoding='utf-8') as f:
            for verse in json.load(f):
                out.write(verse['text'] + '\n')

pretrain_files = [
    'English.json', 'English-WEB.json', 'French.json', 'Spanish.json',
    'Portuguese.json', 'German.json', 'Amharic.json', 'Shona.json',
    'Xhosa.json', 'Malagasy.json', 'Somali.json', 'Zarma.json',
]
with open(f'{processed_path}/train_multilingual.txt', 'w', encoding='utf-8') as out:
    for filename in pretrain_files:
        with open(os.path.join(processed_path, filename), 'r', encoding='utf-8') as f:
            for verse in json.load(f):
                out.write(verse['text'] + '\n')

for filename in ['Ewe-NT.json', 'Swahili-NT.json']:
    lang = filename.replace('.json', '')
    with open(os.path.join(processed_path, filename), 'r', encoding='utf-8') as f:
        verses = json.load(f)
    with open(f'{processed_path}/eval_{lang}.txt', 'w', encoding='utf-8') as out:
        for verse in verses:
            out.write(verse['text'] + '\n')


# =============================================================================
# SECTION 3: TOKENIZATION
# =============================================================================

from transformers import BertTokenizer
from datasets import Dataset, load_from_disk
import shutil

base          = '/content/drive/MyDrive/ReligionBERT'
processed_path = f'{base}/datasets/processed'

def tokenize_file(txt_path, tokenizer, max_length=128):
    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    print(f'  Lines loaded: {len(lines):,}')
    return tokenizer(lines, truncation=True, max_length=max_length,
                     padding='max_length', return_special_tokens_mask=True)

# -- English tokenization --
bert_tokenizer = BertTokenizer.from_pretrained(f'{base}/models/baselines/bert-base-uncased')
eng_tokenized  = tokenize_file(f'{processed_path}/train_english.txt', bert_tokenizer)
eng_dataset    = Dataset.from_dict(eng_tokenized)
eng_dataset.save_to_disk(f'{processed_path}/tokenized_english')
print('English tokenized and saved')

# -- Multilingual tokenization in 50K-line chunks (avoids OOM) --
mbert_tokenizer = BertTokenizer.from_pretrained(f'{base}/models/baselines/mbert')

with open(f'{processed_path}/train_multilingual.txt', 'r', encoding='utf-8') as f:
    lines = [line.strip() for line in f if line.strip()]

chunks_dir = f'{processed_path}/tokenized_multilingual_chunks'
os.makedirs(chunks_dir, exist_ok=True)
chunk_size = 50000

for i in range(0, len(lines), chunk_size):
    chunk     = lines[i:i + chunk_size]
    chunk_num = i // chunk_size
    tokenized = mbert_tokenizer(chunk, truncation=True, max_length=128,
                                padding='max_length', return_special_tokens_mask=True)
    dataset   = Dataset.from_dict(tokenized)
    dataset.save_to_disk(os.path.join(chunks_dir, f'chunk_{chunk_num}'))
    print(f'Saved chunk_{chunk_num}')
    del tokenized, dataset, chunk


# =============================================================================
# SECTION 4: PRE-TRAINING (Objective 1 -- Domain-Adaptive MLM)
# =============================================================================

# -- 4a: Download baseline models --
from transformers import BertForMaskedLM, AutoTokenizer, AutoModelForMaskedLM

for name, hf_id in [
    ('bert-base-uncased', 'bert-base-uncased'),
    ('mbert',             'bert-base-multilingual-cased'),
]:
    tok = BertTokenizer.from_pretrained(hf_id)
    mdl = BertForMaskedLM.from_pretrained(hf_id)
    tok.save_pretrained(f'{base}/models/baselines/{name}')
    mdl.save_pretrained(f'{base}/models/baselines/{name}')
    del mdl
    print(f'{name} saved')

tok_x = AutoTokenizer.from_pretrained('xlm-roberta-base')
mdl_x = AutoModelForMaskedLM.from_pretrained('xlm-roberta-base')
tok_x.save_pretrained(f'{base}/models/baselines/xlm-roberta')
mdl_x.save_pretrained(f'{base}/models/baselines/xlm-roberta')
del mdl_x
print('xlm-roberta saved')

# -- 4b: Shared callback -- hard-deletes old checkpoints to prevent Drive overflow --
from transformers import TrainerCallback

class PermanentDeleteCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        checkpoints = sorted([
            d for d in os.listdir(args.output_dir) if d.startswith('checkpoint-')
        ])
        if len(checkpoints) > 2:
            shutil.rmtree(os.path.join(args.output_dir, checkpoints[0]))
            print(f'Permanently deleted {checkpoints[0]}')

# -- 4c: ReligionBERT pre-training (English) --
from transformers import DataCollatorForLanguageModeling, TrainingArguments, Trainer
import wandb

tokenizer = BertTokenizer.from_pretrained(f'{base}/models/baselines/bert-base-uncased')
model     = BertForMaskedLM.from_pretrained(f'{base}/models/baselines/bert-base-uncased')

dataset = load_from_disk(f'{base}/datasets/processed/tokenized_english')
dataset = dataset.remove_columns(['token_type_ids'])
dataset = dataset.train_test_split(test_size=0.05, seed=42)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

training_args = TrainingArguments(
    output_dir=f'{base}/checkpoints/pretraining-english',
    max_steps=30000,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=2,
    learning_rate=3e-5,
    warmup_steps=500,
    weight_decay=0.01,
    fp16=True,
    logging_steps=100,
    eval_steps=500,
    save_steps=500,
    save_total_limit=2,
    eval_strategy='steps',
    metric_for_best_model='eval_loss',
    load_best_model_at_end=True,
    report_to='wandb',
    run_name='religion-bert-english-v2',
)

os.environ["WANDB_RUN_ID"] = "vpq5fxqf"
os.environ["WANDB_RESUME"] = "must"
wandb.login()

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    data_collator=data_collator,
    processing_class=tokenizer,
    callbacks=[PermanentDeleteCallback()],
)

trainer.train(resume_from_checkpoint=True)

model.save_pretrained(f'{base}/models/religion-bert')
tokenizer.save_pretrained(f'{base}/models/religion-bert')
model.push_to_hub("LucasLicht/religion-bert")
tokenizer.push_to_hub("LucasLicht/religion-bert")
print('ReligionBERT saved and pushed')

# -- 4d: Loss curve callback for multilingual run --
import matplotlib.pyplot as plt

class LossCurveCallback(TrainerCallback):
    def __init__(self):
        self.train_losses = []
        self.eval_losses  = []
        self.steps        = []
        self.eval_steps   = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and 'loss' in logs:
            self.train_losses.append(logs['loss'])
            self.steps.append(state.global_step)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics and 'eval_loss' in metrics:
            self.eval_losses.append(metrics['eval_loss'])
            self.eval_steps.append(state.global_step)
            self._save_plot()

    def _save_plot(self):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(self.steps, self.train_losses, label='Training Loss', color='royalblue', linewidth=1.5)
        ax.plot(self.eval_steps, self.eval_losses, label='Validation Loss', color='tomato',
                linewidth=2, marker='o', markersize=4)
        ax.set_xlabel('Steps')
        ax.set_ylabel('Loss')
        ax.set_title('MultiReligionBERT -- Pre-training Loss Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{base}/results/figures/multilingual_pretraining_loss.png', dpi=150)
        plt.close()

# -- 4e: MultiReligionBERT pre-training (multilingual) --
from datasets import concatenate_datasets

mbert_tokenizer = BertTokenizer.from_pretrained(f'{base}/models/baselines/mbert')
mbert_model     = BertForMaskedLM.from_pretrained(f'{base}/models/baselines/mbert')

chunks = [
    load_from_disk(os.path.join(chunks_dir, name))
    for name in sorted(os.listdir(chunks_dir))
]
full_dataset = concatenate_datasets(chunks)
full_dataset = full_dataset.remove_columns(['token_type_ids'])
dataset_m    = full_dataset.train_test_split(test_size=0.02, seed=42)

data_collator_m = DataCollatorForLanguageModeling(tokenizer=mbert_tokenizer, mlm=True, mlm_probability=0.15)

training_args_m = TrainingArguments(
    output_dir=f'{base}/checkpoints/pretraining-multilingual',
    max_steps=30000,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=2,
    learning_rate=3e-5,
    warmup_steps=500,
    weight_decay=0.01,
    fp16=True,
    logging_steps=100,
    eval_steps=500,
    save_steps=500,
    save_total_limit=3,
    eval_strategy='steps',
    metric_for_best_model='eval_loss',
    load_best_model_at_end=True,
    report_to='wandb',
    run_name='religion-bert-multilingual-v2',
)

wandb.login()
loss_callback = LossCurveCallback()

trainer_m = Trainer(
    model=mbert_model,
    args=training_args_m,
    train_dataset=dataset_m['train'],
    eval_dataset=dataset_m['test'],
    data_collator=data_collator_m,
    processing_class=mbert_tokenizer,
    callbacks=[PermanentDeleteCallback(), loss_callback],
)

checkpoint_dir = f'{base}/checkpoints/pretraining-multilingual'
checkpoints    = sorted([d for d in os.listdir(checkpoint_dir) if d.startswith('checkpoint-')])
if checkpoints:
    trainer_m.train(resume_from_checkpoint=True)
else:
    trainer_m.train()

mbert_model.save_pretrained(f'{base}/models/multi-religion-bert')
mbert_tokenizer.save_pretrained(f'{base}/models/multi-religion-bert')
mbert_model.push_to_hub("Licht005/multi-religion-bert")
mbert_tokenizer.push_to_hub("Licht005/multi-religion-bert")
loss_callback._save_plot()
print('MultiReligionBERT saved and pushed')


# =============================================================================
# SECTION 5: DATASET CONSTRUCTION (Objective 2 -- Fine-tuning datasets)
# =============================================================================

# Corrected book code mapping -- corpus uses non-standard codes (e.g. MAR not MRK)
book_names = {
    'GEN':'Genesis','EXO':'Exodus','LEV':'Leviticus','NUM':'Numbers',
    'DEU':'Deuteronomy','JOS':'Joshua','JDG':'Judges','RUT':'Ruth',
    '1SA':'1 Samuel','2SA':'2 Samuel','1KI':'1 Kings','2KI':'2 Kings',
    '1CH':'1 Chronicles','2CH':'2 Chronicles','EZR':'Ezra','NEH':'Nehemiah',
    'EST':'Esther','JOB':'Job','PSA':'Psalms','PRO':'Proverbs',
    'ECC':'Ecclesiastes','SON':'Song of Solomon','ISA':'Isaiah',
    'JER':'Jeremiah','LAM':'Lamentations','EZE':'Ezekiel','DAN':'Daniel',
    'HOS':'Hosea','JOE':'Joel','AMO':'Amos','OBA':'Obadiah','JON':'Jonah',
    'MIC':'Micah','NAH':'Nahum','HAB':'Habakkuk','ZEP':'Zephaniah',
    'HAG':'Haggai','ZEC':'Zechariah','MAL':'Malachi','MAT':'Matthew',
    'MAR':'Mark','LUK':'Luke','JOH':'John','ACT':'Acts','ROM':'Romans',
    '1CO':'1 Corinthians','2CO':'2 Corinthians','GAL':'Galatians',
    'EPH':'Ephesians','PHI':'Philippians','COL':'Colossians',
    '1TH':'1 Thessalonians','2TH':'2 Thessalonians','1TI':'1 Timothy',
    '2TI':'2 Timothy','TIT':'Titus','PHM':'Philemon','HEB':'Hebrews',
    'JAM':'James','1PE':'1 Peter','2PE':'2 Peter','1JO':'1 John',
    '2JO':'2 John','3JO':'3 John','JUD':'Jude','REV':'Revelation'
}

OT_BOOKS = {
    'GEN','EXO','LEV','NUM','DEU','JOS','JDG','RUT','1SA','2SA',
    '1KI','2KI','1CH','2CH','EZR','NEH','EST','JOB','PSA','PRO',
    'ECC','SON','ISA','JER','LAM','EZE','DAN','HOS','JOE','AMO',
    'OBA','JON','MIC','NAH','HAB','ZEP','HAG','ZEC','MAL'
}

import random
from collections import defaultdict, Counter

random.seed(42)

with open(f'{processed_path}/English.json', 'r', encoding='utf-8') as f:
    english_kjv = json.load(f)

kjv = {v['id']: v['text'] for v in english_kjv}

# Group verse IDs by book
by_book = defaultdict(list)
for vid in kjv:
    parts = vid.split('.')
    if len(parts) >= 3 and parts[1] in book_names:
        by_book[parts[1]].append(vid)

# -- 5a: Semantic similarity dataset --
sim_dir = f'{base}/datasets/finetuning/semantic_similarity'
os.makedirs(sim_dir, exist_ok=True)

with open(f'{processed_path}/English-WEB.json', 'r', encoding='utf-8') as f:
    web = {v['id']: v['text'] for v in json.load(f)}
with open(f'{processed_path}/French.json', 'r', encoding='utf-8') as f:
    fra = {v['id']: v['text'] for v in json.load(f)}
with open(f'{processed_path}/Spanish.json', 'r', encoding='utf-8') as f:
    spa = {v['id']: v['text'] for v in json.load(f)}
with open(f'{processed_path}/German.json', 'r', encoding='utf-8') as f:
    deu = {v['id']: v['text'] for v in json.load(f)}

common_ids = list(set(kjv) & set(web) & set(fra) & set(spa) & set(deu))
similarity_pairs = []

# High -- same verse, different translations
translation_pairs = [(kjv, web,'KJV-WEB'),(kjv, fra,'KJV-FRA'),(kjv, spa,'KJV-SPA'),(kjv, deu,'KJV-DEU'),(web, fra,'WEB-FRA')]
for vid in random.sample(common_ids, min(3000, len(common_ids))):
    for t1, t2, pname in translation_pairs:
        if vid in t1 and vid in t2:
            similarity_pairs.append({'sentence1':t1[vid],'sentence2':t2[vid],'score':round(random.uniform(0.85,1.0),2),'label':'high','verse_id':vid,'pair_type':pname})

# Medium -- same book, different chapters
for book in by_book:
    ids = by_book[book]
    if len(ids) < 20: continue
    sampled = random.sample(ids, min(60, len(ids)))
    for i in range(0, len(sampled)-1, 2):
        v1, v2 = sampled[i], sampled[i+1]
        if v1 != v2 and v1 in kjv and v2 in kjv:
            similarity_pairs.append({'sentence1':kjv[v1],'sentence2':kjv[v2],'score':round(random.uniform(0.40,0.70),2),'label':'medium','verse_id':f'{v1}|{v2}','pair_type':'same-book'})

# Low -- cross-testament OT vs NT
ot_ids = [v for b in ['GEN','PSA','ISA','JER','EZE'] for v in by_book.get(b,[])]
nt_ids = [v for b in ['MAT','JOH','ROM','REV'] for v in by_book.get(b,[])]
for _ in range(4000):
    v1, v2 = random.choice(ot_ids), random.choice(nt_ids)
    if v1 in kjv and v2 in kjv:
        similarity_pairs.append({'sentence1':kjv[v1],'sentence2':kjv[v2],'score':round(random.uniform(0.0,0.25),2),'label':'low','verse_id':f'{v1}|{v2}','pair_type':'cross-testament'})

# Hard negatives -- same chapter, different verses
for book in list(by_book.keys())[:20]:
    by_ch = defaultdict(list)
    for vid in by_book[book]:
        parts = vid.split('.')
        if len(parts) >= 3:
            by_ch[parts[2]].append(vid)
    for ch_ids in by_ch.values():
        if len(ch_ids) < 4: continue
        sampled = random.sample(ch_ids, min(4, len(ch_ids)))
        for i in range(0, len(sampled)-1, 2):
            v1, v2 = sampled[i], sampled[i+1]
            if v1 != v2 and v1 in kjv and v2 in kjv:
                similarity_pairs.append({'sentence1':kjv[v1],'sentence2':kjv[v2],'score':round(random.uniform(0.20,0.40),2),'label':'hard_negative','verse_id':f'{v1}|{v2}','pair_type':'same-chapter'})

random.shuffle(similarity_pairs)
n = len(similarity_pairs)
sim_train = similarity_pairs[:int(n*0.8)]
sim_val   = similarity_pairs[int(n*0.8):int(n*0.9)]
sim_test  = similarity_pairs[int(n*0.9):]

for split, data in [('train',sim_train),('val',sim_val),('test',sim_test)]:
    with open(f'{sim_dir}/{split}.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# Balance training set by downsampling high/low to match medium/hard count
high   = [p for p in sim_train if p['label'] == 'high']
medium = [p for p in sim_train if p['label'] == 'medium']
hard   = [p for p in sim_train if p['label'] == 'hard_negative']
low    = [p for p in sim_train if p['label'] == 'low']
target = max(len(medium), len(hard), 2000)
balanced = random.sample(high, min(target,len(high))) + medium + hard + random.sample(low, min(target,len(low)))
random.shuffle(balanced)
with open(f'{sim_dir}/train_balanced.json', 'w') as f:
    json.dump(balanced, f, ensure_ascii=False, indent=2)

# -- 5b: Book classification dataset (150 verses per book, balanced) --
cls_dir = f'{base}/datasets/finetuning/classification'
os.makedirs(cls_dir, exist_ok=True)

classification_samples = []
for book_code, book_name in book_names.items():
    ids = by_book.get(book_code, [])
    if not ids: continue
    for vid in random.sample(ids, min(150, len(ids))):
        if vid in kjv:
            classification_samples.append({
                'verse_id': vid, 'text': kjv[vid],
                'book_code': book_code, 'book_name': book_name,
                'testament': 'Old Testament' if book_code in OT_BOOKS else 'New Testament'
            })

random.shuffle(classification_samples)
n   = len(classification_samples)
cls_train = classification_samples[:int(n*0.8)]
cls_val   = classification_samples[int(n*0.8):int(n*0.9)]
cls_test  = classification_samples[int(n*0.9):]

for split, data in [('train',cls_train),('val',cls_val),('test',cls_test)]:
    with open(f'{cls_dir}/{split}.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

books_with_data = sorted(set(s['book_name'] for s in classification_samples))
label_map = {name: i for i, name in enumerate(books_with_data)}
with open(f'{cls_dir}/label_map_filtered.json', 'w') as f:
    json.dump(label_map, f, indent=2)

# -- 5c: QA dataset via Groq API (Llama 3.3-70B, SQuAD-format extractive QA) --
# !pip install groq -q
from groq import Groq
import time

GROQ_API_KEY = 'API HERE'  # replace with your key
client = Groq(api_key=GROQ_API_KEY)
MODEL  = 'llama-3.3-70b-versatile'

qa_dir = f'{base}/datasets/finetuning/qa'
os.makedirs(qa_dir, exist_ok=True)

def is_suitable(text):
    text = text.strip()
    if len(text) < 60 or len(text) > 300: return False
    words = text.split()
    return sum(1 for w in words[1:] if w and w[0].isupper() and w.isalpha()) >= 2

ot_verses = [{'verse_id':v['id'],'text':v['text'].strip(),'book_code':v['id'].split('.')[1],
               'book_name':book_names.get(v['id'].split('.')[1],''),'testament':'Old Testament'}
             for v in english_kjv if v['id'].split('.')[1] in OT_BOOKS and
             v['id'].split('.')[1] in book_names and is_suitable(v['text'])]

nt_verses = [{'verse_id':v['id'],'text':v['text'].strip(),'book_code':v['id'].split('.')[1],
               'book_name':book_names.get(v['id'].split('.')[1],''),'testament':'New Testament'}
             for v in english_kjv if v['id'].split('.')[1] not in OT_BOOKS and
             v['id'].split('.')[1] in book_names and is_suitable(v['text'])]

all_verses = random.sample(ot_verses, min(600, len(ot_verses))) + random.sample(nt_verses, min(600, len(nt_verses)))
random.shuffle(all_verses)

SYSTEM_PROMPT = """You are an expert annotator creating a question answering dataset from Bible verses.
Generate one factual question-answer pair. The answer MUST be a direct, continuous substring of the verse.
Return ONLY valid JSON with keys: context, question, answer"""

USER_TEMPLATE = 'Verse: "{verse}"\n\nGenerate one question-answer pair where the answer is a direct substring of the verse.'

# Resume logic
if os.path.exists(f'{qa_dir}/qa_checkpoint.json'):
    with open(f'{qa_dir}/qa_checkpoint.json') as f:
        qa_dataset = json.load(f)
    start_index = len(qa_dataset)
else:
    qa_dataset  = []
    start_index = 0

failed = []
for i in range(start_index, len(all_verses)):
    verse_entry = all_verses[i]
    verse_text  = verse_entry['text']
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{'role':'system','content':SYSTEM_PROMPT},
                      {'role':'user','content':USER_TEMPLATE.format(verse=verse_text)}],
            temperature=0.3, max_tokens=200, response_format={'type':'json_object'}
        )
        parsed = json.loads(response.choices[0].message.content.strip())
        question, answer = parsed.get('question','').strip(), parsed.get('answer','').strip()
        answer_start = verse_text.find(answer)
        if answer_start == -1:
            lower = verse_text.lower().find(answer.lower())
            if lower != -1:
                answer       = verse_text[lower:lower+len(answer)]
                answer_start = lower
            else:
                failed.append({'verse_id':verse_entry['verse_id'],'reason':'not substring'})
                continue
        qa_dataset.append({
            'id': f'religionbert-qa-{i:04d}', 'verse_id': verse_entry['verse_id'],
            'book_name': verse_entry['book_name'], 'testament': verse_entry['testament'],
            'context': verse_text, 'question': question, 'answer': answer,
            'answer_start': answer_start, 'answers': {'text':[answer],'answer_start':[answer_start]}
        })
        if len(qa_dataset) % 50 == 0:
            with open(f'{qa_dir}/qa_checkpoint.json', 'w') as f:
                json.dump(qa_dataset, f, ensure_ascii=False, indent=2)
        time.sleep(3.0)  # respect TPD rate limit
    except Exception as e:
        print(f'Error at {i}: {e}')
        time.sleep(10)

random.shuffle(qa_dataset)
n = len(qa_dataset)
for split, data in [('train',qa_dataset[:int(n*0.8)]),('val',qa_dataset[int(n*0.8):int(n*0.9)]),('test',qa_dataset[int(n*0.9):])]:
    with open(f'{qa_dir}/{split}.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

print(f'QA dataset: {n} total examples')


# =============================================================================
# SECTION 6: FINE-TUNING -- Semantic Similarity
# =============================================================================
# Two-stage: Stage 1 balanced (5 epochs, LR 2e-5), Stage 2 full (2 epochs, LR 1e-5)
# Set MODEL_PATH and RUN_NAME per variant before running

from transformers import BertForSequenceClassification
from scipy.stats import pearsonr, spearmanr
import numpy as np

sim_dir = f'{base}/datasets/finetuning/semantic_similarity'

def load_sim_dataset(path, tokenizer, max_length=128):
    with open(path) as f: data = json.load(f)
    enc = tokenizer([d['sentence1'] for d in data], [d['sentence2'] for d in data],
                    truncation=True, max_length=max_length, padding='max_length', return_tensors=None)
    enc['labels'] = [float(d['score']) for d in data]
    return Dataset.from_dict(enc)

def compute_sim_metrics(eval_pred):
    preds, labels = eval_pred
    preds = preds.flatten()
    return {'pearson': pearsonr(preds, labels)[0], 'spearman': spearmanr(preds, labels)[0]}

# Variants: bert-base-uncased, religion-bert, mbert, multi-religion-bert
MODEL_PATH = f'{base}/models/religion-bert'  # CHANGE PER VARIANT
RUN_NAME   = 'religion-bert-sim'             # CHANGE PER VARIANT
OUTPUT_DIR = f'{base}/checkpoints/finetuning/sim_{RUN_NAME}'

tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model     = BertForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=1)

train_ds = load_sim_dataset(f'{sim_dir}/train_balanced.json', tokenizer)
val_ds   = load_sim_dataset(f'{sim_dir}/val.json', tokenizer)
test_ds  = load_sim_dataset(f'{sim_dir}/test.json', tokenizer)

for stage, lr, epochs, train_data, out_dir in [
    (1, 2e-5, 5, train_ds, OUTPUT_DIR),
    (2, 1e-5, 2, load_sim_dataset(f'{sim_dir}/train.json', tokenizer), f'{OUTPUT_DIR}_stage2'),
]:
    args = TrainingArguments(
        output_dir=out_dir, num_train_epochs=epochs,
        per_device_train_batch_size=32, per_device_eval_batch_size=32,
        learning_rate=lr, warmup_ratio=0.1, weight_decay=0.01, fp16=True,
        eval_strategy='epoch', save_strategy='epoch', save_total_limit=2,
        load_best_model_at_end=True, metric_for_best_model='pearson', greater_is_better=True,
        report_to='wandb', run_name=f'{RUN_NAME}-stage{stage}',
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_data, eval_dataset=val_ds,
                      compute_metrics=compute_sim_metrics, processing_class=tokenizer,
                      callbacks=[PermanentDeleteCallback()])
    trainer.train()

results = trainer.evaluate(test_ds)
print(f'Test Pearson: {results["eval_pearson"]:.4f} | Spearman: {results["eval_spearman"]:.4f}')

with open(f'{base}/results/metrics/{RUN_NAME}_sim_results.json', 'w') as f:
    json.dump(results, f, indent=2)
model.save_pretrained(f'{base}/models/finetuned/{RUN_NAME}_similarity')
tokenizer.save_pretrained(f'{base}/models/finetuned/{RUN_NAME}_similarity')


# =============================================================================
# SECTION 7: FINE-TUNING -- Book Classification (66-class)
# =============================================================================
# 10 epochs, LR 2e-5, batch 32, optimized on macro F1
# Set MODEL_PATH and RUN_NAME per variant before running

import evaluate as hf_evaluate

cls_dir = f'{base}/datasets/finetuning/classification'

with open(f'{cls_dir}/label_map_filtered.json') as f:
    label_map = json.load(f)
id2label   = {v: k for k, v in label_map.items()}
num_labels = len(label_map)

def load_cls_dataset(path, tokenizer, max_length=128):
    with open(path) as f: data = json.load(f)
    enc = tokenizer([d['text'] for d in data], truncation=True, max_length=max_length,
                    padding='max_length', return_tensors=None)
    enc['labels'] = [label_map[d['book_name']] for d in data]
    return Dataset.from_dict(enc)

acc_metric = hf_evaluate.load('accuracy')
f1_metric  = hf_evaluate.load('f1')

def compute_cls_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {'accuracy': acc_metric.compute(predictions=preds, references=labels)['accuracy'],
            'macro_f1': f1_metric.compute(predictions=preds, references=labels, average='macro')['f1']}

# Variants: bert-base-uncased, religion-bert, mbert, multi-religion-bert
MODEL_PATH = f'{base}/models/religion-bert'              # CHANGE PER VARIANT
RUN_NAME   = 'religion-bert-classification'              # CHANGE PER VARIANT
OUTPUT_DIR = f'{base}/checkpoints/finetuning/cls_{RUN_NAME}'

tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model     = BertForSequenceClassification.from_pretrained(
    MODEL_PATH, num_labels=num_labels, id2label=id2label,
    label2id=label_map, ignore_mismatched_sizes=True
)

train_ds = load_cls_dataset(f'{cls_dir}/train.json', tokenizer)
val_ds   = load_cls_dataset(f'{cls_dir}/val.json',   tokenizer)
test_ds  = load_cls_dataset(f'{cls_dir}/test.json',  tokenizer)

args = TrainingArguments(
    output_dir=OUTPUT_DIR, num_train_epochs=10,
    per_device_train_batch_size=32, per_device_eval_batch_size=32,
    learning_rate=2e-5, warmup_ratio=0.1, weight_decay=0.01, fp16=True,
    eval_strategy='epoch', save_strategy='epoch', save_total_limit=2,
    load_best_model_at_end=True, metric_for_best_model='macro_f1', greater_is_better=True,
    report_to='wandb', run_name=RUN_NAME,
)

trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
                  compute_metrics=compute_cls_metrics, processing_class=tokenizer,
                  callbacks=[PermanentDeleteCallback()])
trainer.train()

results = trainer.evaluate(test_ds)
print(f'Test Accuracy: {results["eval_accuracy"]:.4f} | Macro F1: {results["eval_macro_f1"]:.4f}')

with open(f'{base}/results/metrics/{RUN_NAME}_cls_results.json', 'w') as f:
    json.dump(results, f, indent=2)
model.save_pretrained(f'{base}/models/finetuned/classification/{RUN_NAME}_classification')
tokenizer.save_pretrained(f'{base}/models/finetuned/classification/{RUN_NAME}_classification')


# =============================================================================
# SECTION 8: FINE-TUNING -- Extractive QA (BertForQuestionAnswering)
# =============================================================================
# Uses BertTokenizerFast for offset mapping. MAX_LENGTH=384, STRIDE=128
# Set MODEL_PATH and RUN_NAME per variant before running

import torch
import collections
from transformers import BertForQuestionAnswering, BertTokenizerFast, DefaultDataCollator

qa_dir = f'{base}/datasets/finetuning/qa'

MAX_LENGTH = 384
STRIDE     = 128

def load_qa_data(path):
    with open(path) as f: return json.load(f)

def tokenize_qa(data, tokenizer, is_train=True):
    tokenized  = tokenizer(
        [d['question'] for d in data], [d['context'] for d in data],
        truncation='only_second', max_length=MAX_LENGTH, stride=STRIDE,
        return_overflowing_tokens=True, return_offsets_mapping=True, padding='max_length'
    )
    sample_map = tokenized.pop('overflow_to_sample_mapping')
    offset_map = tokenized.pop('offset_mapping')
    tokenized['start_positions'] = []
    tokenized['end_positions']   = []
    tokenized['example_id']      = []

    for i, offsets in enumerate(offset_map):
        example   = data[sample_map[i]]
        tokenized['example_id'].append(example['id'])
        input_ids = tokenized['input_ids'][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)
        seq_ids   = tokenized.sequence_ids(i)
        ctx_start = next(j for j, s in enumerate(seq_ids) if s == 1)
        ctx_end   = next(j for j, s in reversed(list(enumerate(seq_ids))) if s == 1)

        if not is_train:
            tokenized['start_positions'].append(cls_index)
            tokenized['end_positions'].append(cls_index)
            continue

        ans_start_char = example['answer_start']
        ans_end_char   = ans_start_char + len(example['answer'])
        if offsets[ctx_start][0] > ans_end_char or offsets[ctx_end][1] < ans_start_char:
            tokenized['start_positions'].append(cls_index)
            tokenized['end_positions'].append(cls_index)
            continue

        tok_start = ctx_start
        while tok_start <= ctx_end and offsets[tok_start][0] <= ans_start_char:
            tok_start += 1
        tok_start -= 1

        tok_end = ctx_end
        while tok_end >= ctx_start and offsets[tok_end][1] >= ans_end_char:
            tok_end -= 1
        tok_end += 1

        tokenized['start_positions'].append(tok_start)
        tokenized['end_positions'].append(tok_end)

    return tokenized

# Direct span prediction evaluation (EM and token F1)
def simple_evaluate(model, tokenizer, data, split_name):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    exact_matches, f1_scores = [], []
    for item in data:
        inputs = tokenizer(item['question'], item['context'], return_tensors='pt',
                           truncation=True, max_length=384, padding='max_length')
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        start = torch.argmax(outputs.start_logits)
        end   = torch.argmax(outputs.end_logits)
        pred  = tokenizer.decode(inputs['input_ids'][0][start:end+1], skip_special_tokens=True).strip() if end >= start else ''
        gold  = item['answer'].lower().strip()
        pred_l = pred.lower().strip()
        exact_matches.append(int(pred_l == gold))
        g_t, p_t = set(gold.split()), set(pred_l.split())
        if not p_t or not g_t:
            f1_scores.append(1.0 if not p_t and not g_t else 0.0)
        else:
            common = g_t & p_t
            pr, re = len(common)/len(p_t), len(common)/len(g_t)
            f1_scores.append(2*pr*re/(pr+re) if (pr+re) > 0 else 0.0)
    em, f1 = np.mean(exact_matches)*100, np.mean(f1_scores)*100
    print(f'{split_name} -- EM: {em:.2f}% | F1: {f1:.2f}%')
    return {'exact_match': em, 'f1': f1}

# Variants: bert-base-uncased, religion-bert, mbert, multi-religion-bert
MODEL_PATH = f'{base}/models/religion-bert'  # CHANGE PER VARIANT
RUN_NAME   = 'religion-bert-qa'              # CHANGE PER VARIANT
OUTPUT_DIR = f'{base}/checkpoints/finetuning/qa_{RUN_NAME}'

tokenizer = BertTokenizerFast.from_pretrained(MODEL_PATH)  # Must use Fast for offset mapping
model     = BertForQuestionAnswering.from_pretrained(MODEL_PATH)

train_data = load_qa_data(f'{qa_dir}/train.json')
val_data   = load_qa_data(f'{qa_dir}/val.json')
test_data  = load_qa_data(f'{qa_dir}/test.json')

train_tok = tokenize_qa(train_data, tokenizer, is_train=True)
val_tok   = tokenize_qa(val_data,   tokenizer, is_train=False)
test_tok  = tokenize_qa(test_data,  tokenizer, is_train=False)

val_ids  = val_tok.pop('example_id')
test_ids = test_tok.pop('example_id')
train_tok.pop('example_id', None)

train_dataset    = Dataset.from_dict(train_tok)
val_for_trainer  = Dataset.from_dict(val_tok)

args = TrainingArguments(
    output_dir=OUTPUT_DIR, num_train_epochs=8,
    per_device_train_batch_size=16, per_device_eval_batch_size=16,
    learning_rate=1e-5, warmup_steps=150, weight_decay=0.01, fp16=True,
    eval_strategy='epoch', save_strategy='epoch', save_total_limit=2,
    logging_steps=10, report_to='wandb', run_name=RUN_NAME,
)

trainer = Trainer(model=model, args=args, train_dataset=train_dataset,
                  eval_dataset=val_for_trainer, data_collator=DefaultDataCollator(),
                  processing_class=tokenizer, callbacks=[PermanentDeleteCallback()])
trainer.train()

val_results  = simple_evaluate(model, tokenizer, val_data,  'Validation')
test_results = simple_evaluate(model, tokenizer, test_data, 'Test')

with open(f'{base}/results/metrics/{RUN_NAME}_qa_results.json', 'w') as f:
    json.dump({'val': val_results, 'test': test_results}, f, indent=2)
model.save_pretrained(f'{base}/models/finetuned/qa/{RUN_NAME}')
tokenizer.save_pretrained(f'{base}/models/finetuned/qa/{RUN_NAME}')
print(f'Saved: {RUN_NAME}')


# =============================================================================
# SECTION 9: RESULTS VISUALIZATION
# =============================================================================

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

figures_dir = f'{base}/results/figures'
metrics_dir = f'{base}/results/metrics'
os.makedirs(figures_dir, exist_ok=True)

results = {
    'Generic BERT':     {'sim_pearson':0.9569,'sim_spearman':0.6556,'cls_accuracy':0.4075,'cls_f1':0.3144,'qa_em':32.50,'qa_f1':58.91},
    'ReligionBERT':     {'sim_pearson':0.9623,'sim_spearman':0.6591,'cls_accuracy':0.4347,'cls_f1':0.3381,'qa_em':40.00,'qa_f1':60.48},
    'mBERT':            {'sim_pearson':0.9624,'sim_spearman':0.6772,'cls_accuracy':0.3972,'cls_f1':0.2948,'qa_em':48.33,'qa_f1':70.54},
    'MultiReligionBERT':{'sim_pearson':0.9635,'sim_spearman':0.6669,'cls_accuracy':0.4360,'cls_f1':0.3369,'qa_em':40.83,'qa_f1':65.39},
}
models = list(results.keys())
colors = {'Generic BERT':'#4C72B0','ReligionBERT':'#DD8452','mBERT':'#55A868','MultiReligionBERT':'#C44E52'}

plt.rcParams.update({'font.family':'serif','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150})

def bar_chart(ax, metric_key, metric_label, ylim_pad=0.97):
    vals = [results[m][metric_key] for m in models]
    bars = ax.bar(models, vals, color=[colors[m] for m in models], width=0.5, edgecolor='white')
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002, f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    ax.set_title(metric_label, fontsize=11)
    ax.set_ylabel(metric_label)
    ax.set_ylim(min(vals)*ylim_pad, max(vals)*1.03)
    ax.set_xticklabels(models, rotation=15, ha='right', fontsize=9)
    ax.yaxis.grid(True, linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)

# Fig 4.1: Semantic similarity
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Figure 4.1: Semantic Similarity Performance', fontsize=13, fontweight='bold', y=1.01)
bar_chart(axes[0], 'sim_pearson', 'Pearson Correlation')
bar_chart(axes[1], 'sim_spearman', 'Spearman Correlation')
plt.tight_layout()
plt.savefig(f'{figures_dir}/fig4_1_semantic_similarity.png', dpi=150, bbox_inches='tight')
plt.show()

# Fig 4.2: Classification
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Figure 4.2: Book Classification Performance', fontsize=13, fontweight='bold', y=1.01)
for ax, (mk, ml) in zip(axes, [('cls_accuracy','Accuracy'),('cls_f1','Macro F1')]):
    vals = [results[m][mk] for m in models]
    bars = ax.bar(models, vals, color=[colors[m] for m in models], width=0.5, edgecolor='white')
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003, f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    ax.set_title(ml); ax.set_ylabel(ml)
    ax.set_ylim(0, max(vals)*1.15)
    ax.set_xticklabels(models, rotation=15, ha='right', fontsize=9)
    ax.yaxis.grid(True, linestyle='--', alpha=0.5); ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig(f'{figures_dir}/fig4_2_classification.png', dpi=150, bbox_inches='tight')
plt.show()

# Fig 4.3: QA
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Figure 4.3: Question Answering Performance', fontsize=13, fontweight='bold', y=1.01)
for ax, (mk, ml) in zip(axes, [('qa_em','Exact Match (%)'),('qa_f1','Token F1 (%)')]):
    vals = [results[m][mk] for m in models]
    bars = ax.bar(models, vals, color=[colors[m] for m in models], width=0.5, edgecolor='white')
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3, f'{val:.2f}%', ha='center', va='bottom', fontsize=9)
    ax.set_title(ml); ax.set_ylabel(ml)
    ax.set_ylim(0, max(vals)*1.18)
    ax.set_xticklabels(models, rotation=15, ha='right', fontsize=9)
    ax.yaxis.grid(True, linestyle='--', alpha=0.5); ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig(f'{figures_dir}/fig4_3_qa.png', dpi=150, bbox_inches='tight')
plt.show()

# Fig 4.4: All tasks grouped bar (normalized 0-1)
fig, ax = plt.subplots(figsize=(14, 6))
fig.suptitle('Figure 4.4: Model Comparison Across All Tasks', fontsize=13, fontweight='bold')
task_metrics = [('Sim\nPearson','sim_pearson',1.0),('Sim\nSpearman','sim_spearman',1.0),
                ('Cls\nAccuracy','cls_accuracy',1.0),('Cls\nMacro F1','cls_f1',1.0),
                ('QA\nExact Match','qa_em',100.0),('QA\nToken F1','qa_f1',100.0)]
x = np.arange(len(task_metrics)); width = 0.18
for i, model in enumerate(models):
    vals   = [results[model][mk]/scale for _,mk,scale in task_metrics]
    offset = (i - len(models)/2 + 0.5) * width
    ax.bar(x+offset, vals, width, label=model, color=colors[model], edgecolor='white', linewidth=0.5)
ax.set_xticks(x); ax.set_xticklabels([m[0] for m in task_metrics], fontsize=10)
ax.set_ylabel('Score (normalized 0-1)'); ax.set_ylim(0, 1.0)
ax.yaxis.grid(True, linestyle='--', alpha=0.4); ax.set_axisbelow(True)
ax.legend(loc='upper right', framealpha=0.9, fontsize=9)
plt.tight_layout()
plt.savefig(f'{figures_dir}/fig4_4_all_tasks_combined.png', dpi=150, bbox_inches='tight')
plt.show()

# Fig 4.5: Radar chart
categories = ['Sim Pearson','Sim Spearman','Cls Accuracy','Cls Macro F1','QA Exact Match','QA Token F1']
scales     = [1.0, 1.0, 1.0, 1.0, 100.0, 100.0]
N      = len(categories)
angles = [n / float(N) * 2 * np.pi for n in range(N)] + [0]
fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
fig.suptitle('Figure 4.5: Radar Chart -- Model Performance', fontsize=12, fontweight='bold', y=1.02)
ax.set_theta_offset(np.pi/2); ax.set_theta_direction(-1)
ax.set_xticks(angles[:-1]); ax.set_xticklabels(categories, size=10)
ax.set_ylim(0, 1)
for model in models:
    vals = [results[model][k]/s for k, s in zip(['sim_pearson','sim_spearman','cls_accuracy','cls_f1','qa_em','qa_f1'], scales)] + [results[model]['sim_pearson']/scales[0]]
    ax.plot(angles, vals, 'o-', linewidth=2, label=model, color=colors[model])
    ax.fill(angles, vals, alpha=0.08, color=colors[model])
ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), framealpha=0.9, fontsize=10)
plt.tight_layout()
plt.savefig(f'{figures_dir}/fig4_5_radar.png', dpi=150, bbox_inches='tight')
plt.show()

# Fig 4.6: Improvement heatmap
fig, ax = plt.subplots(figsize=(10, 5))
fig.suptitle('Figure 4.6: Domain-adapted Model Improvement over Generic Baselines', fontsize=12, fontweight='bold')
comparisons = {
    'ReligionBERT\nvs Generic BERT': {
        'Sim Pearson':   results['ReligionBERT']['sim_pearson']   - results['Generic BERT']['sim_pearson'],
        'Sim Spearman':  results['ReligionBERT']['sim_spearman']  - results['Generic BERT']['sim_spearman'],
        'Cls Accuracy':  results['ReligionBERT']['cls_accuracy']  - results['Generic BERT']['cls_accuracy'],
        'Cls Macro F1':  results['ReligionBERT']['cls_f1']        - results['Generic BERT']['cls_f1'],
        'QA EM':        (results['ReligionBERT']['qa_em']         - results['Generic BERT']['qa_em'])/100,
        'QA F1':        (results['ReligionBERT']['qa_f1']         - results['Generic BERT']['qa_f1'])/100,
    },
    'MultiReligionBERT\nvs mBERT': {
        'Sim Pearson':   results['MultiReligionBERT']['sim_pearson']   - results['mBERT']['sim_pearson'],
        'Sim Spearman':  results['MultiReligionBERT']['sim_spearman']  - results['mBERT']['sim_spearman'],
        'Cls Accuracy':  results['MultiReligionBERT']['cls_accuracy']  - results['mBERT']['cls_accuracy'],
        'Cls Macro F1':  results['MultiReligionBERT']['cls_f1']        - results['mBERT']['cls_f1'],
        'QA EM':        (results['MultiReligionBERT']['qa_em']         - results['mBERT']['qa_em'])/100,
        'QA F1':        (results['MultiReligionBERT']['qa_f1']         - results['mBERT']['qa_f1'])/100,
    }
}
metric_labels = list(list(comparisons.values())[0].keys())
comp_labels   = list(comparisons.keys())
data_matrix   = np.array([[comparisons[c][m] for m in metric_labels] for c in comp_labels])
im = ax.imshow(data_matrix, cmap='RdYlGn', aspect='auto', vmin=-0.05, vmax=0.10)
plt.colorbar(im, ax=ax, label='Improvement', shrink=0.8)
ax.set_xticks(range(len(metric_labels))); ax.set_xticklabels(metric_labels, fontsize=10, rotation=15, ha='right')
ax.set_yticks(range(len(comp_labels)));  ax.set_yticklabels(comp_labels, fontsize=10)
for i in range(len(comp_labels)):
    for j in range(len(metric_labels)):
        val = data_matrix[i, j]
        ax.text(j, i, f'{"+" if val>=0 else ""}{val:.4f}', ha='center', va='center', fontsize=9)
plt.tight_layout()
plt.savefig(f'{figures_dir}/fig4_6_improvement_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()

# Fig 4.7: Pre-training loss curves
religion_bert_steps = [500,2000,5000,10000,15000,20000,25000,28500,30000]
religion_bert_loss  = [1.806,1.632,1.454,1.339,1.261,1.204,1.159,1.129,1.164]
multi_steps = [18500,19000,20000,21000,22000,23000,24000,25000,26000,27000,27500,28000,28500,29000,30000]
multi_loss  = [1.532,1.540,1.539,1.513,1.475,1.479,1.500,1.448,1.442,1.430,1.402,1.396,1.392,1.391,1.392]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Figure 4.7: Pre-training Validation Loss Curves', fontsize=13, fontweight='bold')
axes[0].plot(religion_bert_steps, religion_bert_loss, 'o-', color=colors['ReligionBERT'], linewidth=2, markersize=5)
axes[0].set_title('ReligionBERT (English Bible)'); axes[0].set_xlabel('Steps'); axes[0].set_ylabel('Validation Loss (MLM)')
axes[0].yaxis.grid(True, linestyle='--', alpha=0.5); axes[0].set_axisbelow(True)
axes[1].plot(multi_steps, multi_loss, 's-', color=colors['MultiReligionBERT'], linewidth=2, markersize=5)
axes[1].set_title('MultiReligionBERT (Multilingual Bible)'); axes[1].set_xlabel('Steps'); axes[1].set_ylabel('Validation Loss (MLM)')
axes[1].yaxis.grid(True, linestyle='--', alpha=0.5); axes[1].set_axisbelow(True)
plt.tight_layout()
plt.savefig(f'{figures_dir}/fig4_7_pretraining_loss_curves.png', dpi=150, bbox_inches='tight')
plt.show()

# Fig 4.8: Perplexity on held-out religious text
import math
from transformers import BertForMaskedLM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with open(f'{base}/datasets/processed/English.json') as f:
    held_out = [v['text'] for v in json.load(f)[-500:]]

def compute_perplexity(model_path, held_out_texts, device, max_length=128):
    tok   = BertTokenizer.from_pretrained(model_path)
    mdl   = BertForMaskedLM.from_pretrained(model_path).to(device)
    mdl.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for text in held_out_texts:
            inputs = tok(text, return_tensors='pt', truncation=True, max_length=max_length, padding='max_length').to(device)
            labels = inputs['input_ids'].clone()
            rand   = torch.rand(labels.shape, device=device)
            mask   = (rand < 0.15) & (labels != tok.pad_token_id) & (labels != tok.cls_token_id) & (labels != tok.sep_token_id)
            labels[~mask] = -100
            inputs['input_ids'][mask] = tok.mask_token_id
            if mask.sum() == 0: continue
            outputs = mdl(**inputs, labels=labels)
            total_loss   += outputs.loss.item() * mask.sum().item()
            total_tokens += mask.sum().item()
    del mdl; torch.cuda.empty_cache()
    return math.exp(total_loss / total_tokens)

perplexity_results = {}
for name, path in [('Generic BERT', f'{base}/models/baselines/bert-base-uncased'),
                   ('ReligionBERT',  f'{base}/models/religion-bert')]:
    ppl = compute_perplexity(path, held_out, device)
    perplexity_results[name] = round(ppl, 3)
    print(f'{name}: PPL = {ppl:.3f}')

with open(f'{metrics_dir}/perplexity_results.json', 'w') as f:
    json.dump(perplexity_results, f, indent=2)

fig, ax = plt.subplots(figsize=(7, 5))
fig.suptitle('Figure 4.8: Perplexity on Held-out Religious Text (lower is better)', fontsize=12, fontweight='bold')
ppl_models = list(perplexity_results.keys())
ppl_vals   = list(perplexity_results.values())
bars = ax.bar(ppl_models, ppl_vals, color=[colors[m] for m in ppl_models], width=0.4, edgecolor='white')
for bar, val in zip(bars, ppl_vals):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3, f'{val:.2f}', ha='center', va='bottom', fontsize=10)
ax.set_ylabel('Perplexity'); ax.set_ylim(0, max(ppl_vals)*1.2)
ax.yaxis.grid(True, linestyle='--', alpha=0.5); ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig(f'{figures_dir}/fig4_8_perplexity.png', dpi=150, bbox_inches='tight')
plt.show()


# =============================================================================
# SECTION 10: ZERO-SHOT CROSS-LINGUAL TRANSFER
# =============================================================================
# Fine-tune multilingual models on English classification, then evaluate on
# African languages (Amharic, Shona, Xhosa, Ewe, Swahili) with no target-language training

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import warnings
warnings.filterwarnings('ignore')

cls_dir     = f'{base}/datasets/finetuning/classification'
figures_dir = f'{base}/results/figures'
metrics_dir = f'{base}/results/metrics'

# -- Fine-tune XLM-R on English classification (mBERT and MultiReligionBERT already done) --
xlmr_tokenizer = AutoTokenizer.from_pretrained(f'{base}/models/baselines/xlm-roberta')
xlmr_model     = AutoModelForSequenceClassification.from_pretrained(
    f'{base}/models/baselines/xlm-roberta', num_labels=num_labels,
    id2label=id2label, label2id=label_map, ignore_mismatched_sizes=True
)

xlmr_train = load_cls_dataset(f'{cls_dir}/train.json', xlmr_tokenizer)
xlmr_val   = load_cls_dataset(f'{cls_dir}/val.json',   xlmr_tokenizer)
xlmr_test  = load_cls_dataset(f'{cls_dir}/test.json',  xlmr_tokenizer)

xlmr_args = TrainingArguments(
    output_dir=f'{base}/checkpoints/finetuning/cls_xlmr',
    num_train_epochs=10, per_device_train_batch_size=32, per_device_eval_batch_size=32,
    learning_rate=2e-5, warmup_steps=100, weight_decay=0.01, fp16=True,
    eval_strategy='epoch', save_strategy='epoch', save_total_limit=2,
    load_best_model_at_end=True, metric_for_best_model='macro_f1', greater_is_better=True,
    report_to='none',
)
xlmr_trainer = Trainer(model=xlmr_model, args=xlmr_args, train_dataset=xlmr_train,
                       eval_dataset=xlmr_val, compute_metrics=compute_cls_metrics,
                       processing_class=xlmr_tokenizer, callbacks=[PermanentDeleteCallback()])
xlmr_trainer.train()
xlmr_save = f'{base}/models/finetuned/classification/xlmr-classification'
xlmr_model.save_pretrained(xlmr_save); xlmr_tokenizer.save_pretrained(xlmr_save)

# -- Build target-language evaluation sets --
target_languages = {
    'Amharic':'Amharic.json','Shona':'Shona.json','Xhosa':'Xhosa.json',
    'Ewe':'Ewe-NT.json','Swahili':'Swahili-NT.json',
}

crosslingual_datasets = {}
for lang_name, filename in target_languages.items():
    with open(os.path.join(processed_path, filename), 'r', encoding='utf-8') as f:
        verses = json.load(f)
    samples = []
    for verse in verses:
        parts = verse['id'].split('.')
        if len(parts) < 3: continue
        book_name = book_names.get(parts[1])
        if book_name and book_name in label_map:
            samples.append({'text':verse['text'].strip(),'book_code':parts[1],
                            'book_name':book_name,'label':label_map[book_name],'verse_id':verse['id']})
    by_book_t = defaultdict(list)
    for s in samples: by_book_t[s['book_name']].append(s)
    per_book  = max(1, 300 // len(by_book_t))
    balanced  = []
    for book_s in by_book_t.values():
        balanced.extend(random.sample(book_s, min(per_book, len(book_s))))
    random.shuffle(balanced)
    crosslingual_datasets[lang_name] = balanced
    print(f'{lang_name}: {len(balanced)} samples across {len(by_book_t)} books')

# -- Zero-shot evaluation --
def evaluate_zero_shot(model, tokenizer, samples, lang_name, model_name):
    model = model.to(device); model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for item in samples:
            inputs  = tokenizer(item['text'], return_tensors='pt', truncation=True, max_length=128, padding='max_length')
            inputs  = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            all_preds.append(torch.argmax(outputs.logits, dim=-1).item())
            all_labels.append(item['label'])
    acc = acc_metric.compute(predictions=all_preds, references=all_labels)['accuracy']
    f1  = f1_metric.compute(predictions=all_preds, references=all_labels, average='macro')['f1']
    print(f'  {model_name} on {lang_name}: Acc={acc:.4f} | F1={f1:.4f}')
    return {'accuracy': acc, 'macro_f1': f1}

eval_models = {
    'mBERT':             (BertTokenizer.from_pretrained(f'{base}/models/baselines/mbert'),
                          BertForSequenceClassification.from_pretrained(f'{base}/models/finetuned/classification/mBERT-base_classification')),
    'MultiReligionBERT': (BertTokenizer.from_pretrained(f'{base}/models/multi-religion-bert'),
                          BertForSequenceClassification.from_pretrained(f'{base}/models/finetuned/classification/multi-religion-bert_classification')),
    'XLM-R':             (AutoTokenizer.from_pretrained(xlmr_save),
                          AutoModelForSequenceClassification.from_pretrained(xlmr_save)),
}

crosslingual_results = {m: {} for m in eval_models}
for lang_name, samples in crosslingual_datasets.items():
    print(f'\n{lang_name}:')
    for model_name, (tok, mdl) in eval_models.items():
        crosslingual_results[model_name][lang_name] = evaluate_zero_shot(mdl, tok, samples, lang_name, model_name)

with open(f'{metrics_dir}/crosslingual_results.json', 'w') as f:
    json.dump(crosslingual_results, f, indent=2)

# -- Fig 4.9: Zero-shot grouped bar chart --
languages = list(target_languages.keys())
cl_models = list(eval_models.keys())
cl_colors = {'mBERT':'#55A868','MultiReligionBERT':'#C44E52','XLM-R':'#8172B2'}

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Figure 4.9: Zero-shot Cross-lingual Classification Performance', fontsize=13, fontweight='bold')
for ax, (mk, ml) in zip(axes, [('accuracy','Accuracy'),('macro_f1','Macro F1')]):
    x = np.arange(len(languages)); width = 0.25
    for i, mn in enumerate(cl_models):
        vals   = [crosslingual_results[mn][lang][mk] for lang in languages]
        offset = (i - len(cl_models)/2 + 0.5) * width
        bars   = ax.bar(x+offset, vals, width, label=mn, color=cl_colors[mn], edgecolor='white', linewidth=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005, f'{val:.3f}', ha='center', va='bottom', fontsize=7.5)
    ax.set_xticks(x); ax.set_xticklabels(languages, fontsize=10)
    ax.set_ylabel(ml); ax.set_title(ml)
    ax.yaxis.grid(True, linestyle='--', alpha=0.5); ax.set_axisbelow(True)
    ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(f'{figures_dir}/fig4_9_crosslingual_grouped.png', dpi=150, bbox_inches='tight')
plt.show()

# -- Fig 4.10: Cross-lingual heatmap --
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
fig.suptitle('Figure 4.10: Cross-lingual Performance Heatmap', fontsize=13, fontweight='bold')
for ax, (mk, ml) in zip(axes, [('accuracy','Accuracy'),('macro_f1','Macro F1')]):
    data_matrix = np.array([[crosslingual_results[m][l][mk] for l in languages] for m in cl_models])
    im = ax.imshow(data_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=data_matrix.max()*1.1)
    plt.colorbar(im, ax=ax, shrink=0.8, label=ml)
    ax.set_xticks(range(len(languages))); ax.set_xticklabels(languages, fontsize=10, rotation=15, ha='right')
    ax.set_yticks(range(len(cl_models)));  ax.set_yticklabels(cl_models, fontsize=10)
    ax.set_title(ml)
    for i in range(len(cl_models)):
        for j in range(len(languages)):
            ax.text(j, i, f'{data_matrix[i,j]:.3f}', ha='center', va='center', fontsize=9)
plt.tight_layout()
plt.savefig(f'{figures_dir}/fig4_10_crosslingual_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()

# -- Fig 4.11: MultiReligionBERT vs mBERT gain --
fig, ax = plt.subplots(figsize=(9, 5))
fig.suptitle('Figure 4.11: MultiReligionBERT Gain over mBERT (Accuracy)', fontsize=12, fontweight='bold')
gains = [crosslingual_results['MultiReligionBERT'][l]['accuracy'] - crosslingual_results['mBERT'][l]['accuracy'] for l in languages]
bar_colors = ['#2ecc71' if g >= 0 else '#e74c3c' for g in gains]
bars = ax.bar(languages, gains, color=bar_colors, width=0.5, edgecolor='white')
for bar, val in zip(bars, gains):
    sign = '+' if val >= 0 else ''
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+(0.002 if val>=0 else -0.005),
            f'{sign}{val:.4f}', ha='center', va='bottom' if val>=0 else 'top', fontsize=10)
ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
ax.set_ylabel('Accuracy Improvement')
ax.yaxis.grid(True, linestyle='--', alpha=0.4); ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig(f'{figures_dir}/fig4_11_crosslingual_improvement.png', dpi=150, bbox_inches='tight')
plt.show()

print('All figures saved. Done.')

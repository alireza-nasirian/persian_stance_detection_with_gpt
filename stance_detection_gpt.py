import pandas as pd

from langchain.prompts import (
    ChatPromptTemplate,
)
from langchain_openai import OpenAI, ChatOpenAI
from dotenv import load_dotenv
load_dotenv()
import time
from tqdm import tqdm
import openai

from constants import BASE_DIR


# This is a prompt template used to format each individual example.
example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "tweet: {tweet}, reply: {reply}"),
        ("ai", "{label}"),
    ]
)


final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """
سلام جی پی تی
من یکسری توییت و کامنت مربوط به اون رو بهت میدم و تو احساس کامنت نسبت به توییتش بگو.
احساس مستقل کامنت مهم نیست و باید احساس کامنت نسبت به توییت رو بگی.
تو خروجی هم فقط بنویس مثبت، منفی یا خنثی.
اگر کامنت به توییت ربطی نداشت و نتونستی ارتباطی بینشون پیدا کنی و نتونستی تحلیل کنی هم خنثی بزن.
تو سوال بعدی توییت ها و کامنت ها رو میدم"""),
        ("human", "توییت: {tweet}, کامنت: {reply}"),]
)



chain = final_prompt | ChatOpenAI(model_name="gpt-4o", temperature=0.0)

prdictions = []
data = pd.read_csv( BASE_DIR / 'data/manually_labeled_data.csv')

for index, row in tqdm(data.iterrows(), total=data.shape[0]):

    try:
        label = chain.invoke({"tweet": row['main_tweet'],
                        "reply": row["reply_tweet"]}).content

        prdictions.append(label)
    except openai.RateLimitError as e:
        #Handle rate limit error (we recommend using exponential backoff)
        print(f"OpenAI API request exceeded rate limit: {e}")
        time.sleep(5)

data['GPTO_LABELS'] = prdictions

data.to_csv(BASE_DIR / 'data/sample_data_with_GPTO_predictions_one_word_label.csv', index=False)

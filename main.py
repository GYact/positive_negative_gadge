import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import torch
import openai
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict, Tuple, Optional

# GitHub Actions用OpenAI APIキー
openai.api_key = os.getenv('OPENAI_API_KEY', 'sk-xxxxx')

# 感情ラベル
EMOTION_NAMES = ['Joy', 'Sadness', 'Anticipation', 'Surprise', 'Anger', 'Fear', 'Disgust', 'Trust']

@st.cache_resource
def load_model():
    checkpoint = 'cl-tohoku/bert-base-japanese-whole-word-masking'
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=8)
    return model, tokenizer

def np_softmax(x: np.ndarray) -> np.ndarray:
    """ ソフトマックス関数 """
    return np.exp(x) / np.sum(np.exp(x))

def analyze_emotion(text: str, model, tokenizer, show_fig: bool = False) -> Tuple[Dict[str, float], Optional[plt.Figure]]:
    """ 感情分析関数 """
    model.eval()
    try:
        tokens = tokenizer(text, truncation=True, return_tensors="pt")
        with torch.no_grad():
            output = model(**tokens)
            logits = output.logits.cpu().numpy()[0]
        
        prob = np_softmax(logits)
        results = {emotion: float(p) for emotion, p in zip(EMOTION_NAMES, prob)}
        
        fig = None
        if show_fig:
            fig = plt.figure(figsize=(8, 3))
            df = pd.DataFrame(list(results.items()), columns=['感情', '確率'])
            sns.barplot(x='感情', y='確率', data=df)
            plt.title('感情分析結果')
            plt.xticks(rotation=45)
            plt.tight_layout()

        return results, fig
    
    except Exception as e:
        st.error(f"エラーが発生しました: {str(e)}")
        return {}, None

def reverse_emotion_with_gpt(text: str) -> str:
    """ GPTを使用して感情を反転する """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "ポジティブな文章をネガティブに、ネガティブをポジティブに反転します。"},
                {"role": "user", "content": text}
            ],
            max_tokens=500
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        st.error(f"感情反転中にエラーが発生: {str(e)}")
        return text

def determine_sentiment(results: Dict[str, float]) -> str:
    """ 感情のポジティブ/ネガティブ判定 """
    positive = ['Joy', 'Anticipation', 'Trust', 'Surprise']
    negative = ['Sadness', 'Anger', 'Fear', 'Disgust']
    
    pos_score = sum(results[emo] for emo in positive)
    neg_score = sum(results[emo] for emo in negative)
    
    if pos_score > neg_score:
        return "ポジティブ"
    elif neg_score > pos_score:
        return "ネガティブ"
    else:
        return "ニュートラル"

def main():
    st.title('感情分析・反転ツール')
    st.write('入力テキストの感情分析を行い、感情を反転します。')

    try:
        model, tokenizer = load_model()
        st.success('モデル読み込み完了')
    except Exception as e:
        st.error(f"モデルの読み込みエラー: {str(e)}")
        return

    input_text = st.text_area('テキストを入力してください')
    
    if st.button('感情分析開始'):
        with st.spinner("感情分析中..."):
            results, fig = analyze_emotion(input_text, model, tokenizer, show_fig=True)
            if results:
                st.write("### 感情分析結果")
                st.dataframe(pd.DataFrame(results.items(), columns=['感情', '確率']))
                st.pyplot(fig)

                sentiment = determine_sentiment(results)
                st.write(f"判定: **{sentiment}**")

        if openai.api_key:
            with st.spinner("感情反転中..."):
                reversed_text = reverse_emotion_with_gpt(input_text)
                st.write("### 感情反転後のテキスト")
                st.write(reversed_text)

if __name__ == "__main__":
    main()

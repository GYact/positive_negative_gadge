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
import matplotlib.font_manager as fm

# GitHub ActionsとGitHub Codespaces用の環境変数取得
openai.api_key = os.getenv('OPENAI_API_KEY') or st.secrets.get("OPENAI_API_KEY", "")

EMOTION_NAMES = ['Joy', 'Sadness', 'Anticipation', 'Surprise', 'Anger', 'Fear', 'Disgust', 'Trust']

@st.cache_resource
def load_model():
    checkpoint = 'cl-tohoku/bert-base-japanese-whole-word-masking'
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=8)
    return model, tokenizer

def np_softmax(x: np.ndarray) -> np.ndarray:
    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x

def analyze_emotion(text: str, model, tokenizer, show_fig: bool = False) -> Tuple[Dict[str, float], Optional[plt.Figure]]:
    model.eval()

    try:
        tokens = tokenizer(text, truncation=True, return_tensors="pt")
        tokens.to(model.device)
        with torch.no_grad():
            preds = model(**tokens)
        
        prob = np_softmax(preds.logits.cpu().numpy()[0])
        results = {n_jp: float(p) for n_jp, p in zip(EMOTION_NAMES, prob)}
        fig = None
        if show_fig:
            fig = plt.figure(figsize=(8, 3))
            df = pd.DataFrame(list(results.items()), columns=['emotion', 'prob'])
            sns.barplot(x='emotion', y='prob', data=df)
            plt.title(f'Result', fontsize=15)
            plt.xticks(rotation=45)
            plt.tight_layout()

        return results, fig
    
    except Exception as e:
        st.error(f"エラーが発生しました: {str(e)}")
        return {}, None

def reverse_emotion_with_gpt(text: str) -> str:
    """
    OpenAI GPT-4を使用してテキストの感情を反転させる関数
    """
    if not openai.api_key:
        raise ValueError("OpenAI APIキーが設定されていません")
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "あなたは、与えられたテキストの感情を完全に逆転させる特殊な翻訳者です。ポジティブな表現をネガティブに、ネガティブな表現をポジティブに変換してください。文脈や微妙なニュアンスも考慮して、元のテキストの感情を正反対に変えてください。"},
                {"role": "user", "content": text}
            ],
            max_tokens=1000
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"感情反転中にエラーが発生しました: {str(e)}")
        return text

def main():
    st.title('感情分析・感情反転アプリケーション')
    st.write('テキストを入力して感情分析と感情反転を実行します')
    
    try:
        with st.spinner('モデルを読み込んでいます...'):
            model, tokenizer = load_model()
        st.success('モデルの読み込みが完了しました！')
    except Exception as e:
        st.error(f"モデルの読み込みに失敗しました: {str(e)}")
        return
    
    # OpenAI APIキーの確認
    if not openai.api_key:
        st.warning("OpenAI APIキーが設定されていません。感情反転機能は利用できません。")
    
    input_string = st.text_input('テキストを入力', value='')
    
    if input_string:
        # 感情分析
        with st.spinner('分析中...'):
            results, fig = analyze_emotion(input_string, model, tokenizer, show_fig=True)
        
        if results:
            # 分析結果の表示
            st.write('### 感情分析結果')
            df_results = pd.DataFrame(list(results.items()), columns=['感情', '確率'])
            st.dataframe(df_results)
            if fig:
                st.pyplot(fig)
                plt.close(fig)
        
        # 感情反転
        if openai.api_key:
            with st.spinner('感情反転中...'):
                reversed_text = reverse_emotion_with_gpt(input_string)
            
            st.write('### 感情反転後のテキスト')
            st.write(reversed_text)
            
            # 感情反転後のテキストの感情分析
            with st.spinner('反転後のテキストを分析中...'):
                reversed_results, reversed_fig = analyze_emotion(reversed_text, model, tokenizer, show_fig=True)
            
            if reversed_results:
                st.write('### 感情反転後の感情分析結果')
                df_reversed_results = pd.DataFrame(list(reversed_results.items()), columns=['感情', '確率'])
                st.dataframe(df_reversed_results)
                if reversed_fig:
                    st.pyplot(reversed_fig)
                    plt.close(reversed_fig)

if __name__ == "__main__":
    main()
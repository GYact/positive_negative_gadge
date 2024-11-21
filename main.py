import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict, Tuple, Optional
import matplotlib.font_manager as fm

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

def main():
    st.title('感情分析アプリケーション')
    st.write('テキストを入力して感情分析を実行します')
    try:
        with st.spinner('モデルを読み込んでいます...'):
            model, tokenizer = load_model()
        st.success('モデルの読み込みが完了しました！')
    except Exception as e:
        st.error(f"モデルの読み込みに失敗しました: {str(e)}")
        return
    
    input_string = st.text_input('テキストを入力', value='')
    
    if input_string:
        with st.spinner('分析中...'):
            results, fig = analyze_emotion(input_string, model, tokenizer, show_fig=True)
        
        if results:
            st.write('### 分析結果')
            df_results = pd.DataFrame(list(results.items()), columns=['感情', '確率'])
            st.dataframe(df_results)
            if fig:
                st.pyplot(fig)
                plt.close(fig)

if __name__ == "__main__":
    main()
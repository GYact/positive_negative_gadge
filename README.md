# 感情分析・感情反転アプリケーション

## 概要
このアプリケーションは、入力されたテキストの感情分析を行い、さらにその感情を反転させた新しいテキストを生成する Streamlit ベースのWebアプリケーションです。

## 主な機能
1. **感情分析**: 入力テキストの8つの感情（Joy, Sadness, Anticipation, Surprise, Anger, Fear, Disgust, Trust）を分析
2. **感情判定**: ポジティブ/ネガティブ/ニュートラルの判定
3. **感情反転**: OpenAI GPTを使用して、テキストの感情を反転
4. **可視化**: 感情分析結果をグラフで表示

## 必要要件
```
numpy
pandas
matplotlib
seaborn
streamlit
torch
openai
transformers
```

## セットアップ
1. 必要なパッケージのインストール:
```bash
pip install numpy pandas matplotlib seaborn streamlit torch openai transformers
```

2. OpenAI APIキーの設定:
- 環境変数として設定するか
- コード内の `openai.api_key = 'sk-xxxxx'` を実際のAPIキーに書き換え

## 使用方法
1. アプリケーションの起動:
```bash
streamlit run main.py
```

2. ブラウザで表示されるインターフェースに分析したいテキストを入力

3. 結果の確認:
   - 感情分析結果（8つの感情の確率）
   - 感情判定結果（ポジティブ/ネガティブ/ニュートラル）
   - 感情反転後のテキスト
   - 反転後のテキストの感情分析結果

## 技術的な詳細
- BERTモデル: `cl-tohoku/bert-base-japanese-whole-word-masking`を使用
- 感情分析: 8つの感情カテゴリーに対する確率を算出
- 感情反転: GPT-4を使用して感情表現を反転
- キャッシュ機能: `@st.cache_resource`でモデルの読み込みを最適化

## 注意事項
- OpenAI APIキーが設定されていない場合、感情反転機能は利用できません
- 大量のテキストを処理する場合、モデルの読み込みに時間がかかる可能性があります
- 日本語テキストの処理に最適化されています

## エラーハンドリング
- モデル読み込みエラー
- API通信エラー
- テキスト処理エラー
に対して適切なエラーメッセージを表示します

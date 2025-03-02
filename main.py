import streamlit as st
import pandas as pd
import re
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib

# nltk関連のエラーを回避するため、独自のトークナイザーを実装
def simple_word_tokenize(text):
    """簡易的な単語分割関数（日本語向け）"""
    # 基本的な句読点やスペースで分割
    words = re.findall(r'\w+|[^\s\w]', text)
    return [w for w in words if w.strip()]

def simple_sent_tokenize(text):
    """簡易的な文分割関数（日本語向け）"""
    # 句点や改行で文を分割
    sentences = re.split(r'[。.!?！？\n]+', text)
    return [s for s in sentences if s.strip()]

# タイトルとアプリの説明
st.title('看護試験事例文 情報量評価アプリケーション')
st.write('看護試験の事例文を分析し、情報量を定量的に評価します。')

# サイドバーに評価基準の説明を追加
with st.sidebar:
    st.header('評価基準の説明')
    st.markdown("""
    **情報量スコア計算方法:**
    1. **文章の長さ**: 文字数、単語数、文の数
    2. **専門用語の密度**: 看護・医療用語の出現頻度
    3. **情報の多様性**: 異なる種類の情報（患者基本情報、症状、治療歴など）
    4. **時間的要素**: 経過情報の有無と詳細さ
    5. **文章の複雑さ**: 文の長さのばらつき、接続詞の使用
    """)
    
    st.markdown("""
    **情報カテゴリ:**
    - 患者基本情報（年齢、性別、職業など）
    - 病歴・既往歴
    - 現在の症状
    - バイタルサイン
    - 検査結果
    - 治療内容
    - 患者の心理状態
    - 家族状況
    - 社会的背景
    """)

# 看護・医療用語リスト
nursing_medical_terms = [
    '血圧', '脈拍', '体温', '呼吸', 'SpO2', '意識', '疼痛', '浮腫', '発熱', '嘔吐', '下痢',
    '便秘', '尿閉', '頻尿', '血尿', '排尿', '排便', '食欲', '睡眠', '活動', 'ADL', '歩行',
    '移動', '清潔', '入浴', '更衣', '食事', '内服', '点滴', '注射', '輸血', '手術', '処置',
    '検査', 'CT', 'MRI', 'エコー', 'レントゲン', '心電図', '血液検査', '尿検査', '糖尿病',
    '高血圧', '心不全', '呼吸不全', '腎不全', '肝不全', '脳梗塞', '心筋梗塞', '肺炎', '癌',
    '認知症', '白血球', '赤血球', 'ヘモグロビン', '血小板', 'CRP', 'HbA1c', 'ALT', 'AST',
    'γ-GTP', 'BUN', 'Cr', 'Na', 'K', 'Cl', '酸素', '人工呼吸器', '酸素マスク', '酸素カニューレ',
    '気管内挿管', '気管切開', '中心静脈カテーテル', '末梢静脈カテーテル', '尿道カテーテル',
    '胃管', '経鼻胃管', '胃瘻', '褥瘡', '創部', '感染', '発赤', '腫脹', '熱感', '疼痛', '膿',
    '血液', '分泌物', '排液', '陰圧閉鎖療法', '抗生物質', '抗菌薬', '解熱鎮痛薬', '抗凝固薬',
    '利尿薬', '降圧薬', '睡眠薬', '鎮痛薬', '麻薬', '向精神薬', '副作用', 'アレルギー',
    'アナフィラキシー', 'ショック', '意識レベル', 'JCS', 'GCS', '昏睡', '昏迷', '錯乱',
    '傾眠', '譫妄', '不安', '抑うつ', '幻覚', '妄想', 'ADL', 'IADL', 'QOL', 'IC', 'インフォームドコンセント'
]

# 情報カテゴリとそれに関連するキーワード
info_categories = {
    '患者基本情報': ['歳', '代', '性別', '男性', '女性', '職業', '会社員', '主婦', '無職', '身長', '体重', 'BMI'],
    '病歴・既往歴': ['既往歴', '合併症', '手術歴', '入院歴', '発症', '罹患', '診断', '治療歴'],
    '現在の症状': ['症状', '訴え', '痛み', '疼痛', '不快感', '違和感', '倦怠感', '発熱', '咳', '痰', '呼吸困難', '動悸', '浮腫'],
    'バイタルサイン': ['血圧', '脈拍', '体温', '呼吸', 'SpO2', '意識レベル', 'JCS', 'GCS'],
    '検査結果': ['検査', '血液検査', '尿検査', '画像検査', 'CT', 'MRI', 'エコー', 'レントゲン', '心電図', '数値', '基準値'],
    '治療内容': ['治療', '処置', '手術', '投薬', '内服', '点滴', '注射', '輸血', '酸素療法', 'リハビリ'],
    '患者の心理状態': ['不安', '恐怖', '抑うつ', '悲嘆', '受容', '拒否', '怒り', '精神状態', '気分', '感情'],
    '家族状況': ['家族', '配偶者', '子供', '親', '兄弟', '姉妹', '同居', '介護者', 'キーパーソン'],
    '社会的背景': ['職業', '経済状況', '住環境', '介護保険', '社会資源', '支援', 'サービス', '地域']
}

# テキスト入力エリア
text_input = st.text_area('事例文を入力してください：', height=300)

# 分析ボタン
analyze_button = st.button('分析開始')

# 分析関数
def analyze_nursing_case(text):
    results = {}
    
    # 1. 基本的なテキスト統計
    char_count = len(text)
    words = simple_word_tokenize(text)
    word_count = len(words)
    sentences = simple_sent_tokenize(text)
    sentence_count = len(sentences)
    
    results['基本統計'] = {
        '文字数': char_count,
        '単語数': word_count,
        '文の数': sentence_count,
        '平均文長（文字）': char_count / sentence_count if sentence_count > 0 else 0
    }
    
    # 2. 専門用語の分析
    term_count = 0
    found_terms = []
    for term in nursing_medical_terms:
        count = text.count(term)
        if count > 0:
            term_count += count
            found_terms.append((term, count))
    
    term_density = term_count / word_count if word_count > 0 else 0
    results['専門用語分析'] = {
        '専門用語数': term_count,
        '専門用語密度': term_density,
        '検出された用語': sorted(found_terms, key=lambda x: x[1], reverse=True)
    }
    
    # 3. 情報カテゴリの分析
    category_counts = {}
    category_terms = {}
    
    for category, terms in info_categories.items():
        category_count = 0
        found_category_terms = []
        
        for term in terms:
            count = text.count(term)
            if count > 0:
                category_count += count
                found_category_terms.append((term, count))
        
        category_counts[category] = category_count
        category_terms[category] = found_category_terms
    
    results['情報カテゴリ分析'] = {
        'カテゴリ出現数': category_counts,
        '検出されたカテゴリ用語': category_terms
    }
    
    # 4. 文の複雑さ分析
    sentence_lengths = [len(s) for s in sentences]
    sentence_length_std = np.std(sentence_lengths) if sentence_count > 0 else 0
    
    # 接続詞の検出（簡易版）
    connectives = ['しかし', 'また', 'そして', 'ただし', 'ために', 'ので', 'よって', 'したがって', 
                  'なぜなら', 'さらに', 'それから', 'それに', 'あるいは', 'もしくは', 'それとも']
    connective_count = sum(text.count(c) for c in connectives)
    
    results['文章複雑性分析'] = {
        '文長の標準偏差': sentence_length_std,
        '接続詞使用数': connective_count,
        '接続詞密度': connective_count / sentence_count if sentence_count > 0 else 0
    }
    
    # 5. 時間的要素の分析
    time_patterns = [
        r'\d+日', r'\d+時間', r'\d+分', r'\d+秒', r'\d+ヶ月', r'\d+年',
        '昨日', '今日', '明日', '先週', '今週', '来週', '先月', '今月', '来月',
        '午前', '午後', '朝', '昼', '夕方', '夜', '深夜',
        '入院時', '手術前', '手術後', '治療前', '治療後', '発症時'
    ]
    
    time_references = []
    for pattern in time_patterns:
        matches = re.findall(pattern, text)
        time_references.extend(matches)
    
    results['時間的要素分析'] = {
        '時間参照数': len(time_references),
        '検出された時間表現': Counter(time_references)
    }
    
    # 6. 総合スコアの計算
    # 各指標を0-10のスケールに正規化して加重平均をとる
    
    # 基本量スコア: 文字数を基準に (0-1000文字を0-10にスケーリング、上限あり)
    length_score = min(char_count / 100, 10)
    
    # 専門用語スコア: 密度を基準に (0-0.2を0-10にスケーリング、上限あり)
    term_score = min(term_density * 50, 10)
    
    # 情報多様性スコア: カテゴリカバレッジを基準に
    category_coverage = sum(1 for count in category_counts.values() if count > 0) / len(info_categories)
    diversity_score = category_coverage * 10
    
    # 時間要素スコア: 時間参照の数を基準に (0-10参照を0-10にスケーリング、上限あり)
    time_score = min(len(time_references), 10)
    
    # 文章複雑性スコア: 文長のばらつきと接続詞の使用を考慮
    complexity_score = min((sentence_length_std / 10 + results['文章複雑性分析']['接続詞密度'] * 5) / 2, 10)
    
    # 総合スコア (重み付け)
    weights = {
        '基本量': 0.15,
        '専門用語': 0.25,
        '情報多様性': 0.3,
        '時間要素': 0.15,
        '文章複雑性': 0.15
    }
    
    component_scores = {
        '基本量': length_score,
        '専門用語': term_score,
        '情報多様性': diversity_score,
        '時間要素': time_score,
        '文章複雑性': complexity_score
    }
    
    total_score = sum(score * weights[component] for component, score in component_scores.items())
    
    results['スコア'] = {
        '総合スコア': total_score,
        'コンポーネントスコア': component_scores
    }
    
    # 7. 改善提案
    suggestions = []
    
    if length_score < 5:
        suggestions.append("事例文の長さが不足しています。より詳細な情報を追加してください。")
    
    if term_score < 5:
        suggestions.append("看護・医療専門用語の使用が少ないです。適切な専門用語を追加して情報の質を高めてください。")
    
    low_categories = [category for category, count in category_counts.items() if count == 0]
    if low_categories:
        suggestions.append(f"以下の情報カテゴリが不足しています: {', '.join(low_categories)}")
    
    if time_score < 5:
        suggestions.append("時間的要素（経過、頻度、期間など）の情報が不足しています。時間的文脈を追加してください。")
    
    if complexity_score < 5:
        suggestions.append("文章の構造が単調です。様々な長さの文や適切な接続詞を使用して、情報の関連性を明確にしてください。")
    
    results['改善提案'] = suggestions
    
    return results

# 分析結果の表示
if analyze_button and text_input:
    with st.spinner('分析中...'):
        results = analyze_nursing_case(text_input)
    
    # 基本統計の表示
    st.subheader('1. 基本統計')
    basic_stats = pd.DataFrame(list(results['基本統計'].items()), columns=['指標', '値'])
    st.table(basic_stats)
    
    # 総合スコアの表示
    st.subheader('総合評価スコア')
    total_score = results['スコア']['総合スコア']
    st.markdown(f"### スコア: {total_score:.2f} / 10")
    
    # スコアの評価
    if total_score >= 8:
        st.success('この事例文は情報量が十分で、看護試験の問題として適切です。')
    elif total_score >= 6:
        st.warning('この事例文は基本的な情報を含んでいますが、いくつかの改善が必要です。')
    else:
        st.error('この事例文は情報量が不足しており、看護試験の問題としては不十分です。')
    
    # コンポーネントスコアのグラフ表示
    component_scores = results['スコア']['コンポーネントスコア']
    comp_df = pd.DataFrame({
        'コンポーネント': list(component_scores.keys()),
        'スコア': list(component_scores.values())
    })
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='コンポーネント', y='スコア', data=comp_df, ax=ax)
    ax.set_ylim(0, 10)
    ax.set_title('評価コンポーネント別スコア')
    st.pyplot(fig)
    
    # 情報カテゴリの分析
    st.subheader('2. 情報カテゴリ分析')
    category_data = results['情報カテゴリ分析']['カテゴリ出現数']
    cat_df = pd.DataFrame({
        'カテゴリ': list(category_data.keys()),
        '検出数': list(category_data.values())
    })
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='カテゴリ', y='検出数', data=cat_df, ax=ax)
    plt.xticks(rotation=45, ha='right')
    ax.set_title('情報カテゴリ別の出現数')
    st.pyplot(fig)
    
    # 専門用語の表示
    st.subheader('3. 専門用語分析')
    st.write(f"専門用語数: {results['専門用語分析']['専門用語数']}")
    st.write(f"専門用語密度: {results['専門用語分析']['専門用語密度']:.4f}")
    
    if results['専門用語分析']['検出された用語']:
        terms_df = pd.DataFrame(results['専門用語分析']['検出された用語'], columns=['用語', '出現回数'])
        st.dataframe(terms_df.head(20))
    else:
        st.write("専門用語は検出されませんでした。")
    
    # 時間的要素の表示
    st.subheader('4. 時間的要素分析')
    st.write(f"時間参照数: {results['時間的要素分析']['時間参照数']}")
    if results['時間的要素分析']['検出された時間表現']:
        time_df = pd.DataFrame(list(results['時間的要素分析']['検出された時間表現'].items()), 
                              columns=['時間表現', '出現回数'])
        st.dataframe(time_df)
    else:
        st.write("時間的表現は検出されませんでした。")
    
    # 文章複雑性の表示
    st.subheader('5. 文章複雑性分析')
    complexity_df = pd.DataFrame(list(results['文章複雑性分析'].items()), columns=['指標', '値'])
    st.table(complexity_df)
    
    # 改善提案の表示
    st.subheader('改善提案')
    if results['改善提案']:
        for suggestion in results['改善提案']:
            st.markdown(f"- {suggestion}")
    else:
        st.success("特に改善提案はありません。十分な情報量を持つ良質な事例文です。")
    
    # ダウンロード用の分析結果データフレーム作成
    @st.cache_data
    def convert_df_to_csv(df):
        return df.to_csv(index=False).encode('utf-8-sig')
    
    analysis_summary = pd.DataFrame([
        ["文字数", results['基本統計']['文字数']],
        ["文の数", results['基本統計']['文の数']],
        ["専門用語数", results['専門用語分析']['専門用語数']],
        ["専門用語密度", results['専門用語分析']['専門用語密度']],
        ["検出されたカテゴリ数", sum(1 for count in results['情報カテゴリ分析']['カテゴリ出現数'].values() if count > 0)],
        ["時間参照数", results['時間的要素分析']['時間参照数']],
        ["総合スコア", results['スコア']['総合スコア']]
    ], columns=["指標", "値"])
    
    csv = convert_df_to_csv(analysis_summary)
    st.download_button(
        label="分析結果をCSVでダウンロード",
        data=csv,
        file_name='nursing_case_analysis.csv',
        mime='text/csv',
    )

else:
    if analyze_button:
        st.warning('事例文を入力してください。')
    else:
        st.info('事例文を入力して「分析開始」ボタンをクリックしてください。')

# サンプル事例文の提供
with st.expander("サンプル事例文"):
    sample_text = """
    65歳男性。2型糖尿病、高血圧症、脂質異常症で内科通院中。3日前から38.5℃の発熱、咳嗽、喀痰があり、昨日から呼吸困難感が出現したため救急外来を受診。SpO2 88%(room air)、胸部X線で両側肺野に浸潤影を認め、COVID-19の診断で即日入院となった。
    
    入院時、体温39.0℃、脈拍110/分、血圧145/85mmHg、呼吸数28回/分、SpO2 92%(酸素3L/分マスク)。意識清明だが、会話で息切れがみられる。咳は強く、粘稠性の痰の喀出が困難。食事摂取量は3日前から徐々に低下し、水分摂取も不十分。尿量減少はない。
    
    血液検査ではWBC 12,500/μL、CRP 8.5mg/dL、LDH 450U/L、D-dimer 1.2μg/mL、血糖値245mg/dL。HbA1c 7.8%。
    
    既往歴として10年前から2型糖尿病でインスリン注射と内服薬による治療中。5年前から高血圧症と脂質異常症に対して内服治療中。アレルギーはなし。
    
    家族構成は妻と二人暮らし。息子夫婦は隣市に住んでいる。本人は定年退職後も週3日パートとして働いていた。ADLは自立していたが、入院後は酸素投与中であり、息切れのため日常生活動作に介助が必要。食事や排泄にも看護師の見守りを要する。本人は「こんなに急に具合が悪くなるとは思わなかった。早く良くなって退院したい」と話す。
    """
    st.text_area("サンプル事例文（コピーして使用できます）", sample_text, height=200)
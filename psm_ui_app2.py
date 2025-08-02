# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# cd C:\Users\user\iwaiwa\0801_クラスタと予測UI
#
# streamlit run 3.py

# %%
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.graph_objects as go
from scipy.interpolate import interp1d

st.set_page_config(layout="wide")
st.title("💴 Van Westendorp PSM + クラスタリング分析アプリ")

@st.cache_data
def load_data(uploaded_file):
    return pd.read_csv(uploaded_file)

def find_intersection(y1, y2, x):
    diff = np.array(y1) - np.array(y2)
    sign_change = np.where(np.diff(np.sign(diff)) != 0)[0]
    if len(sign_change) == 0:
        return None
    i = sign_change[0]
    try:
        f = interp1d(diff[i:i+2], x[i:i+2])
        return float(f(0))
    except Exception:
        return None

uploaded_file = st.file_uploader("📂 CSVファイルをアップロード", type=["csv"])

if uploaded_file is not None:
    df = load_data(uploaded_file)
    st.markdown("#### 🔍 絞り込みフィルター")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if '年齢' in df.columns:
            min_age = int(df['年齢'].min())
            max_age = int(df['年齢'].max())
            selected_age_range = st.slider("🔍 年齢範囲", min_age, max_age, (min_age, max_age))
        else:
            selected_age_range = None
    
        # 性別フィルター
        if '性別' in df.columns:
            gender_options = df['性別'].dropna().unique().tolist()
            if 'selected_gender' not in st.session_state:
                st.session_state.selected_gender = gender_options
            selected_gender = st.multiselect(
                "🔍 性別", options=gender_options, default=st.session_state.selected_gender
            )
            st.session_state.selected_gender = selected_gender
        else:
            selected_gender = None
    
        # 職業フィルター
        if '職業' in df.columns:
            job_options = df['職業'].dropna().unique().tolist()
            if 'selected_jobs' not in st.session_state:
                st.session_state.selected_jobs = job_options
            selected_jobs = st.multiselect(
                "🔍 職業", options=job_options, default=st.session_state.selected_jobs
            )
            st.session_state.selected_jobs = selected_jobs
        else:
            selected_jobs = None
    
        if '平均購入単価' in df.columns:
            min_price = int(df['平均購入単価'].min())
            max_price = int(df['平均購入単価'].max())
            selected_average_bands = st.slider("🔍 平均購入単価", min_price, max_price, (min_price, max_price))
        else:
            selected_average_bands = None
    
        if 'SNS利用時間' in df.columns:
            min_sns = int(df['SNS利用時間'].min())
            max_sns = int(df['SNS利用時間'].max())
            selected_sns = st.slider("🔍 SNS利用時間", min_sns, max_sns, (min_sns, max_sns))
        else:
            selected_sns = None
    
    with col2:
        # キャラ傾向フィルター
        if 'キャラ傾向' in df.columns:
            char_options = df['キャラ傾向'].dropna().unique().tolist()
            if 'selected_character' not in st.session_state:
                st.session_state.selected_character = char_options
            selected_character = st.multiselect(
                "🔍 キャラ傾向", options=char_options, default=st.session_state.selected_character
            )
            st.session_state.selected_character = selected_character
        else:
            selected_character = None
    
        # 重要視すること1 フィルター
        if '重要視すること' in df.columns:
            imp_options = df['重要視すること'].dropna().unique().tolist()
            if 'selected_importance' not in st.session_state:
                st.session_state.selected_importance = imp_options
            selected_importance = st.multiselect(
                "🔍 重要視すること", options=imp_options, default=st.session_state.selected_importance
            )
            st.session_state.selected_importance = selected_importance
        else:
            selected_importance = None
    
        # 購買頻度フィルター
        if '購買頻度' in df.columns:
            freq_options = df['購買頻度'].dropna().unique().tolist()
            if 'selected_frequency' not in st.session_state:
                st.session_state.selected_frequency = freq_options
            selected_frequency = st.multiselect(
                "🔍 購買頻度", options=freq_options, default=st.session_state.selected_frequency
            )
            st.session_state.selected_frequency = selected_frequency
        else:
            selected_frequency = None
    
    with col3:
        if '購入スタイル' in df.columns:
            style_options = df['購入スタイル'].dropna().unique().tolist()
            st.markdown("🔍 購入スタイル")
    
            if "selected_style" not in st.session_state:
                st.session_state.selected_style = {s: True for s in style_options}
    
            colA, colB = st.columns(2)
            with colA:
                if st.button("✅ 全て選択"):
                    for s in style_options:
                        st.session_state.selected_style[s] = True
            with colB:
                if st.button("❌ 全て解除"):
                    for s in style_options:
                        st.session_state.selected_style[s] = False
    
            selected_style = []
            for s in style_options:
                checked = st.checkbox(s, value=st.session_state.selected_style[s])
                st.session_state.selected_style[s] = checked
                if checked:
                    selected_style.append(s)
        else:
            selected_style = None
            
    # ================
    # フィルター処理
    # ================
    filtered_df = df.copy()
    if selected_age_range:
        filtered_df = filtered_df[filtered_df['年齢'].between(*selected_age_range)]
    if selected_gender:
        filtered_df = filtered_df[filtered_df['性別'].isin(selected_gender)]
    if selected_jobs:
        filtered_df = filtered_df[filtered_df['職業'].isin(selected_jobs)]
    if selected_character:
        filtered_df = filtered_df[filtered_df['キャラ傾向'].isin(selected_character)]
    if selected_frequency:
        filtered_df = filtered_df[filtered_df['購買頻度'].isin(selected_frequency)]
    if selected_style:
        filtered_df = filtered_df[filtered_df['購入スタイル'].isin(selected_style)]
    if selected_importance:
        filtered_df = filtered_df[filtered_df['重要視すること'].isin(selected_importance)]
    if selected_sns:
        filtered_df = filtered_df[filtered_df['SNS利用時間'].between(*selected_sns)]
    if selected_average_bands:
        filtered_df = filtered_df[filtered_df['平均購入単価'].between(*selected_average_bands)]

    st.markdown(f"#### <フィルター後の対象者数: {len(filtered_df)} 人>")

    # ======================
    # クラスタリング
    # ======================
    st.markdown("### 🧩 クラスタリング設定")

    # クラスタリングに使う特徴量選択UI（カテゴリはラベルエンコーディング）
    candidate_features = ['年齢', '性別','職業','購買頻度','平均購入単価', '購入スタイル', '重要視すること']
    selected_features = st.multiselect("クラスタリングに使う属性を選択してください", candidate_features, default=candidate_features)

    if len(selected_features) == 0:
        st.warning("少なくとも1つ以上の特徴量を選択してください。")
    else:
        cluster_count = st.slider("クラスタ数 (K)", 2, 10, 3)

        # 前処理：カテゴリ変数はlabel encoding（簡易版）
        from sklearn.preprocessing import LabelEncoder
        X = filtered_df[selected_features].copy()

        for col in X.columns:
            if X[col].dtype == 'object' or X[col].dtype.name == 'category':
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))

        # 標準化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # KMeansクラスタリング
        kmeans = KMeans(n_clusters=cluster_count, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)

        filtered_df['cluster'] = clusters

        st.markdown(f"#### クラスタリング結果（K={cluster_count}）")
        st.write(filtered_df[['ID'] + selected_features + ['cluster']])

        # クラスタごとの人数表示
#        st.markdown("#### クラスタ人数分布")
#        cluster_counts = filtered_df['cluster'].value_counts().sort_index()
#        st.bar_chart(cluster_counts)


        # クラスタ人数分布の下に追加
        st.markdown("#### クラスタ内容")
    
        num_clusters = filtered_df['cluster'].nunique()
        
        num_features = ['年齢', 'SNS利用時間', '平均購入単価']
        cat_features = ['性別', '職業', '購買頻度',"重要視すること","購入スタイル", 'キャラ傾向']

        rows = []
        
        for c in range(num_clusters):
            cluster_df = filtered_df[filtered_df['cluster'] == c]
            n = len(cluster_df)
            
            row = {"クラスタ": c, "人数": n}
            
            # 数値特徴量の平均値
            for f in num_features:
                if f in cluster_df.columns:
                    row[f"{f}平均"] = round(cluster_df[f].mean(), 2)
            
            # カテゴリ特徴量は最頻値と割合を表示（例）
            for f in cat_features:
                if f in cluster_df.columns:
                    top_val = cluster_df[f].value_counts(normalize=True).idxmax()
                    top_ratio = cluster_df[f].value_counts(normalize=True).max()
                    row[f"{f}（最多）"] = f"{top_val} ({top_ratio:.1%})"
            
            rows.append(row)
        
        summary_df = pd.DataFrame(rows)
        
        st.dataframe(summary_df)






        
        # =======================
        # クラスタ別PSM分析表示
        # =======================
        st.markdown("#### 📊 クラスタ別 Van Westendorp PSM分析")

        labels = ['too_cheap', 'cheap', 'expensive', 'too_expensive']
        brands = sorted(set(col.split('_')[0] for col in df.columns if any(lbl in col for lbl in labels)))

        cluster_tabs = st.tabs([f"Cluster {i}" for i in range(cluster_count)])

        for cluster_id, tab in zip(range(cluster_count), cluster_tabs):
            with tab:
                st.markdown(f"##### Cluster {cluster_id} の分析")
        
                df_cluster = filtered_df[filtered_df['cluster'] == cluster_id]
                results = []  # 各ブランドのPSM指標を保存するリスト
        
                for brand in brands:
                    brand_cols = [f"{brand}_{lbl}" for lbl in labels if f"{brand}_{lbl}" in df_cluster.columns]
                    df_brand = df_cluster[df_cluster[brand_cols].notnull().any(axis=1)]
        
                    if df_brand.empty:
                        st.warning(f"{brand} のデータがありません。")
                        continue
        
                    price_data = {
                        label: df_brand[f"{brand}_{label}"].dropna().astype(int).values
                        for label in labels if f"{brand}_{label}" in df_brand.columns
                    }
        
                    valid_arrays = [arr for arr in price_data.values() if len(arr) > 0]
                    if not valid_arrays:
                        st.warning("有効な価格データがありません。")
                        continue
        
                    all_prices = np.arange(
                        min(np.concatenate(valid_arrays)),
                        max(np.concatenate(valid_arrays)) + 1000,
                        100
                    )
                    n = len(df_brand)
        
                    cumulative = {
                        'too_cheap': [np.sum(price_data.get('too_cheap', []) >= p) / n for p in all_prices],
                        'cheap': [np.sum(price_data.get('cheap', []) >= p) / n for p in all_prices],
                        'expensive': [np.sum(price_data.get('expensive', []) <= p) / n for p in all_prices],
                        'too_expensive': [np.sum(price_data.get('too_expensive', []) <= p) / n for p in all_prices],
                    }
        
                    opp = find_intersection(cumulative['cheap'], cumulative['expensive'], all_prices)
                    idp = find_intersection(cumulative['too_cheap'], cumulative['too_expensive'], all_prices)
                    pme = find_intersection(cumulative['cheap'], cumulative['too_expensive'], all_prices)
                    pmc = find_intersection(cumulative['expensive'], cumulative['too_cheap'], all_prices)
        
                    # グラフ生成
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=all_prices, y=np.array(cumulative['too_cheap'])*100, name='Too Cheap', line=dict(color='blue')))
                    fig.add_trace(go.Scatter(x=all_prices, y=np.array(cumulative['cheap'])*100, name='Cheap', line=dict(color='green')))
                    fig.add_trace(go.Scatter(x=all_prices, y=np.array(cumulative['expensive'])*100, name='Expensive', line=dict(color='orange')))
                    fig.add_trace(go.Scatter(x=all_prices, y=np.array(cumulative['too_expensive'])*100, name='Too Expensive', line=dict(color='red')))
        
                    for val, name, color in zip(
                        [opp, idp, pme, pmc],
                        ['OPP（最適）', 'IDP（無関心）', 'PME（上限）', 'PMC（下限）'],
                        ['purple', 'black', 'magenta', 'cyan']
                    ):
                        if val:
                            fig.add_vline(x=val, line_dash='dash', line_color=color)
                            fig.add_annotation(x=val, y=50, text=name, showarrow=False, textangle=90,
                                               font=dict(color=color, size=12), bgcolor='white')
        
                    fig.update_layout(
                        title=f"{brand} - PSM分析",
                        xaxis_title="価格（円）",
                        yaxis_title="累積比率（%）",
                        height=400,
                        hovermode="x unified",
                        xaxis=dict(tickformat='d')
                    )
        
                    # ---- 全体集計（フィルターあと）----
                    col_plot, col_info = st.columns([3, 1])
                    with col_plot:
                        st.plotly_chart(fig, use_container_width=True)
                    with col_info:
                        st.markdown("#### 👇 指標")
                        st.markdown(f"**{brand} の該当人数：{df_brand.shape[0]}人**")
                        st.write(f"📌 **最適価格（OPP）**: {round(opp) if opp else '計算不可'} 円")
                        st.write(f"📌 **無関心価格（IDP）**: {round(idp) if idp else '計算不可'} 円")
                        st.write(f"📌 **価格受容範囲下限（PMC）**: {round(pmc) if pmc else '計算不可'} 円")
                        st.write(f"📌 **価格受容範囲上限（PME）**: {round(pme) if pme else '計算不可'} 円")
        
                    # 指標データをリストに追加
                    results.append({
                        "ブランド": brand,
                        "該当人数": df_brand.shape[0],
                        "最適価格（OPP）": round(opp) if opp else None,
                        "無関心価格（IDP）": round(idp) if idp else None,
                        "価格受容範囲下限（PMC）": round(pmc) if pmc else None,
                        "価格受容範囲上限（PME）": round(pme) if pme else None
                    })


                
                # 🔽 各クラスタ内の全ブランドまとめ表（PSM指標一覧）
                if results:
                    st.markdown("#### 📊 クラスタ内ブランド別 PSM指標 一覧")
                    df_result = pd.DataFrame(results)
                    st.dataframe(df_result.style.format({col: "{:.0f}" for col in ["OPP", "IDP", "PMC", "PME"]}))
                else:
                    st.info("このクラスタには有効なPSMデータがありません。")
        
        # ---- 全体集計（フィルター前）----
        results_before_filter = []
        for brand in brands:
            brand_cols = [f"{brand}_{lbl}" for lbl in labels if f"{brand}_{lbl}" in df.columns]
            df_brand = df[df[brand_cols].notnull().any(axis=1)]
            if df_brand.empty:
                continue
        
            price_data = {
                label: df_brand[f"{brand}_{label}"].dropna().astype(int).values
                for label in labels if f"{brand}_{label}" in df_brand.columns
            }
        
            valid_arrays = [arr for arr in price_data.values() if len(arr) > 0]
            if not valid_arrays:
                continue
        
            all_prices = np.arange(
                min(np.concatenate(valid_arrays)),
                max(np.concatenate(valid_arrays)) + 1000,
                100
            )
            n = len(df_brand)
        
            cumulative = {
                'too_cheap': [np.sum(price_data.get('too_cheap', []) >= p) / n for p in all_prices],
                'cheap': [np.sum(price_data.get('cheap', []) >= p) / n for p in all_prices],
                'expensive': [np.sum(price_data.get('expensive', []) <= p) / n for p in all_prices],
                'too_expensive': [np.sum(price_data.get('too_expensive', []) <= p) / n for p in all_prices],
            }
        
            opp = find_intersection(cumulative['cheap'], cumulative['expensive'], all_prices)
            idp = find_intersection(cumulative['too_cheap'], cumulative['too_expensive'], all_prices)
            pme = find_intersection(cumulative['cheap'], cumulative['too_expensive'], all_prices)
            pmc = find_intersection(cumulative['expensive'], cumulative['too_cheap'], all_prices)
        
            results_before_filter.append({
                "ブランド": brand,
                "OPP": opp,
                "IDP": idp,
                "PMC": pmc,
                "PME": pme
            })
        
        # ---- 表示部 ----
        st.markdown("---")
        with st.expander("📋 フィルター前 ブランド別 PSM 指標一覧（全体集計）", expanded=False):
            st.markdown(f"**全体調査人数：{len(df)}人**")
            summary_df_before = pd.DataFrame(results_before_filter)
            st.dataframe(summary_df_before.style.format({col: "{:.0f}" for col in summary_df_before.columns if col != "ブランド"}))
        


else:
    st.info("CSVファイルをアップロードしてください。")

# %%

# %% [markdown]
# 前の指標の表示　OK
#

# %% [markdown] jupyter={"source_hidden": true}
# import streamlit as st
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler
# from sklearn.cluster import KMeans
# import plotly.graph_objects as go
# from scipy.interpolate import interp1d
#
# st.set_page_config(layout="wide")
# st.title("💴 Van Westendorp PSM + クラスタリング分析アプリ")
#
# @st.cache_data
# def load_data(uploaded_file):
#     return pd.read_csv(uploaded_file)
#
# def find_intersection(y1, y2, x):
#     diff = np.array(y1) - np.array(y2)
#     sign_change = np.where(np.diff(np.sign(diff)) != 0)[0]
#     if len(sign_change) == 0:
#         return None
#     i = sign_change[0]
#     try:
#         f = interp1d(diff[i:i+2], x[i:i+2])
#         return float(f(0))
#     except Exception:
#         return None
#
# uploaded_file = st.file_uploader("📂 CSVファイルをアップロード", type=["csv"])
#
# if uploaded_file is not None:
#     df = load_data(uploaded_file)
#     st.markdown("#### 🔍 絞り込みフィルター")
#     col1, col2, col3 = st.columns(3)
#     
#     with col1:
#         if '年齢' in df.columns:
#             min_age = int(df['年齢'].min())
#             max_age = int(df['年齢'].max())
#             selected_age_range = st.slider("🔍 年齢範囲", min_age, max_age, (min_age, max_age))
#         else:
#             selected_age_range = None
#     
#         # 性別フィルター
#         if '性別' in df.columns:
#             gender_options = df['性別'].dropna().unique().tolist()
#             if 'selected_gender' not in st.session_state:
#                 st.session_state.selected_gender = gender_options
#             selected_gender = st.multiselect(
#                 "🔍 性別", options=gender_options, default=st.session_state.selected_gender
#             )
#             st.session_state.selected_gender = selected_gender
#         else:
#             selected_gender = None
#     
#         # 職業フィルター
#         if '職業' in df.columns:
#             job_options = df['職業'].dropna().unique().tolist()
#             if 'selected_jobs' not in st.session_state:
#                 st.session_state.selected_jobs = job_options
#             selected_jobs = st.multiselect(
#                 "🔍 職業", options=job_options, default=st.session_state.selected_jobs
#             )
#             st.session_state.selected_jobs = selected_jobs
#         else:
#             selected_jobs = None
#     
#         if '平均購入単価' in df.columns:
#             min_price = int(df['平均購入単価'].min())
#             max_price = int(df['平均購入単価'].max())
#             selected_average_bands = st.slider("🔍 平均購入単価", min_price, max_price, (min_price, max_price))
#         else:
#             selected_average_bands = None
#     
#         if 'SNS利用時間' in df.columns:
#             min_sns = int(df['SNS利用時間'].min())
#             max_sns = int(df['SNS利用時間'].max())
#             selected_sns = st.slider("🔍 SNS利用時間", min_sns, max_sns, (min_sns, max_sns))
#         else:
#             selected_sns = None
#     
#     with col2:
#         # キャラ傾向フィルター
#         if 'キャラ傾向' in df.columns:
#             char_options = df['キャラ傾向'].dropna().unique().tolist()
#             if 'selected_character' not in st.session_state:
#                 st.session_state.selected_character = char_options
#             selected_character = st.multiselect(
#                 "🔍 キャラ傾向", options=char_options, default=st.session_state.selected_character
#             )
#             st.session_state.selected_character = selected_character
#         else:
#             selected_character = None
#     
#         # 重要視すること1 フィルター
#         if '重要視すること' in df.columns:
#             imp_options = df['重要視すること'].dropna().unique().tolist()
#             if 'selected_importance' not in st.session_state:
#                 st.session_state.selected_importance = imp_options
#             selected_importance = st.multiselect(
#                 "🔍 重要視すること", options=imp_options, default=st.session_state.selected_importance
#             )
#             st.session_state.selected_importance = selected_importance
#         else:
#             selected_importance = None
#     
#         # 購買頻度フィルター
#         if '購買頻度' in df.columns:
#             freq_options = df['購買頻度'].dropna().unique().tolist()
#             if 'selected_frequency' not in st.session_state:
#                 st.session_state.selected_frequency = freq_options
#             selected_frequency = st.multiselect(
#                 "🔍 購買頻度", options=freq_options, default=st.session_state.selected_frequency
#             )
#             st.session_state.selected_frequency = selected_frequency
#         else:
#             selected_frequency = None
#     
#     with col3:
#         if '購入スタイル' in df.columns:
#             style_options = df['購入スタイル'].dropna().unique().tolist()
#             st.markdown("🔍 購入スタイル")
#     
#             if "selected_style" not in st.session_state:
#                 st.session_state.selected_style = {s: True for s in style_options}
#     
#             colA, colB = st.columns(2)
#             with colA:
#                 if st.button("✅ 全て選択"):
#                     for s in style_options:
#                         st.session_state.selected_style[s] = True
#             with colB:
#                 if st.button("❌ 全て解除"):
#                     for s in style_options:
#                         st.session_state.selected_style[s] = False
#     
#             selected_style = []
#             for s in style_options:
#                 checked = st.checkbox(s, value=st.session_state.selected_style[s])
#                 st.session_state.selected_style[s] = checked
#                 if checked:
#                     selected_style.append(s)
#         else:
#             selected_style = None
#             
#     # ================
#     # フィルター処理
#     # ================
#     filtered_df = df.copy()
#     if selected_age_range:
#         filtered_df = filtered_df[filtered_df['年齢'].between(*selected_age_range)]
#     if selected_gender:
#         filtered_df = filtered_df[filtered_df['性別'].isin(selected_gender)]
#     if selected_jobs:
#         filtered_df = filtered_df[filtered_df['職業'].isin(selected_jobs)]
#     if selected_character:
#         filtered_df = filtered_df[filtered_df['キャラ傾向'].isin(selected_character)]
#     if selected_frequency:
#         filtered_df = filtered_df[filtered_df['購買頻度'].isin(selected_frequency)]
#     if selected_style:
#         filtered_df = filtered_df[filtered_df['購入スタイル'].isin(selected_style)]
#     if selected_importance:
#         filtered_df = filtered_df[filtered_df['重要視すること'].isin(selected_importance)]
#     if selected_sns:
#         filtered_df = filtered_df[filtered_df['SNS利用時間'].between(*selected_sns)]
#     if selected_average_bands:
#         filtered_df = filtered_df[filtered_df['平均購入単価'].between(*selected_average_bands)]
#
#     st.markdown(f"### フィルター後の対象者数: {len(filtered_df)} 人")
#
#     # ======================
#     # クラスタリング
#     # ======================
#     st.markdown("#### 🧩 クラスタリング設定")
#
#     # クラスタリングに使う特徴量選択UI（カテゴリはラベルエンコーディング）
#     candidate_features = ['年齢', '性別','職業','購買頻度','平均購入単価', '購入スタイル', '重要視すること']
#     selected_features = st.multiselect("クラスタリングに使う属性を選択してください", candidate_features, default=candidate_features)
#
#     if len(selected_features) == 0:
#         st.warning("少なくとも1つ以上の特徴量を選択してください。")
#     else:
#         cluster_count = st.slider("クラスタ数 (K)", 2, 10, 3)
#
#         # 前処理：カテゴリ変数はlabel encoding（簡易版）
#         from sklearn.preprocessing import LabelEncoder
#         X = filtered_df[selected_features].copy()
#
#         for col in X.columns:
#             if X[col].dtype == 'object' or X[col].dtype.name == 'category':
#                 le = LabelEncoder()
#                 X[col] = le.fit_transform(X[col].astype(str))
#
#         # 標準化
#         scaler = StandardScaler()
#         X_scaled = scaler.fit_transform(X)
#
#         # KMeansクラスタリング
#         kmeans = KMeans(n_clusters=cluster_count, random_state=42)
#         clusters = kmeans.fit_predict(X_scaled)
#
#         filtered_df['cluster'] = clusters
#
#         st.markdown(f"### クラスタリング結果（K={cluster_count}）")
#         st.write(filtered_df[['ID'] + selected_features + ['cluster']])
#
#         # クラスタごとの人数表示
# #        st.markdown("#### クラスタ人数分布")
# #        cluster_counts = filtered_df['cluster'].value_counts().sort_index()
# #        st.bar_chart(cluster_counts)
#
#
#         # クラスタ人数分布の下に追加
#         st.markdown("#### クラスタ内容")
#     
#         num_clusters = filtered_df['cluster'].nunique()
#         
#         num_features = ['年齢', 'SNS利用時間', '平均購入単価']
#         cat_features = ['性別', '職業', '購買頻度',"重要視すること","購入スタイル", 'キャラ傾向']
#
#         rows = []
#         
#         for c in range(num_clusters):
#             cluster_df = filtered_df[filtered_df['cluster'] == c]
#             n = len(cluster_df)
#             
#             row = {"クラスタ": c, "人数": n}
#             
#             # 数値特徴量の平均値
#             for f in num_features:
#                 if f in cluster_df.columns:
#                     row[f"{f}平均"] = round(cluster_df[f].mean(), 2)
#             
#             # カテゴリ特徴量は最頻値と割合を表示（例）
#             for f in cat_features:
#                 if f in cluster_df.columns:
#                     top_val = cluster_df[f].value_counts(normalize=True).idxmax()
#                     top_ratio = cluster_df[f].value_counts(normalize=True).max()
#                     row[f"{f}（最多）"] = f"{top_val} ({top_ratio:.1%})"
#             
#             rows.append(row)
#         
#         summary_df = pd.DataFrame(rows)
#         
#         st.dataframe(summary_df)
#
#
#
#
#
#
#         
#         # =======================
#         # クラスタ別PSM分析表示
#         # =======================
#         st.markdown("#### 📊 クラスタ別 Van Westendorp PSM分析")
#
#         labels = ['too_cheap', 'cheap', 'expensive', 'too_expensive']
#         brands = sorted(set(col.split('_')[0] for col in df.columns if any(lbl in col for lbl in labels)))
#
#         cluster_tabs = st.tabs([f"Cluster {i}" for i in range(cluster_count)])
#
#         for cluster_id, tab in zip(range(cluster_count), cluster_tabs):
#             with tab:
#                 st.markdown(f"##### Cluster {cluster_id} の分析")
#
#                 df_cluster = filtered_df[filtered_df['cluster'] == cluster_id]
#
#                 for brand in brands:
#                     brand_cols = [f"{brand}_{lbl}" for lbl in labels if f"{brand}_{lbl}" in df_cluster.columns]
#                     df_brand = df_cluster[df_cluster[brand_cols].notnull().any(axis=1)]
#
#                     if df_brand.empty:
#                         st.warning(f"{brand} のデータがありません。")
#                         continue
#
#                     price_data = {
#                         label: df_brand[f"{brand}_{label}"].dropna().astype(int).values
#                         for label in labels if f"{brand}_{label}" in df_brand.columns
#                     }
#
#                     valid_arrays = [arr for arr in price_data.values() if len(arr) > 0]
#                     if not valid_arrays:
#                         st.warning("有効な価格データがありません。")
#                         continue
#
#                     all_prices = np.arange(
#                         min(np.concatenate(valid_arrays)),
#                         max(np.concatenate(valid_arrays)) + 1000,
#                         100
#                     )
#                     n = len(df_brand)
#
#                     cumulative = {
#                         'too_cheap': [np.sum(price_data.get('too_cheap', []) >= p) / n for p in all_prices],
#                         'cheap': [np.sum(price_data.get('cheap', []) >= p) / n for p in all_prices],
#                         'expensive': [np.sum(price_data.get('expensive', []) <= p) / n for p in all_prices],
#                         'too_expensive': [np.sum(price_data.get('too_expensive', []) <= p) / n for p in all_prices],
#                     }
#
#                     opp = find_intersection(cumulative['cheap'], cumulative['expensive'], all_prices)
#                     idp = find_intersection(cumulative['too_cheap'], cumulative['too_expensive'], all_prices)
#                     pme = find_intersection(cumulative['cheap'], cumulative['too_expensive'], all_prices)
#                     pmc = find_intersection(cumulative['expensive'], cumulative['too_cheap'], all_prices)
#
#                     fig = go.Figure()
#                                         
#                     fig.add_trace(go.Scatter(x=all_prices, y=np.array(cumulative['too_cheap'])*100, name='Too Cheap', line=dict(color='blue')))
#                     fig.add_trace(go.Scatter(x=all_prices, y=np.array(cumulative['cheap'])*100, name='Cheap', line=dict(color='green')))
#                     fig.add_trace(go.Scatter(x=all_prices, y=np.array(cumulative['expensive'])*100, name='Expensive', line=dict(color='orange')))
#                     fig.add_trace(go.Scatter(x=all_prices, y=np.array(cumulative['too_expensive'])*100, name='Too Expensive', line=dict(color='red')))
#
#                     # 補助線とラベル表示
#                     for val, name, color in zip(
#                         [opp, idp, pme, pmc],
#                         ['OPP（最適）', 'IDP（無関心）', 'PME（上限）', 'PMC（下限）'],
#                         ['purple', 'black', 'magenta', 'cyan']
#                     ):
#                         if val:
#                             fig.add_vline(x=val, line_dash='dash', line_color=color)
#                             fig.add_annotation(x=val, y=50, text=name, showarrow=False, textangle=90,
#                                                font=dict(color=color, size=12), bgcolor='white')
#
#
#
#                     # ... add traces and annotations ...
#                     
#                     fig.update_layout(
#                         title=f"{brand} - PSM分析",
#                         xaxis_title="価格（円）",
#                         yaxis_title="累積比率（%）",
#                         height=400,
#                         hovermode="x unified",
#                         xaxis=dict(tickformat='d')
#                     )
#                     
#                     # グラフと指標を横並びに表示
#                     col_plot, col_info = st.columns([3, 1])
#                     with col_plot:
#                         st.plotly_chart(fig, use_container_width=True)
#                     with col_info:
#                         st.markdown("#### 👇 指標")
#                         st.markdown(f"**{brand} の該当人数：{df_brand.shape[0]}人**")
#                         st.write(f"📌 **最適価格（OPP）**: {round(opp) if opp else '計算不可'} 円")
#                         st.write(f"📌 **無関心価格（IDP）**: {round(idp) if idp else '計算不可'} 円")
#                         st.write(f"📌 **価格受容範囲下限（PMC）**: {round(pmc) if pmc else '計算不可'} 円")
#                         st.write(f"📌 **価格受容範囲上限（PME）**: {round(pme) if pme else '計算不可'} 円")
#
#
#
#
#         
#         # ---- 全体集計（フィルター前）----
#         results_before_filter = []
#         for brand in brands:
#             brand_cols = [f"{brand}_{lbl}" for lbl in labels if f"{brand}_{lbl}" in df.columns]
#             df_brand = df[df[brand_cols].notnull().any(axis=1)]
#             if df_brand.empty:
#                 continue
#         
#             price_data = {
#                 label: df_brand[f"{brand}_{label}"].dropna().astype(int).values
#                 for label in labels if f"{brand}_{label}" in df_brand.columns
#             }
#         
#             valid_arrays = [arr for arr in price_data.values() if len(arr) > 0]
#             if not valid_arrays:
#                 continue
#         
#             all_prices = np.arange(
#                 min(np.concatenate(valid_arrays)),
#                 max(np.concatenate(valid_arrays)) + 1000,
#                 100
#             )
#             n = len(df_brand)
#         
#             cumulative = {
#                 'too_cheap': [np.sum(price_data.get('too_cheap', []) >= p) / n for p in all_prices],
#                 'cheap': [np.sum(price_data.get('cheap', []) >= p) / n for p in all_prices],
#                 'expensive': [np.sum(price_data.get('expensive', []) <= p) / n for p in all_prices],
#                 'too_expensive': [np.sum(price_data.get('too_expensive', []) <= p) / n for p in all_prices],
#             }
#         
#             opp = find_intersection(cumulative['cheap'], cumulative['expensive'], all_prices)
#             idp = find_intersection(cumulative['too_cheap'], cumulative['too_expensive'], all_prices)
#             pme = find_intersection(cumulative['cheap'], cumulative['too_expensive'], all_prices)
#             pmc = find_intersection(cumulative['expensive'], cumulative['too_cheap'], all_prices)
#         
#             results_before_filter.append({
#                 "ブランド": brand,
#                 "OPP": opp,
#                 "IDP": idp,
#                 "PMC": pmc,
#                 "PME": pme
#             })
#         
#         # ---- 表示部 ----
#         st.markdown("---")
#         with st.expander("📋 フィルター前 ブランド別 PSM 指標一覧（全体集計）", expanded=False):
#             st.markdown(f"**全体調査人数：{len(df)}人**")
#             summary_df_before = pd.DataFrame(results_before_filter)
#             st.dataframe(summary_df_before.style.format({col: "{:.0f}" for col in summary_df_before.columns if col != "ブランド"}))
#         
#
#
# else:
#     st.info("CSVファイルをアップロードしてください。")


# %%

# %% [markdown]
# イマイチ

# %% [markdown] jupyter={"source_hidden": true}
# import streamlit as st
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler
# from sklearn.cluster import KMeans
# import plotly.graph_objects as go
# from scipy.interpolate import interp1d
#
# st.set_page_config(layout="wide")
# st.title("💴 Van Westendorp PSM + クラスタリング分析アプリ")
#
# @st.cache_data
# def load_data(uploaded_file):
#     return pd.read_csv(uploaded_file)
#
# def find_intersection(y1, y2, x):
#     diff = np.array(y1) - np.array(y2)
#     sign_change = np.where(np.diff(np.sign(diff)) != 0)[0]
#     if len(sign_change) == 0:
#         return None
#     i = sign_change[0]
#     try:
#         f = interp1d(diff[i:i+2], x[i:i+2])
#         return float(f(0))
#     except Exception:
#         return None
#
# uploaded_file = st.file_uploader("📂 CSVファイルをアップロード", type=["csv"])
#
# if uploaded_file is not None:
#     df = load_data(uploaded_file)
#
#     st.markdown("#### 🔍 絞り込みフィルター")
#     col1, col2, col3 = st.columns(3)
#
#     with col1:
#         if '年齢' in df.columns:
#             min_age = int(df['年齢'].min())
#             max_age = int(df['年齢'].max())
#             selected_age_range = st.slider("年齢範囲", min_age, max_age, (min_age, max_age))
#         else:
#             selected_age_range = None
#
#         if '性別' in df.columns:
#             gender_options = df['性別'].dropna().unique().tolist()
#             selected_gender = st.multiselect("性別", gender_options, default=gender_options)
#         else:
#             selected_gender = None
#
#
#         if '重要視すること' in df.columns:
#             imp_options = df['重要視すること'].dropna().unique().tolist()
#             selected_importance = st.multiselect("重要視すること", imp_options, default=imp_options)
#         else:
#             selected_importance = None
#
#     
#     with col2:
#         if '購入スタイル' in df.columns:
#             style_options = df['購入スタイル'].dropna().unique().tolist()
#             st.markdown("🔍 購入スタイル")
#         
#             if "selected_style" not in st.session_state:
#                 st.session_state.selected_style = {s: True for s in style_options}
#         
#             colA, colB = st.columns(2)
#             with colA:
#                 if st.button("✅ 全て選択", key="style_select_all"):
#                     for s in style_options:
#                         st.session_state.selected_style[s] = True
#             with colB:
#                 if st.button("❌ 全て解除", key="style_deselect_all"):  # 🔁 keyを変更
#                     for s in style_options:
#                         st.session_state.selected_style[s] = False
#         
#             selected_style = []
#             for s in style_options:
#                 checked = st.checkbox(s, value=st.session_state.selected_style[s])
#                 st.session_state.selected_style[s] = checked
#                 if checked:
#                     selected_style.append(s)
#         else:
#             selected_style = None
#         
#         with col3:
#         
#             if '生活で大切なこと' in df.columns:
#                 life_options = df['生活で大切なこと'].dropna().unique().tolist()
#                 st.markdown("🔍 生活で大切なこと")
#         
#                 if "selected_life" not in st.session_state:
#                     st.session_state.selected_life = {s: True for s in life_options}
#         
#                 colA, colB = st.columns(2)
#                 with colA:
#                     if st.button("✅ 全て選択", key="life_select_all"):
#                         for s in life_options:
#                             st.session_state.selected_life[s] = True
#                 with colB:
#                     if st.button("❌ 全て解除", key="life_deselect_all"):  # 🔁 keyを変更
#                         for s in life_options:  # 🔁 ss → s に修正
#                             st.session_state.selected_life[s] = False
#         
#                 selected_life = []
#                 for s in life_options:
#                     checked = st.checkbox(s, value=st.session_state.selected_life[s])
#                     st.session_state.selected_life[s] = checked
#                     if checked:
#                         selected_life.append(s)
#             else:
#                 selected_life = None
#
#
#             
#     # ================
#     # フィルター処理
#     # ================
#     filtered_df = df.copy()
#     if selected_age_range:
#         filtered_df = filtered_df[filtered_df['年齢'].between(*selected_age_range)]
#     if selected_gender:
#         filtered_df = filtered_df[filtered_df['性別'].isin(selected_gender)]
#     if selected_life:
#         filtered_df = filtered_df[filtered_df['生活で大切なこと'].isin(selected_life)]
#     if selected_style:
#         filtered_df = filtered_df[filtered_df['購入スタイル'].isin(selected_style)]
#     if selected_importance:
#         filtered_df = filtered_df[filtered_df['重要視すること'].isin(selected_importance)]
#
#     st.markdown(f"### フィルター後の対象者数: {len(filtered_df)} 人")
#
#     # ======================
#     # クラスタリング
#     # ======================
#     st.markdown("#### 🧩 クラスタリング設定")
#
#     # クラスタリングに使う特徴量選択UI（カテゴリはラベルエンコーディング）
#     candidate_features = ['年齢', '性別', '生活で大切なこと', '購入スタイル', '重要視すること']
#     selected_features = st.multiselect("クラスタリングに使う属性を選択してください", candidate_features, default=['年齢', '性別',  '生活で大切なこと', '購入スタイル', '重要視すること'])
#
#     if len(selected_features) == 0:
#         st.warning("少なくとも1つ以上の特徴量を選択してください。")
#     else:
#         cluster_count = st.slider("クラスタ数 (K)", 2, 10, 3)
#
#         # 前処理：カテゴリ変数はlabel encoding（簡易版）
#         from sklearn.preprocessing import LabelEncoder
#         X = filtered_df[selected_features].copy()
#
#         for col in X.columns:
#             if X[col].dtype == 'object' or X[col].dtype.name == 'category':
#                 le = LabelEncoder()
#                 X[col] = le.fit_transform(X[col].astype(str))
#
#         # 標準化
#         scaler = StandardScaler()
#         X_scaled = scaler.fit_transform(X)
#
#         # KMeansクラスタリング
#         kmeans = KMeans(n_clusters=cluster_count, random_state=42)
#         clusters = kmeans.fit_predict(X_scaled)
#
#         filtered_df['cluster'] = clusters
#
#         st.markdown(f"### クラスタリング結果（K={cluster_count}）")
#         st.write(filtered_df[['ID'] + selected_features + ['cluster']])
#
#         # クラスタごとの人数表示
# #        st.markdown("#### クラスタ人数分布")
# #        cluster_counts = filtered_df['cluster'].value_counts().sort_index()
# #        st.bar_chart(cluster_counts)
#
#
#         # クラスタ人数分布の下に追加
#         st.markdown("#### クラスタ内容")
#     
#         num_clusters = filtered_df['cluster'].nunique()
#         
#         num_features = ['年齢', 'SNS利用時間', '平均購入単価']
#         cat_features = ['性別', '職業', '購買頻度', 'キャラ傾向',"重要視すること1","購入スタイル1"]
#
#         rows = []
#         
#         for c in range(num_clusters):
#             cluster_df = filtered_df[filtered_df['cluster'] == c]
#             n = len(cluster_df)
#             
#             row = {"クラスタ": c, "人数": n}
#             
#             # 数値特徴量の平均値
#             for f in num_features:
#                 if f in cluster_df.columns:
#                     row[f"{f}平均"] = round(cluster_df[f].mean(), 2)
#             
#             # カテゴリ特徴量は最頻値と割合を表示（例）
#             for f in cat_features:
#                 if f in cluster_df.columns:
#                     top_val = cluster_df[f].value_counts(normalize=True).idxmax()
#                     top_ratio = cluster_df[f].value_counts(normalize=True).max()
#                     row[f"{f}（最多）"] = f"{top_val} ({top_ratio:.1%})"
#             
#             rows.append(row)
#         
#         summary_df = pd.DataFrame(rows)
#         
#         st.dataframe(summary_df)
#
#
#
#
#
#
#         
#         # =======================
#         # クラスタ別PSM分析表示
#         # =======================
#         st.markdown("#### 📊 クラスタ別 Van Westendorp PSM分析")
#
#         labels = ['too_cheap', 'cheap', 'expensive', 'too_expensive']
#         brands = sorted(set(col.split('_')[0] for col in df.columns if any(lbl in col for lbl in labels)))
#
#         cluster_tabs = st.tabs([f"Cluster {i}" for i in range(cluster_count)])
#
#         for cluster_id, tab in zip(range(cluster_count), cluster_tabs):
#             with tab:
#                 st.markdown(f"##### Cluster {cluster_id} の分析")
#
#                 df_cluster = filtered_df[filtered_df['cluster'] == cluster_id]
#
#                 for brand in brands:
#                     brand_cols = [f"{brand}_{lbl}" for lbl in labels if f"{brand}_{lbl}" in df_cluster.columns]
#                     df_brand = df_cluster[df_cluster[brand_cols].notnull().any(axis=1)]
#
#                     if df_brand.empty:
#                         st.warning(f"{brand} のデータがありません。")
#                         continue
#
#                     price_data = {
#                         label: df_brand[f"{brand}_{label}"].dropna().astype(int).values
#                         for label in labels if f"{brand}_{label}" in df_brand.columns
#                     }
#
#                     valid_arrays = [arr for arr in price_data.values() if len(arr) > 0]
#                     if not valid_arrays:
#                         st.warning("有効な価格データがありません。")
#                         continue
#
#                     all_prices = np.arange(
#                         min(np.concatenate(valid_arrays)),
#                         max(np.concatenate(valid_arrays)) + 1000,
#                         100
#                     )
#                     n = len(df_brand)
#
#                     cumulative = {
#                         'too_cheap': [np.sum(price_data.get('too_cheap', []) >= p) / n for p in all_prices],
#                         'cheap': [np.sum(price_data.get('cheap', []) >= p) / n for p in all_prices],
#                         'expensive': [np.sum(price_data.get('expensive', []) <= p) / n for p in all_prices],
#                         'too_expensive': [np.sum(price_data.get('too_expensive', []) <= p) / n for p in all_prices],
#                     }
#
#                     opp = find_intersection(cumulative['cheap'], cumulative['expensive'], all_prices)
#                     idp = find_intersection(cumulative['too_cheap'], cumulative['too_expensive'], all_prices)
#                     pme = find_intersection(cumulative['cheap'], cumulative['too_expensive'], all_prices)
#                     pmc = find_intersection(cumulative['expensive'], cumulative['too_cheap'], all_prices)
#
#                     fig = go.Figure()
#                     fig.add_trace(go.Scatter(x=all_prices, y=np.array(cumulative['too_cheap'])*100, name='Too Cheap', line=dict(color='blue')))
#                     fig.add_trace(go.Scatter(x=all_prices, y=np.array(cumulative['cheap'])*100, name='Cheap', line=dict(color='green')))
#                     fig.add_trace(go.Scatter(x=all_prices, y=np.array(cumulative['expensive'])*100, name='Expensive', line=dict(color='orange')))
#                     fig.add_trace(go.Scatter(x=all_prices, y=np.array(cumulative['too_expensive'])*100, name='Too Expensive', line=dict(color='red')))
#
#                     # 補助線とラベル表示
#                     for val, name, color in zip(
#                         [opp, idp, pme, pmc],
#                         ['OPP（最適）', 'IDP（無関心）', 'PME（上限）', 'PMC（下限）'],
#                         ['purple', 'black', 'magenta', 'cyan']
#                     ):
#                         if val:
#                             fig.add_vline(x=val, line_dash='dash', line_color=color)
#                             fig.add_annotation(x=val, y=50, text=name, showarrow=False, textangle=90,
#                                                font=dict(color=color, size=12), bgcolor='white')
#
#                     fig.update_layout(
#                         title=f"{brand} - PSM分析",
#                         xaxis_title="価格（円）",
#                         yaxis_title="累積比率（%）",
#                         height=400,
#                         hovermode="x unified",
#                         xaxis=dict(tickformat='d')
#                     )
#
#                     st.plotly_chart(fig, use_container_width=True)
#
# else:
#     st.info("CSVファイルをアップロードしてください。")


# %%

# %%

# %%

# %%

# %% [markdown]
# 🤩　クラスタリング凄いよ

# %% [markdown] jupyter={"source_hidden": true}
# import streamlit as st
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler
# from sklearn.cluster import KMeans
# import plotly.graph_objects as go
# from scipy.interpolate import interp1d
#
# st.set_page_config(layout="wide")
# st.title("💴 Van Westendorp PSM + クラスタリング分析アプリ")
#
# @st.cache_data
# def load_data(uploaded_file):
#     return pd.read_csv(uploaded_file)
#
# def find_intersection(y1, y2, x):
#     diff = np.array(y1) - np.array(y2)
#     sign_change = np.where(np.diff(np.sign(diff)) != 0)[0]
#     if len(sign_change) == 0:
#         return None
#     i = sign_change[0]
#     try:
#         f = interp1d(diff[i:i+2], x[i:i+2])
#         return float(f(0))
#     except Exception:
#         return None
#
# uploaded_file = st.file_uploader("📂 CSVファイルをアップロード", type=["csv"])
#
# if uploaded_file is not None:
#     df = load_data(uploaded_file)
#
#     st.markdown("#### 🔍 絞り込みフィルター")
#     col1, col2 = st.columns(2)
#
#     with col1:
#         if '年齢' in df.columns:
#             min_age = int(df['年齢'].min())
#             max_age = int(df['年齢'].max())
#             selected_age_range = st.slider("年齢範囲", min_age, max_age, (min_age, max_age))
#         else:
#             selected_age_range = None
#
#         if '性別' in df.columns:
#             gender_options = df['性別'].dropna().unique().tolist()
#             selected_gender = st.multiselect("性別", gender_options, default=gender_options)
#         else:
#             selected_gender = None
#
#         if '職業' in df.columns:
#             job_options = df['職業'].dropna().unique().tolist()
#             selected_jobs = st.multiselect("職業", job_options, default=job_options)
#         else:
#             selected_jobs = None
#
#     with col2:
#         if '購買頻度' in df.columns:
#             freq_options = df['購買頻度'].dropna().unique().tolist()
#             selected_frequency = st.multiselect("購買頻度", freq_options, default=freq_options)
#         else:
#             selected_frequency = None
#
#         if '重要視すること1' in df.columns:
#             imp_options = df['重要視すること1'].dropna().unique().tolist()
#             selected_importance = st.multiselect("重要視すること1", imp_options, default=imp_options)
#         else:
#             selected_importance = None
#
#     # ================
#     # フィルター処理
#     # ================
#     filtered_df = df.copy()
#     if selected_age_range:
#         filtered_df = filtered_df[filtered_df['年齢'].between(*selected_age_range)]
#     if selected_gender:
#         filtered_df = filtered_df[filtered_df['性別'].isin(selected_gender)]
#     if selected_jobs:
#         filtered_df = filtered_df[filtered_df['職業'].isin(selected_jobs)]
#     if selected_frequency:
#         filtered_df = filtered_df[filtered_df['購買頻度'].isin(selected_frequency)]
#     if selected_importance:
#         filtered_df = filtered_df[filtered_df['重要視すること1'].isin(selected_importance)]
#
#     st.markdown(f"### フィルター後の対象者数: {len(filtered_df)} 人")
#
#     # ======================
#     # クラスタリング
#     # ======================
#     st.markdown("#### 🧩 クラスタリング設定")
#
#     # クラスタリングに使う特徴量選択UI（カテゴリはラベルエンコーディング）
#     candidate_features = ['年齢', '性別', '職業', '購買頻度', '重要視すること1']
#     selected_features = st.multiselect("クラスタリングに使う属性を選択してください", candidate_features, default=['年齢', '性別', '購買頻度'])
#
#     if len(selected_features) == 0:
#         st.warning("少なくとも1つ以上の特徴量を選択してください。")
#     else:
#         cluster_count = st.slider("クラスタ数 (K)", 2, 10, 3)
#
#         # 前処理：カテゴリ変数はlabel encoding（簡易版）
#         from sklearn.preprocessing import LabelEncoder
#         X = filtered_df[selected_features].copy()
#
#         for col in X.columns:
#             if X[col].dtype == 'object' or X[col].dtype.name == 'category':
#                 le = LabelEncoder()
#                 X[col] = le.fit_transform(X[col].astype(str))
#
#         # 標準化
#         scaler = StandardScaler()
#         X_scaled = scaler.fit_transform(X)
#
#         # KMeansクラスタリング
#         kmeans = KMeans(n_clusters=cluster_count, random_state=42)
#         clusters = kmeans.fit_predict(X_scaled)
#
#         filtered_df['cluster'] = clusters
#
#         st.markdown(f"### クラスタリング結果（K={cluster_count}）")
#         st.write(filtered_df[['ID'] + selected_features + ['cluster']])
#
#         # クラスタごとの人数表示
# #        st.markdown("#### クラスタ人数分布")
# #        cluster_counts = filtered_df['cluster'].value_counts().sort_index()
# #        st.bar_chart(cluster_counts)
#
#
#         # クラスタ人数分布の下に追加
#         st.markdown("#### クラスタ内容")
#     
#         num_clusters = filtered_df['cluster'].nunique()
#         
#         num_features = ['年齢', 'SNS利用時間', '平均購入単価']
#         cat_features = ['性別', '職業', '購買頻度', 'キャラ傾向',"重要視すること1","購入スタイル1"]
#
#         rows = []
#         
#         for c in range(num_clusters):
#             cluster_df = filtered_df[filtered_df['cluster'] == c]
#             n = len(cluster_df)
#             
#             row = {"クラスタ": c, "人数": n}
#             
#             # 数値特徴量の平均値
#             for f in num_features:
#                 if f in cluster_df.columns:
#                     row[f"{f}平均"] = round(cluster_df[f].mean(), 2)
#             
#             # カテゴリ特徴量は最頻値と割合を表示（例）
#             for f in cat_features:
#                 if f in cluster_df.columns:
#                     top_val = cluster_df[f].value_counts(normalize=True).idxmax()
#                     top_ratio = cluster_df[f].value_counts(normalize=True).max()
#                     row[f"{f}（最多）"] = f"{top_val} ({top_ratio:.1%})"
#             
#             rows.append(row)
#         
#         summary_df = pd.DataFrame(rows)
#         
#         st.dataframe(summary_df)
#
#
#
#
#
#
#         
#         # =======================
#         # クラスタ別PSM分析表示
#         # =======================
#         st.markdown("#### 📊 クラスタ別 Van Westendorp PSM分析")
#
#         labels = ['too_cheap', 'cheap', 'expensive', 'too_expensive']
#         brands = sorted(set(col.split('_')[0] for col in df.columns if any(lbl in col for lbl in labels)))
#
#         cluster_tabs = st.tabs([f"Cluster {i}" for i in range(cluster_count)])
#
#         for cluster_id, tab in zip(range(cluster_count), cluster_tabs):
#             with tab:
#                 st.markdown(f"##### Cluster {cluster_id} の分析")
#
#                 df_cluster = filtered_df[filtered_df['cluster'] == cluster_id]
#
#                 for brand in brands:
#                     brand_cols = [f"{brand}_{lbl}" for lbl in labels if f"{brand}_{lbl}" in df_cluster.columns]
#                     df_brand = df_cluster[df_cluster[brand_cols].notnull().any(axis=1)]
#
#                     if df_brand.empty:
#                         st.warning(f"{brand} のデータがありません。")
#                         continue
#
#                     price_data = {
#                         label: df_brand[f"{brand}_{label}"].dropna().astype(int).values
#                         for label in labels if f"{brand}_{label}" in df_brand.columns
#                     }
#
#                     valid_arrays = [arr for arr in price_data.values() if len(arr) > 0]
#                     if not valid_arrays:
#                         st.warning("有効な価格データがありません。")
#                         continue
#
#                     all_prices = np.arange(
#                         min(np.concatenate(valid_arrays)),
#                         max(np.concatenate(valid_arrays)) + 1000,
#                         100
#                     )
#                     n = len(df_brand)
#
#                     cumulative = {
#                         'too_cheap': [np.sum(price_data.get('too_cheap', []) >= p) / n for p in all_prices],
#                         'cheap': [np.sum(price_data.get('cheap', []) >= p) / n for p in all_prices],
#                         'expensive': [np.sum(price_data.get('expensive', []) <= p) / n for p in all_prices],
#                         'too_expensive': [np.sum(price_data.get('too_expensive', []) <= p) / n for p in all_prices],
#                     }
#
#                     opp = find_intersection(cumulative['cheap'], cumulative['expensive'], all_prices)
#                     idp = find_intersection(cumulative['too_cheap'], cumulative['too_expensive'], all_prices)
#                     pme = find_intersection(cumulative['cheap'], cumulative['too_expensive'], all_prices)
#                     pmc = find_intersection(cumulative['expensive'], cumulative['too_cheap'], all_prices)
#
#                     fig = go.Figure()
#                     fig.add_trace(go.Scatter(x=all_prices, y=np.array(cumulative['too_cheap'])*100, name='Too Cheap', line=dict(color='blue')))
#                     fig.add_trace(go.Scatter(x=all_prices, y=np.array(cumulative['cheap'])*100, name='Cheap', line=dict(color='green')))
#                     fig.add_trace(go.Scatter(x=all_prices, y=np.array(cumulative['expensive'])*100, name='Expensive', line=dict(color='orange')))
#                     fig.add_trace(go.Scatter(x=all_prices, y=np.array(cumulative['too_expensive'])*100, name='Too Expensive', line=dict(color='red')))
#
#                     # 補助線とラベル表示
#                     for val, name, color in zip(
#                         [opp, idp, pme, pmc],
#                         ['OPP（最適）', 'IDP（無関心）', 'PME（上限）', 'PMC（下限）'],
#                         ['purple', 'black', 'magenta', 'cyan']
#                     ):
#                         if val:
#                             fig.add_vline(x=val, line_dash='dash', line_color=color)
#                             fig.add_annotation(x=val, y=50, text=name, showarrow=False, textangle=90,
#                                                font=dict(color=color, size=12), bgcolor='white')
#
#                     fig.update_layout(
#                         title=f"{brand} - PSM分析",
#                         xaxis_title="価格（円）",
#                         yaxis_title="累積比率（%）",
#                         height=400,
#                         hovermode="x unified",
#                         xaxis=dict(tickformat='d')
#                     )
#
#                     st.plotly_chart(fig, use_container_width=True)
#
# else:
#     st.info("CSVファイルをアップロードしてください。")


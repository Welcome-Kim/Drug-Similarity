import streamlit as st
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit import DataStructs
import numpy as np

# 1. 페이지 설정 및 제목
st.set_page_config(page_title="의약품 구조 유사도 분석 시스템", page_icon="🧪", layout="wide")

# --- 스타일링 (CSS) ---
st.markdown("""
    <style>
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .drug-info-card { background-color: #f0f2f6; padding: 20px; border-radius: 10px; border-left: 5px solid #4a90e2; margin-bottom: 20px; }
    </style>
    """, unsafe_allow_html=True)

# 2. 데이터 로드 및 분석 엔진 초기화 (캐싱)
@st.cache_resource
def init_engine():
    try:
        # 데이터 로드 (파일명이 실제 파일과 일치하는지 확인하세요)
        df = pd.read_csv("Drug_Similarity.csv") 
        
        # Morgan Fingerprint 생성기 (Radius 2, 2048-bit)
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        
        # 서버 시작 시 전체 지문(Fingerprint) 미리 계산 (속도 최적화)
        df['fp'] = df['SMILES'].apply(lambda x: gen.GetFingerprint(Chem.MolFromSmiles(x)) if pd.notna(x) else None)
        return df, gen
    except Exception as e:
        st.error(f"데이터베이스 로드 중 오류가 발생했습니다: {e}")
        return None, None

df_master, mfpgen = init_engine()

# 3. 핵심 분석 함수 (1:N 전체 유사도 계산)
def get_top_similar_results(target_name, df, gen):
    # 선택한 약물 정보 가져오기
    target_row = df[df['ITEM_NAME'] == target_name].iloc[0]
    target_fp = target_row['fp']
    
    # 전체 약물과 1:1 타니모토 유사도 계산
    all_fps = df['fp'].tolist()
    similarities = DataStructs.BulkTanimotoSimilarity(target_fp, all_fps)
    
    res_df = df.copy()
    res_df['Similarity_Score'] = similarities
    
    # 자기 자신 제외 및 유사도 순 정렬
    res_df = res_df[res_df['ITEM_NAME'] != target_name]
    return res_df.sort_values(by='Similarity_Score', ascending=False), target_row

# --- UI 메인 섹션 ---
st.title("의약품 구조 유사도 검색 시스템")
st.caption("검색 약물과 화학 구조가 가장 유사한 대조군을 전체 데이터에서 추출합니다.")

if df_master is not None:
    # 검색창 섹션
    user_input = st.selectbox("분석 약물명을 선택하거나 입력하세요", 
                             options=df_master['ITEM_NAME'].unique(),
                             index=None,
                             placeholder="약물명 검색...")

    if user_input:
        with st.spinner('유사도 매칭 연산 중...'):
            # 1:N 유사도 분석 엔진 가동
            all_results, target_info = get_top_similar_results(user_input, df_master, mfpgen)
            
            # --- [화면 1] 검색 약물 상세 정보 카드 ---
            st.markdown(f"""
            <div class="drug-info-card">
                <h3>🔍 검색 약물 상세 정보</h3>
                <b>품목명:</b> {target_info['ITEM_NAME']} | <b>제약사:</b> {target_info['ENTP_NAME']} <br>
                <b>효능군:</b> {target_info['EFFECT_NAME']} <br>
                <b>성분명:</b> {target_info['INGR_NAME']} | <b>허가일:</b> {target_info['ITEM_PERMIT_DATE']}
            </div>
            """, unsafe_allow_html=True)

            # --- [화면 2] 결과 리스트 및 페이지네이션 ---
            st.write("---")
            items_per_page = 10
            total_results = len(all_results)
            total_pages = int(np.ceil(total_results / items_per_page))
            
            col_title, col_page = st.columns([3, 1])
            with col_title:
                st.subheader(f"'{user_input}'과(와) 구조가 유사한 약물 목록")
            with col_page:
                page = st.number_input(f"페이지 (총 {total_pages}P)", min_value=1, max_value=total_pages, step=1)

            # 데이터 슬라이싱 및 출력용 정제
            start_idx = (page - 1) * items_per_page
            end_idx = start_idx + items_per_page
            current_page_results = all_results.iloc[start_idx:end_idx].copy()

            # 사용할 컬럼만 필터링 (Similarity_Score 포함)
            display_cols = ['ITEM_NAME', 'ENTP_NAME', 'Similarity_Score', 'EFFECT_NAME', 'INGR_NAME', 'ITEM_PERMIT_DATE', 'FORM_NAME']
            final_table = current_page_results[display_cols]
            
            # 컬럼명 변경
            final_table.columns = ['참조 약물명', '제약', '구조 유사도', '효능군', '성분명', '허가일', '제형']

            # 데이터프레임 스타일링 및 출력
            st.dataframe(
                final_table.style.background_gradient(subset=['구조 유사도'], cmap='Blues'),
                use_container_width=True,
                hide_index=True
            )

            # --- [화면 3] 하단 요약 지표 및 가이드 ---
            st.write("---")
            top_val = all_results.iloc[0]['Similarity_Score']
            m1, m2, m3 = st.columns(3)
            m1.metric("최고 유사도", f"{top_val:.4f}")
            m2.metric("전체 분석 데이터", f"{len(df_master):,} 건")
            m3.metric("현재 순위 범위", f"{start_idx + 1}위 ~ {min(end_idx, total_results)}위")

            # 인사이트 가이드
            if top_val >= 0.8:
                st.warning(f"주의: 유사도가 매우 높은({top_val:.2f}) 대조군이 발견되었습니다. 해당 약물의 허가 자료 준용 여부를 검토하십시오.")
            else:
                st.info("검색된 약물들과의 구조적 유사성을 심사에 참고하십시오.")

else:
    st.warning("데이터베이스 연결 실패.")

"""
QMS 웹 대시보드 (Streamlit Cloud 배포용)
- GitHub의 JSON 데이터를 읽어서 표시
- DB 연결 없이 동작

배포: Streamlit Cloud에서 GitHub 레포 연결
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
import plotly.express as px

# ============================================================
# 설정
# ============================================================

# Gist Raw URL
GIST_RAW_BASE = "https://gist.githubusercontent.com/TRSTQuant/544f3dad6de1c3103f794017571e3c41/raw"

# 페이지 설정
st.set_page_config(
    layout='wide',
    page_title='QMS Dashboard',
    page_icon='📊'
)

st.markdown("""
    <style>
    .block-container { padding: 1rem; }
    footer { visibility: hidden; }
    </style>
""", unsafe_allow_html=True)


# ============================================================
# 데이터 로드
# ============================================================

@st.cache_data(ttl=300)  # 5분 캐시
def load_json(filename):
    """GitHub에서 JSON 데이터 로드"""
    url = f"{GIST_RAW_BASE}/{filename}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"데이터 로드 실패: {e}")
        return None


# ============================================================
# 페이지: 펀드 현황
# ============================================================

def render_fund_overview():
    st.header("📊 펀드 현황")

    data = load_json('fund_overview.json')
    if not data:
        return

    df = pd.DataFrame(data['funds'])
    st.caption(f"기준일: {data['date']} | 업데이트: {data['updated']}")

    # 요약 지표
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        total_nav = df['순자산'].sum()
        st.metric("총 순자산", f"{total_nav:,.0f}억원")
    with col2:
        avg_return = df['등락률'].mean()
        st.metric("평균 등락률", f"{avg_return:.2f}%")
    with col3:
        total_inflow = df['순설정'].sum()
        st.metric("순설정", f"{total_inflow:,.1f}억원")
    with col4:
        st.metric("펀드 수", f"{len(df)}개")

    st.divider()

    # 차트
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("자산 배분")
        fig = go.Figure()
        for col in ['주식(%)', 'ETF(%)', '지수선물(%)', '주선(%)']:
            fig.add_trace(go.Bar(name=col.replace('(%)', ''), x=df['펀드명'], y=df[col]))
        fig.update_layout(barmode='stack', height=400,
                          legend=dict(orientation='h', y=1.02),
                          margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("순자산 분포")
        fig = px.pie(df, values='순자산', names='펀드명', hole=0.4)
        fig.update_layout(height=400, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

    # 테이블
    st.subheader("펀드 상세")
    df_display = df[['FundCode', '펀드명', 'BM_NM', '순자산', '기준가', '등락률',
                     '주식(%)', 'ETF(%)', '지수선물(%)', '주선(%)', '순설정']].copy()
    df_display['순자산'] = df_display['순자산'].apply(lambda x: f"{x:,.0f}억")
    df_display['기준가'] = df_display['기준가'].apply(lambda x: f"{x:,.2f}")
    df_display['등락률'] = df_display['등락률'].apply(lambda x: f"{x:.2f}%")
    df_display['순설정'] = df_display['순설정'].apply(lambda x: f"{x:,.1f}억")

    st.dataframe(df_display, use_container_width=True, hide_index=True)

    # 펀드별 포트폴리오
    st.divider()
    st.subheader("펀드별 포트폴리오")

    portfolio_data = load_json('fund_portfolio.json')
    if portfolio_data:
        df_portfolio = pd.DataFrame(portfolio_data['portfolio'])

        # 펀드 선택
        fund_options = df[['FundCode', '펀드명']].apply(lambda x: f"{x['FundCode']} - {x['펀드명']}", axis=1).tolist()
        selected = st.selectbox("펀드 선택", fund_options, key="portfolio_fund")
        selected_fund_code = selected.split(' - ')[0]

        # 선택된 펀드의 포트폴리오
        df_fund_portfolio = df_portfolio[df_portfolio['FundCode'] == selected_fund_code].copy()

        if len(df_fund_portfolio) > 0:
            col1, col2 = st.columns([2, 1])

            with col1:
                # 포트폴리오 테이블
                df_port_display = df_fund_portfolio[['ComCode', 'ComName', 'Quantity', 'Price', 'Value', 'Weight']].copy()
                df_port_display.columns = ['종목코드', '종목명', '수량', '현재가', '평가금액', '비중(%)']
                df_port_display['수량'] = df_port_display['수량'].apply(lambda x: f"{x:,.0f}")
                df_port_display['현재가'] = df_port_display['현재가'].apply(lambda x: f"{x:,.0f}")
                df_port_display['평가금액'] = df_port_display['평가금액'].apply(lambda x: f"{x:,.0f}")
                df_port_display['비중(%)'] = df_port_display['비중(%)'].apply(lambda x: f"{x:.2f}")
                st.dataframe(df_port_display, use_container_width=True, hide_index=True, height=400)

            with col2:
                # 상위 10개 종목 파이차트
                top10 = df_fund_portfolio.nlargest(10, 'Weight')
                fig = px.pie(top10, values='Weight', names='ComName', hole=0.3,
                            title='상위 10개 종목')
                fig.update_layout(height=400, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig, use_container_width=True)

            st.caption(f"총 {len(df_fund_portfolio)}개 종목 | 기준일: {portfolio_data['date']}")
        else:
            st.info("해당 펀드의 포트폴리오 데이터가 없습니다.")


# ============================================================
# 페이지: 펀드 성과
# ============================================================

def render_fund_performance():
    st.header("📈 펀드 성과")

    data = load_json('fund_performance.json')
    if not data:
        return

    # 펀드 기본 정보 (BM 정보 포함)
    fund_data = load_json('fund_overview.json')
    fund_info = {}
    if fund_data:
        for f in fund_data['funds']:
            fund_info[f['FundCode']] = {'펀드명': f['펀드명'], 'BM_NM': f['BM_NM']}

    df = pd.DataFrame(data['performance'])
    st.caption(f"기간: {data['date_from']} ~ {data['date_to']} | 업데이트: {data['updated']}")

    # 펀드 선택 (펀드코드 - 펀드명 형식)
    fund_codes = df['FundCode'].unique().tolist()
    fund_options = [f"{fc} - {fund_info.get(fc, {}).get('펀드명', fc)}" for fc in fund_codes]
    selected_option = st.selectbox("펀드 선택", fund_options)
    selected_fund = selected_option.split(' - ')[0]

    # 선택된 펀드의 BM 정보 표시
    bm_name = fund_info.get(selected_fund, {}).get('BM_NM', '정보 없음')
    st.info(f"**BM**: {bm_name}")

    df_fund = df[df['FundCode'] == selected_fund].copy()
    df_fund = df_fund.sort_values('기준일자').reset_index(drop=True)

    # 누적 수익률 계산 (복리 방식)
    df_fund['CumRtnF'] = (1 + df_fund['RtnF'] / 100).cumprod() * 100 - 100
    df_fund['CumRtnB'] = (1 + df_fund['RtnB'] / 100).cumprod() * 100 - 100
    df_fund['CumExcess'] = df_fund['CumRtnF'] - df_fund['CumRtnB']

    # 요약 지표
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        cum_rtn_f = df_fund['CumRtnF'].iloc[-1]
        st.metric("펀드 수익률", f"{cum_rtn_f:.2f}%")
    with col2:
        cum_rtn_b = df_fund['CumRtnB'].iloc[-1]
        st.metric("BM 수익률", f"{cum_rtn_b:.2f}%")
    with col3:
        excess = cum_rtn_f - cum_rtn_b
        st.metric("초과 수익률", f"{excess:.2f}%")
    with col4:
        std = df_fund['RtnF'].std()
        sharpe = df_fund['RtnF'].mean() / std * np.sqrt(252) if std > 0 else 0
        st.metric("샤프 비율", f"{sharpe:.2f}")

    # 추가 지표
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        trading_days = len(df_fund)
        st.metric("거래일 수", f"{trading_days}일")
    with col2:
        win_rate = (df_fund['RtnF'] > 0).sum() / len(df_fund) * 100
        st.metric("승률", f"{win_rate:.1f}%")
    with col3:
        max_dd = (df_fund['CumRtnF'] - df_fund['CumRtnF'].cummax()).min()
        st.metric("최대 낙폭", f"{max_dd:.2f}%")
    with col4:
        volatility = std * np.sqrt(252)
        st.metric("연환산 변동성", f"{volatility:.2f}%")

    st.divider()

    # 누적 수익률 차트
    st.subheader("누적 수익률")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_fund['기준일자'], y=df_fund['CumRtnF'],
                              name='펀드', line=dict(color='blue', width=2)))
    fig.add_trace(go.Scatter(x=df_fund['기준일자'], y=df_fund['CumRtnB'],
                              name='BM', line=dict(color='gray', width=2)))
    fig.add_trace(go.Scatter(x=df_fund['기준일자'], y=df_fund['CumExcess'],
                              name='초과', line=dict(color='green', width=1, dash='dash')))
    fig.update_layout(height=400, legend=dict(orientation='h', y=1.02),
                      margin=dict(l=20, r=20, t=40, b=20),
                      yaxis_title='수익률 (%)')
    st.plotly_chart(fig, use_container_width=True)

    # 일간 수익률
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("일간 수익률 분포")
        fig = px.histogram(df_fund, x='RtnF', nbins=30, labels={'RtnF': '일간 수익률 (%)'})
        fig.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("펀드 vs BM 수익률")
        fig = px.scatter(df_fund, x='RtnB', y='RtnF',
                        labels={'RtnB': 'BM 수익률 (%)', 'RtnF': '펀드 수익률 (%)'})
        # 45도 기준선 추가
        min_val = min(df_fund['RtnB'].min(), df_fund['RtnF'].min())
        max_val = max(df_fund['RtnB'].max(), df_fund['RtnF'].max())
        fig.add_shape(type='line', x0=min_val, y0=min_val, x1=max_val, y1=max_val,
                     line=dict(color='red', dash='dash', width=1))
        fig.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    # 월별 수익률 테이블
    st.divider()
    st.subheader("월별 수익률")
    df_fund['YearMonth'] = pd.to_datetime(df_fund['기준일자']).dt.to_period('M').astype(str)
    monthly = df_fund.groupby('YearMonth').agg({
        'RtnF': lambda x: (1 + x/100).prod() * 100 - 100,
        'RtnB': lambda x: (1 + x/100).prod() * 100 - 100
    }).round(2)
    monthly['초과'] = (monthly['RtnF'] - monthly['RtnB']).round(2)
    monthly.columns = ['펀드(%)', 'BM(%)', '초과(%)']
    st.dataframe(monthly.T, use_container_width=True)


# ============================================================
# 페이지: 페어 트레이딩
# ============================================================

def render_pairs_trading():
    st.header("🔄 페어 트레이딩")

    data = load_json('pairs_list.json')
    if not data:
        return

    df = pd.DataFrame(data['pairs'])
    universe_codes = data.get('universe_codes', [])
    st.caption(f"기준일: {data['date']} | 업데이트: {data['updated']}")

    # 필터
    col1, col2, col3 = st.columns(3)
    with col1:
        pair_types = ['ALL'] + df['PairType'].unique().tolist()
        ptype = st.selectbox("페어타입", pair_types)
    with col2:
        univ_filters = ['전체', '롱(유니버스)', '숏(유니버스)', '롱&숏(유니버스)', '롱|숏(유니버스)']
        univ_filter = st.selectbox("유니버스 필터", univ_filters)
    with col3:
        signal_filters = ['전체', '롱(1)', '숏(-1)', '변경']
        signal_filter = st.selectbox("시그널 필터", signal_filters)

    # 필터 적용
    df_filtered = df.copy()

    if ptype != 'ALL':
        df_filtered = df_filtered[df_filtered['PairType'] == ptype]

    if univ_filter == '롱(유니버스)':
        df_filtered = df_filtered[df_filtered['Long_InUniv'] == True]
    elif univ_filter == '숏(유니버스)':
        df_filtered = df_filtered[df_filtered['Short_InUniv'] == True]
    elif univ_filter == '롱&숏(유니버스)':
        df_filtered = df_filtered[(df_filtered['Long_InUniv'] == True) & (df_filtered['Short_InUniv'] == True)]
    elif univ_filter == '롱|숏(유니버스)':
        df_filtered = df_filtered[(df_filtered['Long_InUniv'] == True) | (df_filtered['Short_InUniv'] == True)]

    if signal_filter == '롱(1)':
        df_filtered = df_filtered[df_filtered['Signal'] == 1]
    elif signal_filter == '숏(-1)':
        df_filtered = df_filtered[df_filtered['Signal'] == -1]
    elif signal_filter == '변경':
        df_filtered = df_filtered[df_filtered['SignalChange'] != 0]

    # 요약 지표
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("전체 페어", f"{len(df_filtered)}개")
    with col2:
        st.metric("롱 시그널", f"{len(df_filtered[df_filtered['Signal'] == 1])}개")
    with col3:
        st.metric("숏 시그널", f"{len(df_filtered[df_filtered['Signal'] == -1])}개")
    with col4:
        st.metric("시그널 변경", f"{len(df_filtered[df_filtered['SignalChange'] != 0])}개")

    st.divider()

    # 테이블
    st.subheader(f"페어 리스트 ({len(df_filtered)}개)")
    display_cols = ['PairType', 'ComName1', 'ComName2', 'TargetRatio', 'CloseRatio',
                    'SignalProb', 'Signal', 'SignalChange']
    df_display = df_filtered[display_cols].copy()
    df_display['TargetRatio'] = df_display['TargetRatio'].apply(lambda x: f"{x:.4f}")
    df_display['CloseRatio'] = df_display['CloseRatio'].apply(lambda x: f"{x:.4f}")
    df_display['SignalProb'] = df_display['SignalProb'].apply(lambda x: f"{x:.2f}")

    st.dataframe(df_display, use_container_width=True, hide_index=True)

    # 페어타입별 분포
    st.subheader("페어타입별 분포")
    df_ptype = df_filtered.groupby('PairType').agg(
        페어수=('ComCode1', 'count'),
        순시그널=('Signal', 'sum')
    ).reset_index()

    fig = px.bar(df_ptype, x='PairType', y='페어수', color='순시그널',
                 color_continuous_scale='RdYlGn', color_continuous_midpoint=0)
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# 페이지: 펀드 비교
# ============================================================

def render_fund_comparison():
    st.header("📊 펀드 비교")

    data = load_json('fund_performance.json')
    fund_data = load_json('fund_overview.json')
    if not data or not fund_data:
        return

    # 펀드 정보 매핑
    fund_info = {}
    for f in fund_data['funds']:
        fund_info[f['FundCode']] = {'펀드명': f['펀드명'], 'BM_NM': f['BM_NM']}

    df = pd.DataFrame(data['performance'])
    st.caption(f"기간: {data['date_from']} ~ {data['date_to']} | 업데이트: {data['updated']}")

    # 펀드 다중 선택
    fund_codes = df['FundCode'].unique().tolist()
    fund_options = [f"{fc} - {fund_info.get(fc, {}).get('펀드명', fc)}" for fc in fund_codes]
    selected_options = st.multiselect("펀드 선택 (복수 선택 가능)", fund_options, default=fund_options[:3])

    if not selected_options:
        st.warning("비교할 펀드를 선택하세요.")
        return

    selected_funds = [opt.split(' - ')[0] for opt in selected_options]

    # 각 펀드별 누적 수익률 계산
    cum_data = []
    for fc in selected_funds:
        df_fund = df[df['FundCode'] == fc].copy()
        df_fund = df_fund.sort_values('기준일자').reset_index(drop=True)
        df_fund['CumRtn'] = (1 + df_fund['RtnF'] / 100).cumprod() * 100 - 100
        fund_name = fund_info.get(fc, {}).get('펀드명', fc)
        for _, row in df_fund.iterrows():
            cum_data.append({
                '기준일자': row['기준일자'],
                'FundCode': fc,
                '펀드명': fund_name,
                '누적수익률': row['CumRtn']
            })

    df_cum = pd.DataFrame(cum_data)

    # 누적 수익률 비교 차트
    st.subheader("누적 수익률 비교")
    fig = px.line(df_cum, x='기준일자', y='누적수익률', color='펀드명',
                  labels={'누적수익률': '수익률 (%)'})
    fig.update_layout(height=450, legend=dict(orientation='h', y=-0.15),
                      margin=dict(l=20, r=20, t=20, b=80))
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # 성과 요약 테이블
    st.subheader("성과 요약")
    summary_data = []
    for fc in selected_funds:
        df_fund = df[df['FundCode'] == fc].copy()
        df_fund = df_fund.sort_values('기준일자').reset_index(drop=True)
        df_fund['CumRtnF'] = (1 + df_fund['RtnF'] / 100).cumprod() * 100 - 100
        df_fund['CumRtnB'] = (1 + df_fund['RtnB'] / 100).cumprod() * 100 - 100

        cum_rtn = df_fund['CumRtnF'].iloc[-1]
        bm_rtn = df_fund['CumRtnB'].iloc[-1]
        std = df_fund['RtnF'].std()
        sharpe = df_fund['RtnF'].mean() / std * np.sqrt(252) if std > 0 else 0
        win_rate = (df_fund['RtnF'] > 0).sum() / len(df_fund) * 100
        max_dd = (df_fund['CumRtnF'] - df_fund['CumRtnF'].cummax()).min()

        summary_data.append({
            '펀드코드': fc,
            '펀드명': fund_info.get(fc, {}).get('펀드명', fc),
            '수익률(%)': f"{cum_rtn:.2f}",
            'BM수익률(%)': f"{bm_rtn:.2f}",
            '초과(%)': f"{cum_rtn - bm_rtn:.2f}",
            '샤프비율': f"{sharpe:.2f}",
            '승률(%)': f"{win_rate:.1f}",
            '최대낙폭(%)': f"{max_dd:.2f}"
        })

    df_summary = pd.DataFrame(summary_data)
    st.dataframe(df_summary, use_container_width=True, hide_index=True)


# ============================================================
# 메인
# ============================================================

def main():
    st.sidebar.title("📊 QMS Dashboard")
    st.sidebar.markdown("---")

    menu = st.sidebar.radio(
        "메뉴",
        ["📊 펀드 현황", "📈 펀드 성과", "📊 펀드 비교", "🔄 페어 트레이딩"],
        label_visibility="collapsed"
    )

    st.sidebar.markdown("---")
    st.sidebar.caption("데이터: 매일 업데이트")

    if menu == "📊 펀드 현황":
        render_fund_overview()
    elif menu == "📈 펀드 성과":
        render_fund_performance()
    elif menu == "📊 펀드 비교":
        render_fund_comparison()
    elif menu == "🔄 페어 트레이딩":
        render_pairs_trading()


if __name__ == "__main__":
    main()

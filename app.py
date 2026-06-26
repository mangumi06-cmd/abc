import streamlit as st
import pyupbit
import pandas as pd
import matplotlib.pyplot as plt
import google.generativeai as genai


plt.rcParams['font.family'] = 'Malgun Gothic' 
plt.rcParams['axes.unicode_minus'] = False

def get_ai_analysis(all_results, use_rsi, use_bb, mode, rsi_buy, rsi_sell, bb_buy_pct, bb_sell_pct):
    
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        genai.configure(api_key=api_key)
        client = genai.GenerativeModel('gemini-2.5-flash') 
    except Exception as e:
        st.error(f"🔑 Gemini API 설정 중 오류가 발생했습니다: {e}")
        return None

    
    strategy_summary = f"""
    [사용자 설정 전략 스펙]
    - RSI 조건 사용 여부: {use_rsi} (매수: {rsi_buy} 이하 / 매도: {rsi_sell} 이상)
    - BB 조건 사용 여부: {use_bb} (매수 하단 이격: {bb_buy_pct}%, 매도 상단 이격: {bb_sell_pct}%)
    - 지표 결합 방식: {mode}
    """
    
    
    data_summary = "[5대 역사적 시나리오별 시뮬레이션 결과]\n"
    for sc_name, res in all_results.items():
        data_summary += f"- {sc_name}: 최종수익률 {res['total_return_pct']:.2f}%, 매매횟수 {res['trade_count']}회, 승률 {res['win_rate']:.1f}%\n"

    
    prompt = f"""
    당신은 가상자산 규제 및 투자자 보호 가이드라인을 준수하는 '금융 기술적 분석 교육 AI 조교'입니다.
    아래 제공된 [사용자 설정 전략 스펙]과 [5대 역사적 시나리오별 시뮬레이션 결과]를 바탕으로, 이 투자 전략이 가진 '기술적 취약점'과 '시장 환경별 문제점'을 학술적/통계적으로 분석해 주세요.

    {strategy_summary}
    {data_summary}

    [⚠️ 작성 지침 - 규제 준수]
    1. 절대로 "이 전략을 쓰면 돈을 법니다/잃습니다" 같은 단정적이거나 수익을 보장/예단하는 표현을 사용하지 마십시오.
    2. "특정 장세(예: 폭락장)에서 지표의 이격이 너무 좁아 잦은 손절매가 발생할 위험이 관찰됨"과 같이 '현상 분석'과 '리스크 경고', '교육적 개선 방향' 중심으로 서술하십시오.
    3. 말투는 정중하고 객관적인 연구원 어조(~니다)로 작성하고, Markdown 문법을 사용해 가독성 있게 출력하십시오.
    """

    
    try:
        response = client.generate_content("prompt")
        return response.text
    except Exception as e:
        return f"🚨 AI 분석 중 오류가 발생했습니다: {str(e)}"


def calculate_rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.ewm(com=period - 1, adjust=False).mean()
    ma_down = down.ewm(com=period - 1, adjust=False).mean()
    rs = ma_up / ma_down
    return 100 - (100 / (1 + rs))

def load_scenario_data(ticker, scenario_name):
    if scenario_name == "💥 2022 FTX 대폭락장":
        to_date = "2023-01-01T09:00:00"
        count = 200
    elif scenario_name == "🚀 2021 역대급 대불장":
        to_date = "2021-05-01T09:00:00"
        count = 200
    elif scenario_name == "😴 2023 지루한 박스권 횡보장":
        to_date = "2023-10-01T09:00:00"
        count = 200
    elif scenario_name == "📈 평상시 (완만한 상승장)":
        to_date = "2025-01-01T09:00:00"
        count = 200
    elif scenario_name == "📉 평상시 (우하향 안정기)":
        to_date = "2024-09-01T09:00:00"
        count = 200
    else:
        return None

    df = pyupbit.get_ohlcv(ticker, interval="day", to=to_date, count=count)
    if df is not None:
        df['MA20'] = df['close'].rolling(window=20).mean()
        df['std'] = df['close'].rolling(window=20).std()
        df['BB_upper'] = df['MA20'] + (2 * df['std'])
        df['BB_lower'] = df['MA20'] - (2 * df['std'])
        df['RSI'] = calculate_rsi(df['close'], period=14)
        df = df.dropna()
    return df


def run_backtest(df, init_seed, use_rsi, use_bb, mode, rsi_buy, rsi_sell, bb_buy_pct, bb_sell_pct):
    cash = init_seed
    coin_balance = 0
    fee_rate = 0.0005
    
    buy_count = 0
    sell_count = 0
    trade_history = []
    entry_price = 0
    asset_history = []

    for i in range(len(df)):
        row = df.iloc[i]
        close_p = row['close']
        rsi_v = row['RSI']
        bb_low = row['BB_lower']
        bb_up = row['BB_upper']
        
        rsi_buy_signal = (rsi_v <= rsi_buy) if use_rsi else False
        rsi_sell_signal = (rsi_v >= rsi_sell) if use_rsi else False
        
        bb_buy_target = bb_low * (1 - (bb_buy_pct / 100))
        bb_sell_target = bb_up * (1 + (bb_sell_pct / 100))
        
        bb_buy_signal = (close_p <= bb_buy_target) if use_bb else False
        bb_sell_signal = (close_p >= bb_sell_target) if use_bb else False
        
        final_buy_trigger = False
        final_sell_trigger = False
        
        if use_rsi and use_bb:
            if mode == "AND (둘 다 충족)":
                final_buy_trigger = rsi_buy_signal and bb_buy_signal
                final_sell_trigger = rsi_sell_signal and bb_sell_signal
            else:
                final_buy_trigger = rsi_buy_signal or bb_buy_signal
                final_sell_trigger = rsi_sell_signal or bb_sell_signal
        elif use_rsi:
            final_buy_trigger = rsi_buy_signal
            final_sell_trigger = rsi_sell_signal
        elif use_bb:
            final_buy_trigger = bb_buy_signal
            final_sell_trigger = bb_sell_signal

        
        if final_buy_trigger and cash > 0 and coin_balance == 0:
            buy_money = cash * (1 - fee_rate)
            coin_balance = buy_money / close_p
            cash = 0
            entry_price = close_p
            buy_count += 1
        elif final_sell_trigger and coin_balance > 0:
            sell_money = coin_balance * close_p * (1 - fee_rate)
            cash = sell_money
            coin_balance = 0
            sell_count += 1
            
            profit_rate = (close_p - entry_price) / entry_price
            trade_history.append(profit_rate)
            
        
        total_asset = cash + (coin_balance * close_p)
        asset_history.append(total_asset)

    final_asset = asset_history[-1] if asset_history else init_seed
    total_return_pct = ((final_asset - init_seed) / init_seed) * 100
    wins = len([p for p in trade_history if p > 0])
    win_rate = (wins / len(trade_history) * 100) if trade_history else 0.0
    
    return {
        'final_asset': final_asset,
        'total_return_pct': total_return_pct,
        'trade_count': sell_count,
        'win_rate': win_rate,
        'asset_history': asset_history,
        'dates': df.index
    }


st.set_page_config(layout="wide")
st.title("📊 전략 백테스팅 및 역사적 시나리오 비교 대시보드")


st.sidebar.header("🪙 0. 투자 종목 선택")

coin_choice = st.sidebar.selectbox(
    "테스트할 코인을 선택하세요",
    ["BTC (비트코인)", "ETH (이더리움)", "XRP (리플)", "SOL (솔라나)", "DOGE (도지코인)"]
)

ticker = f"KRW-{coin_choice.split(' ')[0]}"

st.sidebar.header("💰 1. 자본금 및 조건 설정")
init_seed = st.sidebar.number_input("초기 자본금 (KRW)", value=10000000, step=1000000)

use_rsi = st.sidebar.checkbox("🟢 RSI 조건 사용", value=True)
use_bb = st.sidebar.checkbox("🔵 볼린저 밴드 조건 사용", value=True)

mode = "AND (둘 다 충족)"
if use_rsi and use_bb:
    mode = st.sidebar.radio("🔀 조건 결합 방식", ["AND (둘 다 충족)", "OR (하나만 충족)"])


rsi_buy, rsi_sell, bb_buy_pct, bb_sell_pct = 30, 70, 0.5, 0.5

if use_rsi:
    st.sidebar.subheader("RSI 매매 기준")
    rsi_buy = st.sidebar.slider("매수 진입 RSI (이하)", 10, 50, 30)
    rsi_sell = st.sidebar.slider("매도 청산 RSI (이상)", 50, 90, 70)

if use_bb:
    st.sidebar.subheader("볼린저 밴드 이격 기준")
    bb_buy_pct = st.sidebar.slider("매수 BB 하단 이격도 (%)", 0.0, 5.0, 0.5)
    bb_sell_pct = st.sidebar.slider("매도 BB 상단 이격도 (%)", 0.0, 5.0, 0.5)


st.write("### 📅 2. 백테스팅 환경")
scenario = st.selectbox(
    "시뮬레이션할 시장 상황을 선택하세요",
    ["💥 2022 FTX 대폭락장", "🚀 2021 역대급 대불장", "😴 2023 지루한 박스권 횡보장", "📈 평상시 (완만한 상승장)", "📉 평상시 (우하향 안정기)"]
)



df_scenario = load_scenario_data(ticker, scenario)

if df_scenario is not None and not df_scenario.empty:
    result = run_backtest(df_scenario, init_seed, use_rsi, use_bb, mode, rsi_buy, rsi_sell, bb_buy_pct, bb_sell_pct)
    
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("최종 순자산", f"{result['final_asset']:,.0f} 원", f"{result['total_return_pct']:.2f} %")
    with col2:
        st.metric("총 매도 청산 횟수", f"{result['trade_count']} 회")
    with col3:
        st.metric("전략 승률", f"{result['win_rate']:.1f} %")
        
    st.write("---")
    st.write("### 📈 3. 시나리오 및 수익 데이터 분석 차트")
    
    
    fig, (ax_main, ax_rsi, ax_asset) = plt.subplots(3, 1, figsize=(14, 14), sharex=True, 
                                                    gridspec_kw={'height_ratios': [2, 1, 1.5]})
    

    ax_main.plot(df_scenario.index, df_scenario['close'], label='비트코인 일봉 종가', color='purple', linewidth=2)
    if use_bb:
        ax_main.plot(df_scenario.index, df_scenario['BB_upper'], label='BB 상단선', color='gray', linestyle='--', alpha=0.6)
        ax_main.plot(df_scenario.index, df_scenario['BB_lower'], label='BB 하단선', color='gray', linestyle='--', alpha=0.6)
        ax_main.fill_between(df_scenario.index, df_scenario['BB_lower'], df_scenario['BB_upper'], color='gray', alpha=0.05)
    ax_main.set_title(f"{scenario} - 일봉 주가 및 기술적 지표 변동 흐름", fontsize=14, fontweight='bold')
    ax_main.set_ylabel("가격 (KRW)")
    ax_main.legend(loc='upper left')
    ax_main.grid(True, linestyle=':', alpha=0.6)
    
 
    ax_rsi.plot(df_scenario.index, df_scenario['RSI'], label='RSI (14)', color='green', linewidth=1.5)
    if use_rsi:
        ax_rsi.axhline(y=rsi_sell, color='red', linestyle=':', linewidth=1.5, label=f'매도 기준선 ({rsi_sell})')
        ax_rsi.axhline(y=rsi_buy, color='blue', linestyle=':', linewidth=1.5, label=f'매수 기준선 ({rsi_buy})')
    ax_rsi.axhline(y=50, color='lightgray', linestyle='-', linewidth=1) 
    ax_rsi.set_ylim(10, 90)
    ax_rsi.set_ylabel("RSI 수치")
    ax_rsi.legend(loc='upper left')
    ax_rsi.grid(True, linestyle=':', alpha=0.6)
    
 
    ax_asset.plot(result['dates'], result['asset_history'], label='하루 단위 보유 총자산', color='dodgerblue', linewidth=2.5)
    ax_asset.axhline(y=init_seed, color='black', linestyle='-', linewidth=1, alpha=0.7, label='투자 원금')
    ax_asset.set_title("🕒 백테스팅 일별 총자산 가치 변화 (수익 추이 곡선)", fontsize=13, fontweight='bold')
    ax_asset.set_xlabel("날짜 (Time)")
    ax_asset.set_ylabel("자산 가치 (KRW)")
    ax_asset.legend(loc='upper left')
    ax_asset.grid(True, linestyle=':', alpha=0.6)
    
   
    fig.autofmt_xdate()
    plt.tight_layout()
    st.pyplot(fig)

    all_scenarios = [
    "💥 2022 FTX 대폭락장", 
    "🚀 2021 역대급 대불장", 
    "😴 2023 지루한 박스권 횡보장", 
    "📈 평상시 (완만한 상승장)", 
    "📉 평상시 (우하향 안정기)"
]

all_results = {}
for sc in all_scenarios:
    df_sc = load_scenario_data(ticker, sc)
    if df_sc is not None:
        res = run_backtest(df_sc, init_seed, use_rsi, use_bb, mode, rsi_buy, rsi_sell, bb_buy_pct, bb_sell_pct)
        all_results[sc] = res

st.write("---")
st.write("### 🤖 4. AI 기반 전략 취약점 오디팅 (Auditing)")
st.write("내가 설정한 지표 조건이 5대 시나리오 전체에서 어떤 맹점을 가졌는지 AI에게 정밀 진단을 요청할 수 있습니다.")


if st.button("🔍 5대 시나리오 통합 AI 진단 리포트 출력"):
    with st.spinner("AI가 5개 시나리오 데이터를 교차 검증하며 전략의 취약점을 분석하고 있습니다... (약 5초 소요)"):
        ai_report = get_ai_analysis(
            all_results, use_rsi, use_bb, mode, rsi_buy, rsi_sell, bb_buy_pct, bb_sell_pct
        )
        
        if ai_report:
            st.success("📋 AI 오디팅 완료! 분석 결과는 아래와 같습니다.")
            st.info(ai_report)
    
else:
    st.error("데이터 수집에 실패했거나 선택한 시나리오의 데이터가 비어있습니다.")

import pyupbit
import pandas as pd
import time
import json
import numpy as np

# 1. 보조지표 계산 함수 (RSI & 볼린저 밴드)
def calculate_indicators(df, period_rsi=14, period_bb=20):
    # RSI 계산
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period_rsi).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period_rsi).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # 볼린저 밴드 계산 (중간선, 상단선, 하단선)
    df['bb_middle'] = df['close'].rolling(window=period_bb).mean() # 20일 이동평균선
    std = df['close'].rolling(window=period_bb).std()              # 표준편차
    df['bb_upper'] = df['bb_middle'] + (std * 2)                  # 상단선
    df['bb_lower'] = df['bb_middle'] - (std * 2)                  # 하단선
    
    return df

# 뉴스 데이터베이스 (기존과 동일)
news_db = {
    "2020": {
        "2020-11-24 09:00": ("📈 [시황] 페이팔(PayPal) 암호화폐 결제 전면 허용", "글로벌 결제 공룡 페이팔이 암호화폐 거래 서비스를 개시하며 전통 자금 유입의 신호탄을 쏘았습니다."),
        "2020-12-16 18:00": ("🔥 [속보] 비트코인, 사상 최초 '2만 달러' 장벽 돌파", "비트코인이 출범 이후 사상 최초로 20,000달러를 강력하게 돌파했습니다."),
        "2020-12-17 09:00": ("📰 [외신] 마이크로스트레티지, 수천억 규모 대규모 추가 매집 공시", "기관 중심의 장기 우상향 랠리 신뢰도를 극대화하고 있습니다."),
        "2020-12-17 13:00": ("🚨 [시황] 비트코인 폭등에 알트코인 동반 과열... 도지코인 징후", "자금이 메이저 알트코인 및 밈코인으로 급격히 순환매되고 있습니다."),
        "2020-12-17 21:00": ("💬 [트윗] 일론 머스크, 의미심장한 도지코인 언급", "전 세계 개인 투자자들의 매수 주문이 쏟아지며 호가창 매도 잔량이 청산되고 있습니다.")
    },
    "2022": {
        "2022-11-04 09:00": ("📰 [외신] 코인데스크 '알라메다 리서치' 대차대조표 부실 의혹 고발", "FTX 거래소의 자매회사 알라메다 리서치의 부실 회계 상태를 폭로했습니다."),
        "2022-11-07 09:00": ("🔥 [속보] 바이낸스 창폰자오, 보유 중인 FTT 토큰 전량 시장 매도 선언", "리스크 관리 차원에서 보유 중인 FTT 토큰 전량을 청산하겠다고 선언하며 뱅크런의 도화선이 당겨졌습니다."),
        "2022-11-08 14:00": ("🚨 [시황] FTX 거래소, 고객 출금 지연 및 대규모 자금 이탈 현상 발생", "출금 지연 및 트랜잭션 마비 현상이 관측되며 시장 전반에 극심한 패닉셀이 관측됩니다."),
        "2022-11-08 20:00": ("⚠️ [긴급] FTX CEO 샘 뱅크먼, '바이낸스에 구제금융 및 인수 합병 요청 중'", "파산 위기에 몰린 FTX가 바이낸스 측에 유동성 지원을 구걸하고 있다는 소식입니다."),
        "2022-11-09 11:00": ("📈 [시황] 고래 자금 탈중앙화 거래소 유니스왑(UNI)으로 대거 피난 역주행", "자산 보존을 원하는 스마트 머니들이 디파이(DeFi) 생태계로 탈출하고 있습니다."),
        "2022-11-12 09:00": ("🚨 [파국] FTX 거래소 최종 인수 결렬 및 미국 챕터11 파산 보호 신청 공시", "부실 규모가 감당 불가능하여 인수가 무산되었으며 알트코인 중심의 무차별 투매가 시작됩니다.")
    },
    "2026": {
        "2026-05-13 21:30": ("📊 [공시] 미 연방준비제도(Fed), 인플레이션 우려 속 기준금리 동결 발표", "미국 연방공개시장위원회(FOMO) 정례 회의 브리핑을 통해 인플레이션 압력이 여전히 견고하다고 판단, 시장의 예상을 깨고 금리 동결 및 긴축 유지 스탠스를 고수하며 시장에 차익 실현 매물이 가파르게 출현 중입니다."),
        "2026-05-20 10:00": ("🛡️ [규제] SEC-리플(XRP) 수년간의 소송 최종 종결 및 합의 가이드라인 확정", "미국 증권거래위원회(SEC)와 리플 랩스 간의 기나긴 증권성 법적 공방이 공식적인 최종 합의안 도출로 마감되었습니다. 규제 불확실성이 완전 해소되며 가상자산 전반에 막대한 기관 유입 기대감이 반영됩니다."),
        "2026-05-20 15:00": ("⚡ [기술] 솔라나(SOL) 네트워크 고도화 패치 완료... 처리 속도 3배 증가", "솔라나 메인넷 생태계가 고질적인 네트워크 노드 정체 현상을 해결하는 대규모 업그레이드를 성공적으로 정착시키며, 초당 트랜잭션(TPS) 처리량이 대폭 상승하여 매수 대기 자금이 집중되고 있습니다."),
        "2026-05-25 09:00": ("📰 [시황] 글로벌 웹3 플랫폼, 아발란체(AVAX) 기반 대규모 서브넷 도입 계약 체결", "대형 빅테크 기업이 자사 메타버스 및 결제 백엔드 시스템으로 아발란체 블록체인을 채택했다는 소식이 전해지며 레이어1 섹터 전반에 강력한 기술적 반등 흐름이 연출되고 있습니다.")
    }
}

def get_data(ticker, target_date):
    df_list = []
    current_to = target_date
    for _ in range(25):
        df = pyupbit.get_ohlcv(ticker, interval="minute60", to=current_to, count=200)
        if df is None or len(df) == 0: break
        df_list.append(df)
        current_to = df.index[0]
        time.sleep(0.2)
    return pd.concat(df_list).sort_index() if df_list else pd.DataFrame()

# 2. 메인 빌드 로직 (코인별 개별 키값 생성)
scenarios = {"2020": "2020-12-25 00:00:00", "2022": "2022-11-25 00:00:00", "2026": "2026-03-01 00:00:00"}

for year, target_date in scenarios.items():
    combined_data = {}
    tickers = ["KRW-BTC", "KRW-ETH", "KRW-SOL", "KRW-XRP", "KRW-ADA"] if year == "2026" else ["KRW-BTC", "KRW-ETH", "KRW-XRP"]
    
    for ticker in tickers:
        df = get_data(ticker, target_date)
        if df.empty: continue
            
        df = calculate_indicators(df)
        ticker_name = ticker.replace("KRW-", "")
        
        for date, row in df.iterrows():
            date_str = date.strftime('%Y-%m-%d %H:%M')
            if date_str not in combined_data:
                combined_data[date_str] = {"date": date_str, "hasNews": False}
            
            # 여기서 각 코인별로 데이터를 분리해서 저장합니다!
            combined_data[date_str][f"{ticker_name}_close"] = int(row['close'])
            combined_data[date_str][f"{ticker_name}_upper"] = int(row['bb_upper']) if pd.notnull(row['bb_upper']) else 0
            combined_data[date_str][f"{ticker_name}_middle"] = int(row['bb_middle']) if pd.notnull(row['bb_middle']) else 0
            combined_data[date_str][f"{ticker_name}_lower"] = int(row['bb_lower']) if pd.notnull(row['bb_lower']) else 0
            
            if date_str in news_db.get(year, {}):
                combined_data[date_str].update({"hasNews": True, "newsTitle": news_db[year][date_str][0], "newsContent": news_db[year][date_str][1]})

    # 파일 저장
    final_list = sorted(list(combined_data.values()), key=lambda x: x['date'])
    with open(f"data_{year}.js", "w", encoding="utf-8") as f:
        f.write(f"const market_data = {json.dumps(final_list, indent=4)};\nexport default market_data;")
    print(f"{year}년 데이터 생성 완료.")
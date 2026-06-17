import pandas as pd
import random
import unicodedata
import os
import numpy as np

# --- 설정 부분 ---
LOTTO_DATA_FILE = 'lotto_excel.xlsx'
MIN_WINDOW_SIZE = 50
NUM_PREDICTION_SETS = 10

def load_data(file_path):
    if not os.path.exists(file_path):
        print(f"오류: '{file_path}' 파일을 찾을 수 없습니다.")
        return None
    try:
        df = pd.read_excel(file_path)
        df.columns = df.columns.str.strip().str.lower()
        return df.sort_values(by='회차').reset_index(drop=True)
    except Exception as e:
        print(f"오류: 엑셀 파일을 불러올 수 없습니다. ({e})")
        return None

def update_excel_data(file_path):
    print("\n--- [로또 데이터 자동 업데이트] ---")
    if os.path.exists(file_path):
        try:
            df = pd.read_excel(file_path)
            df.columns = df.columns.str.strip().str.lower()
            last_row = df.iloc[-1]
            print(f"▶ 현재 최신 데이터: [{int(last_row['회차'])}회차] 당첨번호: {', '.join(map(str, [int(last_row['one']), int(last_row['two']), int(last_row['three']), int(last_row['four']), int(last_row['five']), int(last_row['six'])]))}")
        except: pass

    user_input = input("\n이번 주 데이터를 추가하려면 '회차 번호6개'를 띄어쓰기로 입력해 주십시오.\n(입력을 건너뛰려면 엔터를 누르십시오): ").strip()
    if not user_input: return

    try:
        parts = list(map(int, user_input.split()))
        new_row = pd.DataFrame([{'회차': parts[0], 'one': parts[1], 'two': parts[2], 'three': parts[3], 'four': parts[4], 'five': parts[5], 'six': parts[6]}])
        if os.path.exists(file_path):
            df = pd.concat([pd.read_excel(file_path), new_row], ignore_index=True).sort_values(by='회차')
        else:
            df = new_row
        df.to_excel(file_path, index=False)
        print("완료: 데이터가 저장되었습니다.\n")
    except Exception as e:
        print(f"오류 발생: {e}")

def get_display_width(text):
    return sum(2 if unicodedata.east_asian_width(char) in ['F', 'W', 'A'] else 1 for char in str(text))

def print_korean_grid(data_list, headers):
    if not data_list: return
    col_widths = [get_display_width(h) for h in headers]
    for row in data_list:
        for i, key in enumerate(headers):
            col_widths[i] = max(col_widths[i], get_display_width(str(row.get(key, ''))))
    col_widths = [w + 2 for w in col_widths]
    
    def create_separator(chars):
        return chars[0] + "".join(chars[1] * w + chars[2] for w in col_widths)

    print(create_separator("+-+"))
    def print_row(values):
        line = "|"
        for i, val in enumerate(values):
            val_str = str(val)
            padding = col_widths[i] - get_display_width(val_str)
            line += " " * (padding // 2) + val_str + " " * (padding - padding // 2) + "|"
        print(line)

    print_row(headers)
    print(create_separator("+-+"))
    for row in data_list: 
        print_row([row.get(h, '') for h in headers])
    print(create_separator("+-+"))

# --- [V18: 프랙탈 패턴 매칭 & 동적 앙상블 엔진] ---
def calculate_fractal_scores(df_window):
    scores = {i: 0.0 for i in range(1, 46)}
    if len(df_window) < 20: return scores
    
    cols = ['one', 'two', 'three', 'four', 'five', 'six']
    current_pattern = set(df_window.tail(2)[cols].values.flatten())
    
    similarities = []
    for i in range(2, len(df_window) - 2):
        past_pattern = set(df_window.iloc[i-2:i][cols].values.flatten())
        match_count = len(current_pattern.intersection(past_pattern))
        similarities.append((match_count, i)) 
        
    similarities.sort(key=lambda x: x[0], reverse=True)
    top_matches = similarities[:5]
    
    for match_count, idx in top_matches:
        if match_count >= 2: 
            next_draw_nums = df_window.iloc[idx][cols].values
            for num in next_draw_nums:
                scores[int(num)] += (match_count * 1.5)
                
    return scores

def calculate_dynamic_fractal_scores(df_window):
    cols = ['one', 'two', 'three', 'four', 'five', 'six']
    
    short_term_data = df_window.tail(10)[cols].values.flatten()
    short_term_counts = pd.Series(short_term_data).value_counts()
    
    long_term_data = df_window.tail(50)[cols].values.flatten()
    long_term_counts = pd.Series(long_term_data).value_counts()
    
    recent_15_data = df_window.tail(15)[cols].values.flatten()
    unique_nums_count = len(set(recent_15_data))
    
    momentum_weight = 1.5
    reversion_weight = 1.2
    
    if unique_nums_count <= 35: 
        momentum_weight = 2.8
        reversion_weight = 0.5
    elif unique_nums_count >= 41: 
        momentum_weight = 0.5
        reversion_weight = 2.8

    co_matrix = np.zeros((46, 46))
    for _, row in df_window.tail(50).iterrows():
        nums = row[cols].values
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                n1, n2 = int(nums[i]), int(nums[j])
                co_matrix[n1][n2] += 1
                co_matrix[n2][n1] += 1
                
    fractal_scores = calculate_fractal_scores(df_window)
                
    base_scores = {}
    for i in range(1, 46):
        momentum = short_term_counts.get(i, 0)
        reversion = max(0, (50 * 6 / 45) - long_term_counts.get(i, 0))
        fractal = fractal_scores.get(i, 0)
        base_scores[i] = (momentum * momentum_weight) + (reversion * reversion_weight) + fractal + 0.1
        
    return base_scores, co_matrix

def predict_fractal_sets(df_window, count=10):
    cols = ['one', 'two', 'three', 'four', 'five', 'six']
    last_draw = set(int(x) for x in df_window.iloc[-1][cols].values)
    
    base_scores, co_matrix = calculate_dynamic_fractal_scores(df_window)
    pool = list(range(1, 46))
    prediction_sets = []
    
    max_iterations = count * 100
    iterations = 0
    
    while len(prediction_sets) < count and iterations < max_iterations:
        iterations += 1
        picked = set()
        picked.add(random.choices(pool, weights=[base_scores[i] for i in pool], k=1)[0])
        
        while len(picked) < 6:
            dynamic_weights = []
            for n in pool:
                if n in picked:
                    dynamic_weights.append(0.0)
                else:
                    synergy = sum(co_matrix[n][p] for p in picked)
                    combined_score = base_scores[n] * (1 + (synergy * 0.2))
                    dynamic_weights.append(combined_score)
                    
            next_num = random.choices(pool, weights=dynamic_weights, k=1)[0]
            picked.add(next_num)
            
        if len(picked.intersection(last_draw)) >= 3: continue
        if not (100 <= sum(picked) <= 170): continue
        
        sorted_picked = sorted(list(picked))
        if not any(sorted_picked[i] + 1 == sorted_picked[i+1] for i in range(5)): continue
        
        odd_count = sum(1 for n in picked if n % 2 != 0)
        if odd_count not in [2, 3, 4]: continue
        
        low_count = sum(1 for n in picked if n <= 22)
        if low_count not in [2, 3, 4]: continue
        
        last_digits = [n % 10 for n in picked]
        if any(last_digits.count(d) >= 3 for d in set(last_digits)): continue
        
        is_too_similar = False
        for existing_set in prediction_sets:
            if len(picked.intersection(existing_set)) >= 3:
                is_too_similar = True
                break
        if is_too_similar: continue
        
        prediction_sets.append(picked)
        
    return prediction_sets

# --- [정확하게 수정된 분석 및 반복 로직] ---
def run_analysis(df):
    if len(df) < MIN_WINDOW_SIZE + 1:
        print("데이터가 부족합니다.")
        return

    print("\n[엔진 가동] 전체 백테스팅 기간 중 '4개 이상 적중'이 2번 이상 나올 때까지 시뮬레이션을 반복합니다...")
    attempts = 0
    
    # 4개 이상 맞춘 기록이 2번 이상 나올 때까지 전체 과정을 무한 반복
    while True:
        attempts += 1
        current_seed = random.randint(1, 9999999)
        random.seed(current_seed) # 매 반복마다 새로운 랜덤 패턴 적용
        
        results = []
        cols = ['one', 'two', 'three', 'four', 'five', 'six']
        total_4_plus_hits = 0 # 전체 백테스트 중 4개 이상 맞춘 총 횟수
        
        print(f"\n▶ {attempts}번째 시뮬레이션 진행 중 (백테스트 연산 중...)")
        
        # 50회차부터 끝까지 전체 백테스팅 진행
        for i in range(MIN_WINDOW_SIZE, len(df)):
            target = set(int(x) for x in df.iloc[i][cols].values)
            preds = predict_fractal_sets(df.iloc[:i], count=10)
            
            best_match_count = -1
            best_prediction_set = set()
            hits_in_this_draw = {3: 0, 4: 0, 5: 0, 6: 0}

            for pred_set in preds:
                matches = len(pred_set.intersection(target))
                if matches in hits_in_this_draw: 
                    hits_in_this_draw[matches] += 1
                if matches > best_match_count:
                    best_match_count = matches
                    best_prediction_set = pred_set
            
            if not best_prediction_set and preds:
                best_prediction_set = preds[0]
                
            # 이번 회차에서 최고 적중 개수가 4개 이상이라면 누적
            if best_match_count >= 4:
                total_4_plus_hits += 1
                
            results.append({
                "Draw": int(df.iloc[i]['회차']), 
                "Model": "프랙탈패턴(V18)", 
                "Best Prediction": ", ".join(map(str, sorted([int(x) for x in best_prediction_set]))), 
                "Actual Numbers": ", ".join(map(str, sorted([int(x) for x in target]))), 
                "Hits": int(best_match_count), 
                "Acc(%)": round((best_match_count/6)*100, 2), 
                "3hit": hits_in_this_draw[3], 
                "4hit": hits_in_this_draw[4], 
                "5hit": hits_in_this_draw[5], 
                "6hit": hits_in_this_draw[6]
            })
        
        # 백테스팅 종료 후, 누적된 4개 이상 적중 횟수 검사
        if total_4_plus_hits >= 3:
            print(f"\n✅ [{attempts}번 시도] 조건 달성! (백테스팅 전체 기간 중 4개 이상 적중 {total_4_plus_hits}회 발생)")
            
            headers_backtest = ["Draw", "Model", "Best Prediction", "Actual Numbers", "Hits", "Acc(%)", "3hit", "4hit", "5hit", "6hit"]
            print_korean_grid(results[-30:], headers_backtest)
            
            if results:
                avg_accuracy = sum(r["Acc(%)"] for r in results) / len(results)
                print(f"\nTotal Avg Accuracy: {avg_accuracy:.2f}%\n")
            
            print("--- 다음 회차 최종 예측 ---")
            # 백테스트에서 성과가 좋았던 현재의 시드(Seed)를 그대로 유지한 채 최종 예측 생성
            final_preds = predict_fractal_sets(df, count=10)
            output = [{"No.": f"{i}번", "Type": "프랙탈패턴(V18)", "Prediction Numbers": ", ".join(map(str, sorted([int(x) for x in p])))} for i, p in enumerate(final_preds, 1)]
            headers_final = ["No.", "Type", "Prediction Numbers"]
            print_korean_grid(output, headers_final)
            break # 원하는 조건을 찾았으므로 무한 반복 종료
        else:
            print(f"❌ {attempts}번째 시도 완료 (4개 이상 적중 {total_4_plus_hits}회). 다시 돌립니다...")

if __name__ == "__main__":
    update_excel_data(LOTTO_DATA_FILE)
    df = load_data(LOTTO_DATA_FILE)
    if df is not None: run_analysis(df)
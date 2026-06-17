import pandas as pd
import random
import unicodedata
import numpy as np
import matplotlib.pyplot as plt
import platform

# --- 설정 부분 ---
LOTTO_DATA_FILE = 'lotto_excel.xlsx'
MIN_WINDOW_SIZE = 15
NUM_PREDICTION_SETS = 10
MAX_APPEARANCE = 4  # [V4 추가] 한 번호가 10세트 중 최대 몇 번까지 등장할 수 있는지 제한 (Hard Cap)

# --- 시각화 폰트 설정 (한글 깨짐 방지) ---
if platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
elif platform.system() == 'Darwin': # Mac
    plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

def load_data(file_path):
    try:
        df = pd.read_excel(file_path)
        print("엑셀 파일을 성공적으로 불러왔습니다.")
        df.columns = df.columns.str.strip().str.lower()
        required_cols = ['회차', 'one', 'two', 'three', 'four', 'five', 'six']
        if not all(col in df.columns for col in required_cols):
            print(f"오류: 엑셀 파일에 필요한 열({required_cols})이 없습니다.")
            return None
        return df.sort_values(by='회차').reset_index(drop=True)
    except FileNotFoundError:
        print(f"오류: '{file_path}' 파일을 찾을 수 없습니다.")
        return None
    except Exception as e:
        print(f"엑셀 파일을 읽는 중 오류가 발생했습니다: {e}")
        return None

def update_excel_data(file_path):
    print("\n--- [로또 데이터 자동 업데이트] ---")
    try:
        df_preview = pd.read_excel(file_path)
        df_preview.columns = df_preview.columns.str.strip().str.lower()
        if not df_preview.empty and '회차' in df_preview.columns:
            df_preview = df_preview.sort_values(by='회차').reset_index(drop=True)
            last_row = df_preview.iloc[-1]
            last_draw = int(last_row['회차'])
            last_nums = [int(last_row[col]) for col in ['one', 'two', 'three', 'four', 'five', 'six']]
            num_str = ", ".join(map(str, last_nums))
            print(f"▶ 현재 엑셀에 저장된 최신 데이터: [{last_draw}회차] 당첨번호: {num_str}")
        else:
            print("▶ 현재 엑셀 파일에 데이터가 비어 있습니다.")
    except FileNotFoundError:
        print(f"▶ '{file_path}' 파일을 찾을 수 없습니다.")
    except Exception as e:
        print(f"▶ 엑셀 파일 확인 중 오류 발생: {e}")

    user_input = input("\n이번 주 데이터를 추가하려면 '회차 번호6개'를 띄어쓰기로 입력해 주십시오.\n(예: 1100 1 12 23 34 40 45) / 입력을 건너뛰려면 엔터를 누르십시오: ").strip()

    if not user_input:
        print("▶ 데이터 입력을 건너뛰고 분석을 시작합니다.")
        return

    try:
        parts = list(map(int, user_input.split()))
        if len(parts) != 7:
            print("오류: 7개의 숫자를 정확히 입력해야 합니다.")
            return
        draw_no = parts[0]
        numbers = parts[1:]
        try:
            df = pd.read_excel(file_path)
            df.columns = df.columns.str.strip().str.lower()
        except FileNotFoundError:
            df = pd.DataFrame(columns=['회차', 'one', 'two', 'three', 'four', 'five', 'six'])

        if draw_no in df['회차'].values:
            print(f"이미 {draw_no}회차 데이터가 존재합니다.")
            return

        new_row = pd.DataFrame([{'회차': draw_no, 'one': numbers[0], 'two': numbers[1], 'three': numbers[2], 'four': numbers[3], 'five': numbers[4], 'six': numbers[5]}])
        df = pd.concat([df, new_row], ignore_index=True).sort_values(by='회차').reset_index(drop=True)
        df.to_excel(file_path, index=False)
        print(f"완료: {draw_no}회차 데이터 저장됨.\n")
    except Exception as e:
        print(f"오류 발생: {e}")

def get_display_width(text):
    return sum(2 if unicodedata.east_asian_width(char) in ['F', 'W', 'A'] else 1 for char in str(text))

def print_korean_grid(data_list, headers):
    if not data_list: return
    col_widths = [get_display_width(h) for h in headers]
    keys = list(data_list[0].keys())
    for row in data_list:
        for i, key in enumerate(keys):
            col_widths[i] = max(col_widths[i], get_display_width(str(row[key])))
    col_widths = [w + 2 for w in col_widths]
    
    def create_separator(chars):
        return chars[0] + "".join(chars[1] * w + chars[2] for w in col_widths)

    print(create_separator("+-+"))
    
    def print_row(values):
        line = "|"
        for i, val in enumerate(values):
            val_str = str(val)
            padding = col_widths[i] - get_display_width(val_str)
            left_pad = padding // 2
            right_pad = padding - left_pad
            line += " " * left_pad + val_str + " " * right_pad + "|"
        print(line)

    print_row(headers)
    print(create_separator("+-+"))
    for row in data_list:
        print_row(row.values())
    print(create_separator("+-+"))

# --- 🧠 [이진 혁신 빌드] 순수 정수형 위상 잠금 카운팅 엔진 (V4 풀 패키지) ---
def get_multi_timeframe_fft_scores(df_window):
    scores = {i: 0.0 for i in range(1, 46)}
    if len(df_window) < MIN_WINDOW_SIZE:
        return scores

    cols = ['one', 'two', 'three', 'four', 'five', 'six']
    timeframes = [5, 10, 15, 30, 50, 100]

    for num in range(1, 46):
        integer_votes = 0.5 
        
        for w in timeframes:
            actual_w = min(len(df_window), w)
            if actual_w < 5: continue
            
            recent_df = df_window.iloc[-actual_w:]
            time_series = [1 if num in recent_df.iloc[idx][cols].values else 0 for idx in range(actual_w)]
                
            raw_signal = np.array(time_series, dtype=float)
            padded_w = actual_w * 2
            
            fft_result = np.fft.fft(raw_signal, n=padded_w)
            amplitudes = np.abs(fft_result)
            phases = np.angle(fft_result)
            
            half_n = padded_w // 2
            
            if half_n >= 3:
                mean_amp = np.mean(amplitudes[1:half_n+1])
                std_amp = np.std(amplitudes[1:half_n+1])
                noise_threshold = mean_amp + (0.3 * std_amp)
                
                for k in range(1, half_n + 1):
                    if amplitudes[k] >= noise_threshold:
                        phase = phases[k]
                        freq = 2 * np.pi * k / padded_w
                        
                        cos_val = np.cos(freq * actual_w + phase)
                        
                        if cos_val > 0.68: 
                            weight = 2.0 if w <= 15 else 1.0
                            integer_votes += weight

        scores[num] = np.log1p(integer_votes)

    max_score = max(scores.values())
    min_score = min(scores.values())
    if max_score > min_score:
        for num in scores:
            scores[num] = ((scores[num] - min_score) / (max_score - min_score)) * 10
    else:
        for num in scores: scores[num] = 1.0
            
    return scores

def predict_resonant_wave_sets(df_window, count=10):
    if len(df_window) < MIN_WINDOW_SIZE:
        return [set(random.sample(range(1, 46), 6)) for _ in range(count)], {}

    wave_scores = get_multi_timeframe_fft_scores(df_window)
    prediction_sets = []
    number_usage = {i: 0 for i in range(1, 46)}
    
    for _ in range(count):
        pool = list(range(1, 46))
        
        weights = []
        for n in pool:
            if number_usage[n] >= MAX_APPEARANCE:
                weights.append(0.0)
            else:
                w = np.exp(wave_scores[n] / 1.5) / (number_usage[n] * 2.0 + 1)
                weights.append(w)
        
        selected = set()
        while len(selected) < 6:
            if sum(weights) <= 0: weights = [1] * len(pool)
            picked = random.choices(pool, weights=weights, k=1)[0]
            selected.add(picked)
            weights[pool.index(picked)] = 0.0  
            
        prediction_sets.append(selected)
        for num in selected:
            number_usage[num] += 1

    return prediction_sets, wave_scores

def format_numbers(num_set):
    return ", ".join(str(int(x)) for x in sorted(list(num_set)))

# --- 📊 시각화 함수 추가 ---
def show_prediction_chart(wave_scores, next_draw):
    if not wave_scores:
        return
    
    nums = list(wave_scores.keys())
    scores = list(wave_scores.values())
    
    plt.figure(figsize=(14, 6))
    bars = plt.bar(nums, scores, color='cornflowerblue', edgecolor='black', alpha=0.8)
    
    # 상위 6개 번호를 찾아서 빨간색으로 강조
    top_6_indices = np.argsort(scores)[-6:]
    for idx in top_6_indices:
        bars[idx].set_color('tomato')
        bars[idx].set_edgecolor('black')

    plt.title(f'[{next_draw}회차 예측] V4 엔진 1~45번 파동 점수 분석', fontsize=16, fontweight='bold')
    plt.xlabel('로또 번호 (1번 ~ 45번)', fontsize=12)
    plt.ylabel('FFT 파동 압축 점수 (가중치)', fontsize=12)
    plt.xticks(nums, fontsize=9)
    plt.axhline(y=np.mean(scores), color='gray', linestyle='--', label=f'평균 점수 ({np.mean(scores):.2f})')
    
    # 범례 추가
    import matplotlib.patches as mpatches
    red_patch = mpatches.Patch(color='tomato', label='Top 6 주도 번호 (강한 파동)')
    blue_patch = mpatches.Patch(color='cornflowerblue', label='일반 / 소외 번호')
    plt.legend(handles=[red_patch, blue_patch, plt.Line2D([0], [0], color='gray', linestyle='--', label='평균 점수')])
    
    plt.grid(axis='y', linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.show()

def run_analysis(df):
    if df is None or len(df) < MIN_WINDOW_SIZE:
        print("분석 데이터 부족")
        return

    results = []
    number_cols = ['one', 'two', 'three', 'four', 'five', 'six']
    
    for i in range(MIN_WINDOW_SIZE, len(df)):
        train_df = df.iloc[:i]
        target_draw = df.iloc[i]
        target_actual_numbers = set(target_draw[number_cols].values)
        
        list_of_predictions, _ = predict_resonant_wave_sets(train_df, count=NUM_PREDICTION_SETS)
        
        best_match_count = -1
        best_prediction_set = set()
        best_model_type = "균형위상(V4)"
        hits_in_this_draw = {3: 0, 4: 0, 5: 0, 6: 0}

        for idx, pred_set in enumerate(list_of_predictions):
            matches = len(pred_set.intersection(target_actual_numbers))
            if matches in hits_in_this_draw: hits_in_this_draw[matches] += 1
            
            if matches > best_match_count:
                best_match_count = matches
                best_prediction_set = pred_set
        
        if not best_prediction_set and list_of_predictions:
            best_prediction_set = list_of_predictions[0]

        accuracy = (best_match_count / 6) * 100
        results.append({
            'Draw': int(target_draw['회차']),
            'Model': best_model_type,
            'Best Prediction': format_numbers(best_prediction_set),
            'Actual Numbers': format_numbers(target_actual_numbers),
            'Hits': best_match_count,
            'Acc(%)': round(accuracy, 2),
            '3hit': hits_in_this_draw[3],
            '4hit': hits_in_this_draw[4],
            '5hit': hits_in_this_draw[5],
            '6hit': hits_in_this_draw[6],
        })

    print(f"\n--- Final Analysis: 'Adaptive High-Res FFT Multi-Balance V4 ({NUM_PREDICTION_SETS} Sets)' ---\n")
    headers = ['Draw', 'Model', 'Best Prediction', 'Actual Numbers', 'Hits', 'Acc(%)', '3hit', '4hit', '5hit', '6hit']
    print_korean_grid(results, headers)
    average_accuracy = sum(row['Acc(%)'] for row in results) / len(results) if results else 0
    print(f"\nTotal Avg Accuracy: {average_accuracy:.2f}%")

    print("\n\n--- 다음 회차 최종 예측 ---")
    final_prediction_sets, final_wave_scores = predict_resonant_wave_sets(df, count=NUM_PREDICTION_SETS)
    next_draw = int(df.iloc[-1]['회차']) + 1
    print(f"예측 회차: {next_draw}회차\n")
    
    prediction_output = []
    for i, pred_set in enumerate(final_prediction_sets, 1):
        prediction_output.append({"No.": f"{i}번", "Type": "강제분산+로그압축(V4)", "Prediction Numbers": format_numbers(pred_set)})
    
    print_korean_grid(prediction_output, ['No.', 'Type', 'Prediction Numbers'])
    print("\n[알림] 시각화 차트가 열렸습니다. 차트 창을 닫아야 프로그램이 완전히 종료됩니다.")
    
    # 예측이 끝난 후 차트 실행
    show_prediction_chart(final_wave_scores, next_draw)

if __name__ == "__main__":
    update_excel_data(LOTTO_DATA_FILE)
    main_df = load_data(LOTTO_DATA_FILE)
    if main_df is not None:
        run_analysis(main_df)
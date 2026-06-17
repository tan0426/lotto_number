import pandas as pd
from tabulate import tabulate
import numpy as np
import sys
import os
import datetime
from collections import Counter

# --- 설정 부분 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(current_dir, 'pension_lotto_formatted.xlsx')

FINAL_RECOMMEND_COUNT = 10  
BACKTEST_COUNT = 50       
MIN_WINDOW_SIZE = 20

LOG_DIR_NAME = "Log"
LOG_DIR = os.path.join(current_dir, LOG_DIR_NAME)

DIGIT_COLUMNS = []
CONSECUTIVE_CHECK_COLUMNS = []

# --- 로그 저장용 클래스 ---
class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding='utf-8', buffering=1)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def load_data(file_path):
    global DIGIT_COLUMNS, CONSECUTIVE_CHECK_COLUMNS
    try:
        df = pd.read_excel(file_path)
        print("엑셀 파일을 성공적으로 불러왔습니다.")
        df.columns = df.columns.str.strip().str.lower()
        if '1만' in df.columns:
            cols = ['조', '십만', '1만', '1천', '백', '십', '일']
            check_cols = ['일', '십', '백', '1천', '1만', '십만']
        elif '만' in df.columns:
            cols = ['조', '십만', '만', '천', '백', '십', '일']
            check_cols = ['일', '십', '백', '천', '만', '십만']
        else: return None
        DIGIT_COLUMNS = cols
        CONSECUTIVE_CHECK_COLUMNS = check_cols
        return df.sort_values(by='회차').reset_index(drop=True)
    except Exception as e:
        print(f"오류: {e}")
        return None

# --- [새로운 전략 엔진] Tail-Block 999 분석 ---
def get_front_freq_weights(df, col_name, window=30):
    if col_name not in df.columns:
        return np.ones(10) / 10.0
    recent_vals = df[col_name].tail(window).values
    freq_weights = np.zeros(10)
    for val in recent_vals:
        freq_weights[int(val)] += 1
    freq_weights = freq_weights + 0.1 
    return freq_weights / freq_weights.sum()

def get_hot_tail_blocks(df, k=10):
    """끝 3자리를 하나의 문자열 블록으로 취급하여 가장 자주 나온 패턴 추출"""
    tail_blocks = []
    for _, row in df.iterrows():
        block = f"{int(row['백'])}{int(row['십'])}{int(row['일'])}"
        tail_blocks.append(block)
    
    # 가장 많이 등장한 블록 카운트
    block_counts = Counter(tail_blocks)
    most_common = block_counts.most_common(k)
    return [block for block, count in most_common]

def generate_bucket_candidates(df_window, last_row=None):
    lower_cols = [col.lower() for col in DIGIT_COLUMNS]
    tail_cols = ['백', '십', '일']
    front_cols = [col for col in lower_cols if col not in tail_cols and col != '조']

    # 앞자리 확률 계산
    front_probs = {col: get_front_freq_weights(df_window, col) for col in front_cols}
    
    # 역대 가장 많이 나온 끝 3자리 블록 상위 10개 추출
    hot_blocks = get_hot_tail_blocks(df_window, k=10)

    candidates = []
    seen = set()
    
    while len(candidates) < FINAL_RECOMMEND_COUNT:
        cand = {}
        cand['조'] = np.random.randint(1, 6)
        
        # 앞자리는 최근 빈도 기반 무작위 추출
        for col in front_cols:
            cand[col] = np.random.choice(10, p=front_probs[col])
            
        # 끝 3자리는 핫 블록 통째로 삽입
        selected_block = np.random.choice(hot_blocks)
        cand['백'] = int(selected_block[0])
        cand['십'] = int(selected_block[1])
        cand['일'] = int(selected_block[2])
            
        cand_tuple = tuple(cand[c] for c in lower_cols)
        if cand_tuple not in seen:
            seen.add(cand_tuple)
            candidates.append(cand)
            
    return candidates

# --- 원본 백테스팅 및 로그 분석 로직 ---
def run_analysis(df):
    if df is None or len(df) < MIN_WINDOW_SIZE: return
    if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"log_tail_block_{timestamp}.txt"
    log_filepath = os.path.join(LOG_DIR, log_filename)
    sys.stdout = Logger(log_filepath)
    lower_digit_columns = [col.lower() for col in DIGIT_COLUMNS]
    lower_consecutive_columns = [col.lower() for col in CONSECUTIVE_CHECK_COLUMNS]
    results = []
    
    print(f"--- 분석 설정: Tail-Block 999 Strategy ---")
    print(f"Target Count: {FINAL_RECOMMEND_COUNT} Games")
    print(f"Strategy: Insert Top-10 Historical 3-Digit Blocks Directly.")
    print(f"'Jo' digit 1-5 Random. Front digits Frequency Scaled.")
    
    start_index = max(MIN_WINDOW_SIZE, len(df) - BACKTEST_COUNT)
    for i in range(start_index, len(df)):
        sys.stdout.terminal.write(f"\r>> 검증 진행 중... [{i - start_index + 1}/{len(df) - start_index}]")
        sys.stdout.terminal.flush()
        train_df = df.iloc[:i].copy(); target_draw = df.iloc[i]
        prediction_sets = generate_bucket_candidates(train_df, train_df.iloc[-1])
        actual_digits = target_draw[lower_digit_columns].to_dict()
        best_match_count = -1; best_prediction_set = {}
        hits_in_this_draw = {k: 0 for k in range(len(lower_consecutive_columns) + 2)}

        for pred_set in prediction_sets:
            consecutive_matches = 0
            for col in lower_consecutive_columns:
                if pred_set.get(col) == actual_digits.get(col): consecutive_matches += 1
                else: break
            if consecutive_matches > 0: hits_in_this_draw[consecutive_matches] += 1
            if consecutive_matches > best_match_count:
                best_match_count = consecutive_matches; best_prediction_set = pred_set
        
        if best_match_count == -1: best_match_count = 0; best_prediction_set = prediction_sets[0]
        accuracy = (best_match_count / len(lower_consecutive_columns)) * 100
        results.append({
            'Draw': int(target_draw['회차']),
            'Best Pred': ''.join(map(str, [best_prediction_set.get(col,'') for col in lower_digit_columns])),
            'Actual': ''.join(map(str, [actual_digits[col] for col in lower_digit_columns])),
            'Hits-C': best_match_count, 'Acc(%)': round(accuracy, 2),
            '1-Hit': hits_in_this_draw.get(1, 0), '2-Hits': hits_in_this_draw.get(2, 0),
            '3-Hits': hits_in_this_draw.get(3, 0), '4-Hits': hits_in_this_draw.get(4, 0),
            '5-Hits': hits_in_this_draw.get(5, 0), '6-Hits': hits_in_this_draw.get(6, 0),
        })

    print(f"\n\n--- Backtesting Results (Latest {len(results)} Draws) ---\n")
    print(tabulate(results, headers="keys", tablefmt="pipe", numalign="center"))
    average_accuracy = sum(row['Acc(%)'] for row in results) / len(results)
    print(f"\nOverall Average Accuracy: {average_accuracy:.2f}%\n")
    
    last_row = df.iloc[-1]
    final_candidates = generate_bucket_candidates(df.copy(), last_row)
    print(f"\n--- Prediction for Next Draw #{int(last_row['회차'])+1} ({FINAL_RECOMMEND_COUNT} Games) ---")
    prediction_output = []
    for j, pred_set in enumerate(final_candidates):
        jo_val = pred_set['조']
        rest_val = ''.join(map(str, [pred_set[col] for col in lower_digit_columns[1:]])) 
        formatted_pred = f"{jo_val}조 {rest_val}"
        prediction_output.append({f"Rank": f"#{j+1}", "Predicted": formatted_pred})
        
    print(tabulate(prediction_output, headers="keys", tablefmt="pipe"))
    
    if isinstance(sys.stdout, Logger):
        sys.stdout.log.close(); sys.stdout = sys.stdout.terminal
    print(f"로그가 '{log_filepath}'에 저장되었습니다.")

# --- 원본 자동 업데이트 인터페이스 ---
def input_new_draw(df, file_path):
    print("\n--- [데이터 자동 업데이트] ---")
    try:
        if not df.empty:
            last_row = df.iloc[-1]
            last_draw = int(last_row['회차'])
            last_jo = int(last_row['조'])
            
            if '1만' in df.columns: num_cols = ['십만', '1만', '1천', '백', '십', '일']
            else: num_cols = ['십만', '만', '천', '백', '십', '일']
                
            last_nums = "".join([str(int(last_row[col])) for col in num_cols])
            print(f"▶ 현재 엑셀에 저장된 최신 데이터: [{last_draw}회차] 당첨번호: {last_jo}조 {last_nums}")
        else:
            print("▶ 현재 엑셀 파일에 데이터가 비어 있습니다.")
    except Exception as e:
        print(f"▶ 최신 회차 확인 중 오류 발생: {e}")

    while True:
        user_input = input("\n이번 주 데이터를 추가하려면 '회차 조 6자리번호'를 띄어쓰기로 입력해 주십시오.\n(예: 123 3 123456) / 입력을 건너뛰려면 엔터를 누르십시오: ").strip()
        if not user_input:
            print("▶ 데이터 입력을 건너뛰고 분석을 시작합니다.\n")
            break
            
        parts = user_input.split()
        try:
            if len(parts) == 3:
                new_draw = int(parts[0]); jo = int(parts[1]); num_str = parts[2]
                if len(num_str) != 6: raise ValueError("당첨 번호는 6자리여야 합니다.")
                nums = [int(x) for x in num_str]
            elif len(parts) == 8:
                new_draw = int(parts[0]); jo = int(parts[1])
                nums = [int(x) for x in parts[2:]]
            else:
                print("입력 형식이 올바르지 않습니다. 다시 확인해주세요. (예: 123 3 123456)")
                continue
                
            last_draw = int(df['회차'].iloc[-1])
            if new_draw <= last_draw:
                print(f"입력하신 회차({new_draw})가 마지막 회차({last_draw})보다 작거나 같습니다. 다시 입력해주세요.")
                continue
                
            is_type_1 = '1만' in df.columns
            new_row = {
                '회차': new_draw, '조': jo, '십만': nums[0],
                '1만' if is_type_1 else '만': nums[1],
                '1천' if is_type_1 else '천': nums[2],
                '백': nums[3], '십': nums[4], '일': nums[5]
            }
            
            for col in df.columns:
                if col not in new_row: new_row[col] = None
                    
            new_df = pd.DataFrame([new_row])
            df = pd.concat([df, new_df], ignore_index=True)
            df.to_excel(file_path, index=False)
            
            print(f"\n✅ {new_draw}회차 당첨번호({jo}조 {''.join(map(str, nums))})가 엑셀 파일에 성공적으로 업데이트 되었습니다!")
            print("▶ 데이터 업데이트를 완료하고 분석을 시작합니다.\n")
            break
            
        except ValueError as e:
            print(f"입력 오류: 숫자를 정확히 입력해주세요. ({e})")
            
    return df

if __name__ == "__main__":
    main_df = load_data(DATA_FILE)
    if main_df is not None:
        main_df = input_new_draw(main_df, DATA_FILE)
        run_analysis(main_df)
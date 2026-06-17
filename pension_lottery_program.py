import pandas as pd
from tabulate import tabulate
import numpy as np
import sys
import os
import datetime
from collections import Counter

# --- 설정 부분 (여기만 수정하세요!) ---
current_dir = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(current_dir, 'pension_lotto_formatted.xlsx')

# [사용자 설정]
FINAL_RECOMMEND_COUNT = 10  

# [검증 설정]
BACKTEST_COUNT = 50       
MIN_WINDOW_SIZE = 20

# [로그 저장 설정]
LOG_DIR_NAME = "Log"
LOG_DIR = os.path.join(current_dir, LOG_DIR_NAME)

# [설정 유지]
BEAM_WIDTH = 100           
CANDIDATE_EXPANSION = 7    
RECENCY_WEIGHT_ALPHA = 0.96 
HISTORY_WEIGHT = 2.5       

# [1단계 가중치]
STEP_WEIGHTS_ORIGIN = {
    0: 40.0, 1: 300.0, 2: 300.0, 
    3: 1.0, 4: 1.0, 5: 1.0
}

# 전역 변수
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

class GoldenEngine:
    def __init__(self, df_window, digit_cols, alpha):
        self.digit_cols = digit_cols
        self.reversed_cols = digit_cols[::-1] 
        n = len(df_window)
        self.weights_trend = np.power(alpha, np.arange(n, 0, -1))
        self.weights_history = np.ones(n)
        self.df = df_window.copy()
        self.df['__w_trend__'] = self.weights_trend
        self.df['__w_hist__'] = self.weights_history
        self.probs_trend = {}
        self.probs_hist = {}
        self.marginal_trend = {}
        self._precompute()

    def _precompute(self):
        try:
            for col in self.reversed_cols:
                w = self.df.groupby(col)['__w_trend__'].sum()
                total = w.sum()
                self.marginal_trend[col] = w / total if total > 0 else pd.Series(dtype=float)

            for i in range(len(self.reversed_cols) - 1):
                curr_col = self.reversed_cols[i]
                next_col = self.reversed_cols[i+1]
                w_t = self.df.groupby([curr_col, next_col])['__w_trend__'].sum()
                if not w_t.empty:
                    xtab_t = w_t.unstack(fill_value=0)
                    sum_t = xtab_t.sum(axis=1)
                    sum_t[sum_t==0] = 1
                    self.probs_trend[(curr_col, next_col, 1)] = xtab_t.div(sum_t, axis=0)
                w_h = self.df.groupby([curr_col, next_col])['__w_hist__'].sum()
                if not w_h.empty:
                    xtab_h = w_h.unstack(fill_value=0)
                    sum_h = xtab_h.sum(axis=1)
                    sum_h[sum_h==0] = 1
                    self.probs_hist[(curr_col, next_col, 1)] = xtab_h.div(sum_h, axis=0)
                if i >= 1:
                    prev_col = self.reversed_cols[i-1]
                    keys = self.df[curr_col].astype(str) + "_" + self.df[prev_col].astype(str)
                    w_t2 = self.df.groupby([keys, next_col])['__w_trend__'].sum()
                    if not w_t2.empty:
                        xtab_t2 = w_t2.unstack(fill_value=0)
                        sum_t2 = xtab_t2.sum(axis=1)
                        sum_t2[sum_t2==0] = 1
                        self.probs_trend[(curr_col, next_col, 2)] = xtab_t2.div(sum_t2, axis=0)
                    w_h2 = self.df.groupby([keys, next_col])['__w_hist__'].sum()
                    if not w_h2.empty:
                        xtab_h2 = w_h2.unstack(fill_value=0)
                        sum_h2 = xtab_h2.sum(axis=1)
                        sum_h2[sum_h2==0] = 1
                        self.probs_hist[(curr_col, next_col, 2)] = xtab_h2.div(sum_h2, axis=0)
        except: pass

    def get_combined_score(self, curr_col, next_col, prev_val, prev_prev_val=None):
        try:
            p_trend = None
            if prev_prev_val is not None:
                key = f"{prev_val}_{prev_prev_val}"
                mat = self.probs_trend.get((curr_col, next_col, 2))
                if mat is not None and key in mat.index:
                    p_trend = mat.loc[key]
            if p_trend is None:
                mat = self.probs_trend.get((curr_col, next_col, 1))
                if mat is not None and prev_val in mat.index:
                    p_trend = mat.loc[prev_val]
            if p_trend is None or p_trend.sum() == 0:
                p_trend = self.marginal_trend.get(next_col)

            p_hist = None
            if prev_prev_val is not None:
                key = f"{prev_val}_{prev_prev_val}"
                mat = self.probs_hist.get((curr_col, next_col, 2))
                if mat is not None and key in mat.index:
                    p_hist = mat.loc[key]
            if p_hist is None:
                mat = self.probs_hist.get((curr_col, next_col, 1))
                if mat is not None and prev_val in mat.index:
                    p_hist = mat.loc[prev_val]

            if p_trend is None: return None
            p_trend = p_trend.replace(0, 1e-9)
            final_score = np.log(p_trend)
            if p_hist is not None:
                p_hist_aligned = p_hist.reindex(p_trend.index).fillna(0)
                final_score += (p_hist_aligned * HISTORY_WEIGHT)
            return final_score
        except: return None

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

# [Logic Helpers]
def calculate_physics_target(df, col_name):
    if col_name not in df.columns: return 0
    values = df[col_name].values
    if len(values) < 3: return values[-1]
    velocities = [values[i] - values[i-1] for i in range(1, len(values))]
    accelerations = [velocities[i] - velocities[i-1] for i in range(1, len(velocities))]
    next_move = velocities[-1] + accelerations[-1] if velocities and accelerations else 0
    target = (values[-1] + int(round(next_move))) % 10
    return target

def calculate_memory_target(df, col_name, window=3):
    if col_name not in df.columns: return 0
    values = df[col_name].values
    if len(values) < window + 1: return values[-1]
    current_pattern = tuple(values[-window:])
    pattern_counts = {}
    for i in range(len(values) - window - 1):
        if tuple(values[i : i+window]) == current_pattern:
            next_val = values[i + window]
            pattern_counts[next_val] = pattern_counts.get(next_val, 0) + 1
    if not pattern_counts: return pd.Series(values[-30:]).mode()[0]
    return max(pattern_counts, key=pattern_counts.get)

def calculate_hot_number(df, col_name, window=20):
    if col_name not in df.columns: return 0
    values = df[col_name].iloc[-window:].values
    if len(values) == 0: return 0
    counts = Counter(values)
    return counts.most_common(1)[0][0]

def calculate_full_stack_gaps(df, digit_columns, window=3):
    gaps = {}
    for col in digit_columns:
        if col not in df.columns: continue
        recent_values = df[col].iloc[-window:].values
        market_energy = np.mean(recent_values)
        t_phys = calculate_physics_target(df, col)
        t_mem = calculate_memory_target(df, col)
        pred_energy = (t_phys + t_mem) / 2.0
        gaps[col] = market_energy - pred_energy
    return gaps

# [V41 CORE: TAIL GENERATION]
def generate_base_candidates_v41(engine, start_digit):
    lower_digit_columns = [col.lower() for col in DIGIT_COLUMNS]
    reversed_digit_columns = lower_digit_columns[::-1]
    beams = [([start_digit], 0.0)] 
    for i in range(len(reversed_digit_columns) - 1):
        curr_col = reversed_digit_columns[i]
        next_col = reversed_digit_columns[i+1]
        step_weight = STEP_WEIGHTS_ORIGIN.get(i, 1.0)
        all_candidates = []
        for path, score in beams:
            prev_val = path[-1]
            prev_prev_val = path[-2] if len(path) >= 2 else None
            scores = engine.get_combined_score(curr_col, next_col, prev_val, prev_prev_val)
            if scores is not None:
                for next_val, s in scores.nlargest(CANDIDATE_EXPANSION).items():
                    all_candidates.append((path + [next_val], score + (s * step_weight)))
        if not all_candidates: break
        all_candidates.sort(key=lambda x: x[1], reverse=True)
        beams = all_candidates[:BEAM_WIDTH]
    results = []
    for path, score in beams:
        if len(path) == len(lower_digit_columns):
            results.append(dict(zip(lower_digit_columns, path[::-1])))
    return results

def refine_with_full_stack_v41(engine, cand, strategy_type, full_gaps):
    fixed_tail = [cand['일'], cand['십'], cand['백']]
    current_path = list(fixed_tail)
    for i in range(3, len(engine.reversed_cols)):
        c_col_name = engine.reversed_cols[i]
        prev_col_name = engine.reversed_cols[i-1]
        c_val_prev = current_path[-1]
        c_val_prev_prev = current_path[-2]
        scores = engine.get_combined_score(prev_col_name, c_col_name, c_val_prev, c_val_prev_prev)
        if scores is not None:
            best_val = scores.idxmax()
            gap = full_gaps.get(c_col_name, 0.0)
            shift = int(round(gap))
            if strategy_type == 'energy_balanced':
                best_val = (best_val + shift) % 10
            elif strategy_type == 'high_energy':
                best_val = (best_val + shift + 1) % 10
            elif strategy_type == 'raw_physics':
                best_val = best_val 
            current_path.append(best_val)
        else:
            current_path.append(0)
    new_cand_dict = {}
    for idx, col_name in enumerate(engine.reversed_cols):
        if idx < len(current_path): new_cand_dict[col_name] = current_path[idx]
    return new_cand_dict

def get_v41_candidates(df_window, last_row, engine, full_gaps, lower_cols):
    v41_list = []
    seen_combinations = set()
    round_idx = 0
    # [AUTO-SCALE] FINAL_RECOMMEND_COUNT 만큼 생성될 때까지 반복
    while len(v41_list) < FINAL_RECOMMEND_COUNT:
        digit = round_idx % 10
        base_candidates = generate_base_candidates_v41(engine, digit)
        for base_cand in base_candidates:
            current_count = len(v41_list)
            
            threshold_v1 = int(FINAL_RECOMMEND_COUNT * 0.5)
            threshold_v2 = int(FINAL_RECOMMEND_COUNT * 0.8)
            
            if current_count < threshold_v1: strategy = 'energy_balanced'
            elif current_count < threshold_v2: strategy = 'high_energy'
            else: strategy = 'raw_physics'
            
            refined_cand = refine_with_full_stack_v41(engine, base_cand, strategy, full_gaps)
            is_exact_last = True
            for col in lower_cols:
                if refined_cand.get(col) != last_row.get(col): 
                    is_exact_last = False; break
            if is_exact_last: continue
            cand_tuple = tuple(refined_cand[col] for col in lower_cols)
            if cand_tuple in seen_combinations: continue 
            v41_list.append(refined_cand)
            seen_combinations.add(cand_tuple)
            break 
        round_idx += 1
        if round_idx > FINAL_RECOMMEND_COUNT * 20: break 
    return v41_list[:FINAL_RECOMMEND_COUNT]

# [V70 Strategy - AUTO SCALED]
def get_directional_4th(df, col_1000, game_idx):
    values = df[col_1000].values
    if len(values) < 2: return values[-1]
    
    velocity = values[-1] - values[-2]
    last_val = values[-1]
    
    threshold_1 = int(FINAL_RECOMMEND_COUNT * 0.5)
    threshold_2 = int(FINAL_RECOMMEND_COUNT * 0.8)
    
    t_momentum = (last_val + velocity) % 10
    t_reversion = (last_val - int(round(velocity * 0.5))) % 10
    t_hot = calculate_hot_number(df, col_1000)
    
    if game_idx < threshold_1:
        return t_momentum
    elif game_idx < threshold_2:
        return t_reversion
    else:
        return t_hot

def reflow_upper_digits(engine, cand, full_gaps):
    current_path = [cand['일'], cand['십'], cand['백'], cand['천'] if '천' in cand else cand['1천']]
    for k in range(4, len(engine.reversed_cols)):
        c_col_name = engine.reversed_cols[k]
        prev_col_name = engine.reversed_cols[k-1]
        c_val_prev = current_path[-1]
        c_val_prev_prev = current_path[-2]
        scores = engine.get_combined_score(prev_col_name, c_col_name, c_val_prev, c_val_prev_prev)
        if scores is not None:
            best_val = scores.idxmax()
            gap = full_gaps.get(c_col_name, 0.0)
            shift = int(round(gap))
            final_val = (best_val + shift) % 10
        else: final_val = 0
        current_path.append(final_val)
    patched_cand = cand.copy()
    for idx, col_name in enumerate(engine.reversed_cols):
        if idx >= 4: patched_cand[col_name] = current_path[idx]
    return patched_cand

def generate_bucket_candidates(df_window, last_row):
    lower_cols = [col.lower() for col in DIGIT_COLUMNS]
    engine = GoldenEngine(df_window, DIGIT_COLUMNS, RECENCY_WEIGHT_ALPHA)
    full_gaps = calculate_full_stack_gaps(df_window, lower_cols)
    
    v41_candidates = get_v41_candidates(df_window, last_row, engine, full_gaps, lower_cols)
    
    final_list = []
    col_1000 = '1천' if '1천' in lower_cols else '천'
    seen_combinations = set()
    
    for i, cand in enumerate(v41_candidates):
        best_4th = get_directional_4th(df_window, col_1000, i)
        cand[col_1000] = best_4th
        
        final_cand = reflow_upper_digits(engine, cand, full_gaps)
        final_cand['조'] = ((final_cand['조'] - 1) % 5) + 1
        
        cand_tuple = tuple(final_cand[col] for col in lower_cols)
        loop_cnt = 0
        while cand_tuple in seen_combinations and loop_cnt < 10:
            final_cand['조'] = (final_cand['조'] % 5) + 1
            cand_tuple = tuple(final_cand[col] for col in lower_cols)
            loop_cnt += 1
            
        final_list.append(final_cand)
        seen_combinations.add(cand_tuple)
        
    return final_list[:FINAL_RECOMMEND_COUNT]

def run_analysis(df):
    if df is None or len(df) < MIN_WINDOW_SIZE: return
    if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"log_v76_jo_fix_{timestamp}.txt"
    log_filepath = os.path.join(LOG_DIR, log_filename)
    sys.stdout = Logger(log_filepath)
    lower_digit_columns = [col.lower() for col in DIGIT_COLUMNS]
    lower_consecutive_columns = [col.lower() for col in CONSECUTIVE_CHECK_COLUMNS]
    results = []
    
    print(f"--- 분석 설정: Hybrid V76 (Jo Digit Fix 1-5) / BEAM=100 ---")
    print(f"Target Count: {FINAL_RECOMMEND_COUNT} Games")
    print(f"Strategy: Tail Fixed [V41]. 'Jo' digit 1-5. 4th [1000s] Auto-Scaled.")
    
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

def input_new_draw(df, file_path):
    """터미널에서 새로운 당첨 번호를 한 줄로 입력받아 엑셀에 업데이트하는 함수"""
    print("\n--- [데이터 자동 업데이트] ---")
    
    # --- [추가된 부분] 현재 로드된 데이터에서 최신 회차 정보 출력 ---
    try:
        if not df.empty:
            last_row = df.iloc[-1]
            last_draw = int(last_row['회차'])
            last_jo = int(last_row['조'])
            
            # 엑셀 버전에 따라 컬럼명이 다를 수 있으므로 체크
            if '1만' in df.columns:
                num_cols = ['십만', '1만', '1천', '백', '십', '일']
            else:
                num_cols = ['십만', '만', '천', '백', '십', '일']
                
            last_nums = "".join([str(int(last_row[col])) for col in num_cols])
            print(f"▶ 현재 엑셀에 저장된 최신 데이터: [{last_draw}회차] 당첨번호: {last_jo}조 {last_nums}")
        else:
            print("▶ 현재 엑셀 파일에 데이터가 비어 있습니다.")
    except Exception as e:
        print(f"▶ 최신 회차 확인 중 오류 발생: {e}")
    # -------------------------------------------------------------------

    while True:
        # 입력 안내문 (엔터 시 스킵)
        user_input = input("\n이번 주 데이터를 추가하려면 '회차 조 6자리번호'를 띄어쓰기로 입력해 주십시오.\n(예: 123 3 123456) / 입력을 건너뛰려면 엔터를 누르십시오: ").strip()
        
        # 아무것도 입력하지 않고 엔터를 친 경우
        if not user_input:
            print("▶ 데이터 입력을 건너뛰고 분석을 시작합니다.\n")
            break
            
        parts = user_input.split()
        try:
            # 1. '123 3 123456' 처럼 3덩어리로 입력한 경우
            if len(parts) == 3:
                new_draw = int(parts[0])
                jo = int(parts[1])
                num_str = parts[2]
                if len(num_str) != 6:
                    raise ValueError("당첨 번호는 6자리여야 합니다.")
                nums = [int(x) for x in num_str]
                
            # 2. '123 3 1 2 3 4 5 6' 처럼 번호 6개를 전부 띄어 쓴 경우
            elif len(parts) == 8:
                new_draw = int(parts[0])
                jo = int(parts[1])
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
                '회차': new_draw,
                '조': jo,
                '십만': nums[0],
                '1만' if is_type_1 else '만': nums[1],
                '1천' if is_type_1 else '천': nums[2],
                '백': nums[3],
                '십': nums[4],
                '일': nums[5]
            }
            
            # 기존 엑셀에 있던 다른 부가 데이터 컬럼이 있다면 빈칸(None)으로 채움
            for col in df.columns:
                if col not in new_row:
                    new_row[col] = None
                    
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
        # 1. 엑셀 로드 후 데이터 업데이트 프롬프트 띄우기
        main_df = input_new_draw(main_df, DATA_FILE)
        
        # 2. 입력이 끝나면(혹은 건너뛰면) 바로 백테스팅 및 분석 진행
        run_analysis(main_df)
import pandas as pd

# --- 설정 ---
input_file_name = 'pension_lotto_excel.xlsx' # 변환할 원본 엑셀 파일 이름
output_file_name = 'pension_lotto_formatted.xlsx' # 변환된 데이터를 저장할 새 파일 이름 (이름 변경)

try:
    # 1. 원본 엑셀 파일 읽기
    original_df = pd.read_excel(input_file_name, header=None, engine='openpyxl')

    # 데이터가 첫 번째 열에 있다고 가정합니다.
    data_list = original_df.iloc[:, 0].tolist()

    formatted_data = []
    회차_counter = 0

    # 2. 데이터를 읽으면서 원하는 형식으로 변환
    # 데이터 리스트를 2개씩 묶어서 처리합니다 (조, 숫자)
    for i in range(0, len(data_list), 2):
        if i + 1 < len(data_list): # 리스트 범위를 벗어나지 않도록 체크
            회차_counter += 1

            # 첫 번째 항목: 조 정보 (예: '3조')
            group_str = str(data_list[i])
            group_number = int(group_str.replace('조', '').strip())

            # 두 번째 항목: 6자리 숫자 정보
            number_value = str(data_list[i+1])

            # 6자리 숫자를 추출하여 각 자리 숫자로 분리
            if len(number_value) == 6: # 정확히 6자리 숫자인지 확인
                six_digits_str = number_value # 6자리 문자열 추출
                # 각 자리 숫자를 정수로 변환
                digits = [int(d) for d in six_digits_str]

                # 변환된 데이터 추가 (회차, 조, 십만, 1만, 1천, 백, 십, 일)
                formatted_data.append([
                    회차_counter,
                    group_number,
                    digits[0], # 십만 자리
                    digits[1], # 1만 자리
                    digits[2], # 1천 자리
                    digits[3], # 1백 자리
                    digits[4], # 십 자리
                    digits[5]  # 일 자리
                ])
            else:
                 print(f"경고: 회차 {회차_counter} 데이터 '{number_value}'는 6자리가 아니어서 처리할 수 없습니다.")


    # 3. 변환된 데이터를 새로운 DataFrame으로 생성 (열 이름 수정)
    # 십만 자리를 포함하여 6자리로 열 이름을 변경합니다.
    formatted_df = pd.DataFrame(formatted_data, columns=['회차', '조', '십만', '1만', '1천', '백', '십', '일'])

    # 4. 새로운 엑셀 파일로 저장
    formatted_df.to_excel(output_file_name, index=False, engine='openpyxl')

    print(f"데이터 변환 완료. '{output_file_name}' 파일로 저장되었습니다.")

except FileNotFoundError:
    print(f"오류: 원본 파일 '{input_file_name}'을 찾을 수 없습니다. 파일 이름과 경로를 확인해 주세요.")
except ValueError as ve:
     print(f"데이터 처리 오류: {ve}. 원본 파일의 데이터 형식을 확인해 주세요.")
except Exception as e:
    print(f"오류가 발생했습니다: {e}")

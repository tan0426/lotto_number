import numpy as np
import pandas as pd
import random

file_name = 'lotto_excel.xlsx'

lotto_excel = pd.read_excel(file_name, engine='openpyxl')
lotto_excel_list = lotto_excel.values.tolist()
lotto_excel_len = len(lotto_excel)

num_list = []
times_list = []
weights_list = []
lotto_number_list = []
lotto_table = {}
i = 0
j = 0

pick_times = 5 #몇줄 뽑을건지

#num_list 리스트 생성
for i in range(1, 46, 1):
    num_list.append(i)

#times_list 리스트 생성
for i in range(1, 46, 1):
    times_list.append(0)

#weights_list 리스트 생성
for i in range(1, 46, 1):
    weights_list.append(0)

#lotto_number_list 리스트 생성
for i in range(1, 7, 1):
    lotto_number_list.append(0)

#2차원 lotto_table 딕셔너리 생성
for i in range(0, 45, 1):
    lotto_table[num_list[i]] = times_list[i]

#lotto_table 딕셔너리에 뽑힌 횟수 입력
for i in range(0, lotto_excel_len, 1):
    for j in range(1, 7, 1):
        number = lotto_excel_list[i][j]
        lotto_table[number] = lotto_table[number] + 1

#뽑힌 횟수에 따라 가중치 입력
for i in range(1, 46, 1):
    weights_list[i-1] = lotto_table[i]

#중복 없이 6개 연달아 뽑기
for j in range(0, pick_times, 1):
    for i in range(0, 6, 1):
        random_num_one = random.choices(range(1,46), weights= weights_list)[0]
        while random_num_one in lotto_number_list:
            random_num_one = random.choices(range(1,46), weights= weights_list)[0]
        
        lotto_number_list[i] = random_num_one

    lotto_number_list.sort() #정렬

    print(lotto_number_list)
    j = j + 1

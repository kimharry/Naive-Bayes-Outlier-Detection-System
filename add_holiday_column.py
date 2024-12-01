from datetime import datetime, timedelta
from holidayskr import is_holiday

def if_holiday(date_input):
    # 입력된 날짜 파싱
    date_obj = datetime.strptime(date_input, "%Y-%m-%d")

    # 한국 시간 조정 (필요한 경우에만 사용)
    kst_offset = timedelta(hours=9)
    date_kst = date_obj + kst_offset

    # 평일 확인 (월요일=0, 일요일=6)
    is_weekday = date_kst.weekday() < 5
    # 공휴일 확인
    is_holiday_status = is_holiday(date_input)

    if is_weekday and not is_holiday_status:
        return 0
    else:
        return 1
    
def load_raw_data(fname):
    instances = []
    labels = []
    with open(fname, "r") as f:
        f.readline()
        for line in f:
            tmp = line.strip().split(", ")
            tmp[1] = float(tmp[1])
            tmp[2] = float(tmp[2])
            tmp[3] = float(tmp[3])
            tmp[4] = float(tmp[4])
            tmp[5] = int(tmp[5])
            tmp[6] = int(tmp[6])
            tmp[7] = float(tmp[7])
            tmp[8] = int(tmp[8])
            instances.append(tmp[:-1])
            labels.append(tmp[-1])
    return instances, labels

instances, labels = load_raw_data("./data/testing.csv")

# save the new data
with open("./data/testing_holiday.csv", "w") as f:
    f.write("date, is_holiday, avg (temperature), max (temperature), min (temperature), avg (humidity), max (humidity), min (humidity), power, label\n")
    for idx in range(len(instances)):
        f.write("{}, {}, {}, {}, {}, {}, {}, {}, {}, {}\n".format(
            instances[idx][0], if_holiday(instances[idx][0]), instances[idx][1], 
            instances[idx][2], instances[idx][3], instances[idx][4], 
            instances[idx][5], instances[idx][6], instances[idx][7], 
            labels[idx]
        ))
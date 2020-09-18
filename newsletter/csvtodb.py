# import csv
# import os
# import django
# import sys
# from newsletter.models import News,New_News	# 2. App이름.models

# os.chdir('.')

# print("Current dir=", end=""), print(os.getcwd())

# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# print("BASE_DIR=", end=""), print(BASE_DIR)

# sys.path.append(BASE_DIR)

# os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ai_teammate.settings")	# 1. 여기서 프로젝트명.settings입력
# django.setup()
###########################################################################################
#############################################################################################
##########################################################################################

# 위의 과정까지가 python manage.py shell을 키는 것과 비슷한 효과

# 
CSV_PATH = './result_0918.csv'	# 3. csv 파일 경로
import csv
from newsletter.models import *	# 2. App이름.models

bulk_list = []
with open(CSV_PATH, encoding='UTF-8') as csvfile:	# 4. newline =''
    data_reader = csv.reader(csvfile)
    next(data_reader, None)
    for row in data_reader:
        bulk_list.append(News(		# 5. class명.objects.create
            link = row[0],
            title = row[1],
            date = row[2],
            content = row[3],
            tag = row[4],
            big_category = row[5],
            category = row[6],
            team_category = row[7]
        ))

News.objects.bulk_create(bulk_list)
News.objects.values()


CSV_PATH = './merge_data.csv'	# 3. csv 파일 경로
import csv
from newsletter.models import *	# 2. App이름.models

bulk_list = []
with open(CSV_PATH, encoding='UTF-8') as csvfile:	# 4. newline =''
    data_reader = csv.reader(csvfile)
    next(data_reader, None)
    for row in data_reader:
        bulk_list.append(New_News(		# 5. class명.objects.create
            link = row[0],
            tag = row[1],
            content = row[2],
            title = row[3],
            date = row[4],
            big_category = row[5],
        ))

New_News.objects.bulk_create(bulk_list)
New_News.objects.values()


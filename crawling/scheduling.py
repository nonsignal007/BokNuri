import schedule
import time
from crawling import bokjiro_pdf

bokjiro_pdf = bokjiro_pdf()

def job(crawling):
    print("새 데이터를 크롤링합니다...")
    data = crawl_data("https://example.com")
    with open("data.txt", "w") as file:
        file.write(data)
    print("크롤링 완료 및 저장!")

# 매일 1회 크롤링
schedule.every().day.at("00:00").do(job)

# 계속 실행
while True:
    schedule.run_pending()
    time.sleep(1)

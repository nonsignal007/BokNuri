import chromedriver_autoinstaller

path = chromedriver_autoinstaller.install()

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import requests


def bokjiro_pdf(page_url, js_function, iframe_selector, pdf_xpath, save_path):
    page_url = "https://www.bokjiro.go.kr/ssis-tbu/twatxa/wlfarePr/selectWlfareSubMain.do"
    js_function = 'fnSideMenu("MTWAT00072")'  # JavaScript 함수 호출
    iframe_selector = 'iframe[title="안내책자"]'
    pdf_xpath = '//a[contains(@title, "장애인")]'
    
    save_path = "2024_장애인_안내책자.pdf"

    # Selenium WebDriver 설정
    driver = webdriver.Chrome()
    driver.get(page_url)

    # JavaScript 실행
    driver.execute_script(js_function)

    # iframe 전환 대기 (iframe이 로드될 때까지 대기)
    iframe_locator = (By.CSS_SELECTOR, iframe_selector)  # iframe의 title 속성 기준
    WebDriverWait(driver, 10).until(EC.presence_of_element_located(iframe_locator))

    # iframe 전환
    iframe_element = driver.find_element(*iframe_locator)
    driver.switch_to.frame(iframe_element)

    try:
        download_link_element = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, pdf_xpath))
        )
        download_url = download_link_element.get_attribute("href")
        response = requests.get(download_url, stream=True)

        with open(save_path, "wb") as file:
            file.write(response.content)

    except Exception as e:
        print(f"요소를 찾는 중 오류 발생: {e}")

    driver.switch_to.default_content()

    driver.quit()


# 실행 예제
page_url = "https://www.bokjiro.go.kr/ssis-tbu/twatxa/wlfarePr/selectWlfareSubMain.do"
js_function = 'fnSideMenu("MTWAT00072")'  # JavaScript 함수 호출
iframe_selector = 'iframe[title="안내책자"]'
pdf_xpath = '//a[contains(@title, "장애인")]'

save_path = "2024_장애인_안내책자.pdf"

bokjiro_pdf(page_url, js_function, iframe_selector, pdf_xpath, save_path)